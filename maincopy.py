from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
import io
import base64
from torchvision import transforms

app = Flask(__name__)

# Add CORS for web access
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": "*"}})

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["benign", "malignant", "normal"]

# Image transform (same as training)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

class HybridDenseNet(nn.Module):
    def __init__(self, num_classes=3, backbone_name="densenet121", pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        
        if backbone_name == "densenet121":
            backbone = models.densenet121(pretrained=pretrained)
            in_features = 1024
        elif backbone_name == "densenet169":
            backbone = models.densenet169(pretrained=pretrained)
            in_features = 1664
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = backbone.features
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.seg_head = SegmentationHead(in_channels=in_features, out_channels=1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.relu(feats)
        pooled = self.avgpool(feats)
        pooled = torch.flatten(pooled, 1)
        class_out = self.classifier(pooled)
        seg_out = self.seg_head(feats)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return class_out, seg_out

def predict_tumor(image_b64: str, pixel_spacing: float = None):
    try:
        model = HybridDenseNet(num_classes=3, backbone_name="densenet121", pretrained=False).to(device)
        state_dict = torch.load("hybrid_breast_cancer_desnet(3).pth", map_location=device)
        model.load_state_dict(state_dict)
        if model is None:
            return jsonify({"error": "Model not loaded"}), 503
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        return jsonify({"error": f"Model failed: {str(e)}"}), 500

    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        if isinstance(outputs, tuple):
            class_out, seg_out = outputs
        else:
            class_out, seg_out = outputs, None

        probs = F.softmax(class_out, dim=1)
        top_prob, top_class = torch.max(probs, 1)
        predicted_class = class_names[top_class.item()]
        predicted_prob = float(top_prob.item())

        bbox = None
        length_px = diameter_px = None
        length_mm = diameter_mm = None
        annotated_b64 = None

        if seg_out is not None:
            seg_mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
            seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                resize_w, resize_h = seg_mask.shape[1], seg_mask.shape[0]
                scale_x = orig_w / resize_w
                scale_y = orig_h / resize_h
                x_orig, y_orig = int(x * scale_x), int(y * scale_y)
                w_orig, h_orig = int(w * scale_x), int(h * scale_y)
                bbox = [x_orig, y_orig, x_orig + w_orig, y_orig + h_orig]
                length_px = max(w_orig, h_orig)
                diameter_px = min(w_orig, h_orig)

                if pixel_spacing is not None:
                    if isinstance(pixel_spacing, (int, float)):
                        spacing_x = spacing_y = float(pixel_spacing)
                    else:
                        spacing_x, spacing_y = pixel_spacing
                    length_mm = length_px * spacing_x
                    diameter_mm = diameter_px * spacing_y

                img_cv = np.array(image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img_cv, (x_orig, y_orig), (x_orig+w_orig, y_orig+h_orig), (0, 255, 0), 2)
                _, buffer = cv2.imencode(".png", img_cv)
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "class": predicted_class,
        "probability": predicted_prob,
        "bbox": bbox,
        "length_px": length_px if bbox else None,
        "diameter_px": diameter_px if bbox else None,
        "length_mm": length_mm,
        "diameter_mm": diameter_mm,
        "image_size": [orig_w, orig_h],
        "annotated_image": annotated_b64
    }

items = []

@app.route("/", methods=["GET"])
def root():
    return jsonify({"Hello": "world"})

# @app.route("/items", methods=["POST"])
# def create_item():
#     item = request.args.get("item")
#     if not item:
#         return jsonify({"error": "Item parameter is required"}), 400
#     items.append(item)
#     return jsonify(item)

# @app.route("/items/<int:item_id>", methods=["GET"])
# def get_item(item_id):
#     try:
#         item = items[item_id]
#         return jsonify(item)
#     except IndexError:
#         return jsonify({"error": "Item not found"}), 404

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        image_b64 = request.form.get("image_b64")
        pixel_spacing = float(request.form.get("pixel_spacing", 0.5))
        if not image_b64:
            return jsonify({"error": "image_b64 is required"}), 400
        result = predict_tumor(image_b64, pixel_spacing)
        print(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)