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
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add CORS for web access
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": "*"}})

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")
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
        self.conv2 = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.conv2(x)

class HybridResNet(nn.Module):
    def __init__(self, num_classes=3, backbone_name="resnet18", pretrained=True):
        super().__init__()
        backbone_fn = getattr(models, backbone_name)
        backbone = backbone_fn(pretrained=pretrained)

        # Feature extractor (remove avgpool + fc)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        in_features = 512  # ResNet18's output features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Segmentation head
        self.seg_head = SegmentationHead(in_channels=in_features, out_channels=1)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.avgpool(feats)
        pooled = torch.flatten(pooled, 1)
        class_out = self.classifier(pooled)
        seg_out = self.seg_head(feats)
        seg_out = F.interpolate(seg_out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return class_out, seg_out

def predict_tumor(image_b64: str, pixel_spacing: float = None):
    logger.debug("Starting predict_tumor")
    try:
        model = HybridResNet(num_classes=3, backbone_name="resnet18", pretrained=False).to(device)
        model_path = os.path.join(os.path.dirname(__file__), "hybrid_breast_cancer2.pth")
        logger.debug(f"Loading model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise Exception(f"Model failed: {str(e)}")

    try:
        logger.debug("Decoding image")
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        logger.debug(f"Image loaded: {orig_w}x{orig_h}")
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise Exception(f"Invalid image data: {str(e)}")

    input_tensor = image_transform(image).unsqueeze(0).to(device)
    logger.debug("Image transformed and moved to device")

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
        logger.debug(f"Prediction: {predicted_class} ({predicted_prob:.4f})")

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
                logger.debug("Segmentation and bounding box generated")

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

@app.route("/", methods=["GET"], strict_slashes=False)
def root():
    logger.debug("Handling GET /")
    return jsonify({"Hello": "world"})

@app.route("/items", methods=["POST"], strict_slashes=False)
def create_item():
    logger.debug("Handling POST /items")
    item = request.args.get("item")
    if not item:
        logger.warning("Item parameter missing")
        return jsonify({"error": "Item parameter is required"}), 400
    items.append(item)
    logger.debug(f"Item added: {item}")
    return jsonify(item)

@app.route("/items/<int:item_id>", methods=["GET"], strict_slashes=False)
def get_item(item_id):
    logger.debug(f"Handling GET /items/{item_id}")
    try:
        item = items[item_id]
        logger.debug(f"Item retrieved: {item}")
        return jsonify(item)
    except IndexError:
        logger.warning(f"Item {item_id} not found")
        return jsonify({"error": "Item not found"}), 404

@app.route("/predict", methods=["POST"], strict_slashes=False)
def predict_api():
    logger.debug("Handling POST /predict")
    try:
        image_b64 = request.form.get("image_b64")
        pixel_spacing = float(request.form.get("pixel_spacing", 0.5))
        logger.debug(f"Received image_b64 (len={len(image_b64) if image_b64 else 0}), pixel_spacing={pixel_spacing}")
        if not image_b64:
            logger.warning("image_b64 is required")
            return jsonify({"error": "image_b64 is required"}), 400
        result = predict_tumor(image_b64, pixel_spacing)
        logger.info(f"Prediction result: {result}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(debug=True)