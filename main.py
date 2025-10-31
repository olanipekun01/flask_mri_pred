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
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": "*"}})

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")

# Class names
class_names = ["benign", "malignant", "normal"]

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =============================
# Custom Model Components
# =============================

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w

class DepthwiseSeparable(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return F.relu(self.bn(x), inplace=True)

class EdgeStem(nn.Module):
    def __init__(self, out_c=64):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparable(2, 16, 3, 1, 1),
            DepthwiseSeparable(16, 32, 3, 2, 1),
            DepthwiseSeparable(32, out_c, 3, 2, 1)
        )
    def forward(self, sobel_xy):
        return self.block(sobel_xy)

class MGCR(nn.Module):
    def __init__(self, c_rgb, c_edge):
        super().__init__()
        mid = max(16, c_rgb // 8)
        self.rgb_to_edge = nn.Sequential(nn.Conv2d(c_rgb, mid, 1), nn.ReLU(True), nn.Conv2d(mid, c_edge, 1), nn.Sigmoid())
        self.edge_to_rgb = nn.Sequential(nn.Conv2d(c_edge, mid, 1), nn.ReLU(True), nn.Conv2d(mid, c_rgb, 1), nn.Sigmoid())
        self.align_edge = nn.Conv2d(c_edge, c_rgb, 1)
    def forward(self, frgb, fedge):
        g_r2e = self.rgb_to_edge(frgb)
        g_e2r = self.edge_to_rgb(fedge)
        e_mod = fedge * g_r2e
        r_mod = frgb * g_e2r
        frgb_new = frgb + self.align_edge(e_mod)
        fedge_new = fedge + r_mod.mean(dim=1, keepdim=True)
        return frgb_new, fedge_new

class SpectralGateResidual(nn.Module):
    def __init__(self, channels, init_lowpass=0.7):
        super().__init__()
        self.alpha_low = nn.Parameter(torch.ones(channels) * init_lowpass)
        self.alpha_high = nn.Parameter(torch.ones(channels) * (1.0 - init_lowpass))
    def forward(self, x):
        B, C, H, W = x.shape
        Xf = torch.fft.rfft2(x, norm='ortho')
        yy = torch.linspace(-1, 1, H, device=x.device).unsqueeze(1).expand(H, W)
        xx = torch.linspace(-1, 1, W, device=x.device).unsqueeze(0).expand(H, W)
        rr = torch.sqrt(xx**2 + yy**2)[:, :Xf.shape[-1]]
        low = (1.0 / (1.0 + rr * 8.0)).unsqueeze(0).unsqueeze(0)
        high = 1.0 - low
        aL = self.alpha_low.view(1, C, 1, 1)
        aH = self.alpha_high.view(1, C, 1, 1)
        Wf = aL * low + aH * high
        Xf_mod = Xf * Wf
        x_mod = torch.fft.irfft2(Xf_mod, s=(H, W), norm='ortho')
        return x + x_mod

class TinyFPN(nn.Module):
    def __init__(self, c2=64, c3=128, c4=256, c5=512, p=128):
        super().__init__()
        self.l2 = nn.Conv2d(c2, p, 1)
        self.l3 = nn.Conv2d(c3, p, 1)
        self.l4 = nn.Conv2d(c4, p, 1)
        self.l5 = nn.Conv2d(c5, p, 1)
        self.s2 = nn.Conv2d(p, p, 3, padding=1)
    def forward(self, c2, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='nearest')
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        return self.s2(p2)

class SegHead(nn.Module):
    def __init__(self, in_c=128):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparable(in_c, in_c, 3, 1, 1),
            nn.Conv2d(in_c, 1, 1)
        )
    def forward(self, p2, img_hw):
        seg = self.block(p2)
        return F.interpolate(seg, size=img_hw, mode='bilinear', align_corners=True)

class ClsHeadMaskGuided(nn.Module):
    def __init__(self, in_c=128, hidden=256, num_classes=3):
        super().__init__()
        self.proj = nn.Conv2d(in_c, hidden, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, p2, seg_logits, g_cls):
        att = torch.sigmoid(seg_logits)
        att_resized = F.interpolate(att, size=p2.shape[2:], mode='bilinear', align_corners=False)
        att_guided = att_resized * g_cls.view(-1, 1, 1, 1)
        z = self.proj(p2 * att_guided)
        v = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(v)

class TaskAgreementController(nn.Module):
    def __init__(self, in_c=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, 128),
            nn.ReLU(True),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
    def forward(self, x):
        g = self.fc(x)
        return g[:, 0], g[:, 1]

# =============================
# Main Model
# =============================

class HyRes18_MGCR_SGR_TAC(nn.Module):
    def __init__(self, num_classes=3, load_resnet18_weights=None, dilate_last=True):
        super().__init__()
        base = models.resnet18(weights=None)
        if dilate_last:
            for name, module in base.layer4.named_modules():
                if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                    if module.stride == (2, 2):
                        module.stride = (1, 1)
                    module.dilation = (2, 2)
                    module.padding = module.dilation
                elif isinstance(module, models.resnet.BasicBlock):
                    if module.downsample is not None and isinstance(module.downsample[0], nn.Conv2d):
                        if module.downsample[0].stride == (2, 2):
                            module.downsample[0].stride = (1, 1)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        if load_resnet18_weights is not None:
            sd = torch.load(load_resnet18_weights, map_location='cpu')
            base.load_state_dict(sd, strict=False)
            with torch.no_grad():
                self.stem[0].weight.copy_(base.conv1.weight)
                self.stem[1].weight.copy_(base.bn1.weight)
                self.stem[1].bias.copy_(base.bn1.bias)
            self.layer1.load_state_dict(base.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(base.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(base.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(base.layer4.state_dict(), strict=False)

        self.edge_stem_1_4 = EdgeStem(out_c=64)
        self.mgcr1 = MGCR(64, 64)
        self.mgcr2 = MGCR(128, 128)
        self.edge_reducer = nn.Conv2d(64, 128, 1)
        self.sgr3 = SpectralGateResidual(256)
        self.sgr4 = SpectralGateResidual(512)
        self.se4 = SE(512)
        self.fpn = TinyFPN(64, 128, 256, 512, p=128)
        self.tac = TaskAgreementController(in_c=512)
        self.seg_head = SegHead(in_c=128)
        self.cls_head = ClsHeadMaskGuided(in_c=128, hidden=256, num_classes=num_classes)

    @staticmethod
    def _sobel_xy(img):
        if img.shape[1] == 3:
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        else:
            gray = img
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sx = F.conv2d(gray, kx, padding=1)
        sy = F.conv2d(gray, ky, padding=1)
        return torch.tanh(torch.cat([sx, sy], dim=1))

    def forward(self, x):
        B, C, H, W = x.shape
        sob = self._sobel_xy(x)
        e14 = self.edge_stem_1_4(sob)
        z = self.stem(x)
        c2 = self.layer1(z)
        c2, e2 = self.mgcr1(c2, e14)
        c3 = self.layer2(c2)
        e3 = F.interpolate(e2, size=c3.shape[2:], mode='bilinear', align_corners=False)
        e3 = self.edge_reducer(e3)
        c3, _ = self.mgcr2(c3, e3)
        c4 = self.layer3(c3)
        c4 = self.sgr3(c4)
        c5 = self.layer4(c4)
        c5 = self.sgr4(c5)
        c5 = self.se4(c5)
        g_seg, g_cls = self.tac(c5)
        p2 = self.fpn(c2, c3, c4, c5)
        seg_out = self.seg_head(p2, img_hw=(H, W)) * g_seg.view(-1, 1, 1, 1)
        class_out = self.cls_head(p2, seg_out.detach(), g_cls)
        return class_out, seg_out


# =============================
# Load Model (Global)
# =============================

model = None

def load_model():
    global model
    model = HyRes18_MGCR_SGR_TAC(num_classes=3, load_resnet18_weights=None, dilate_last=True).to(device)

    # Find checkpoint
    preferred = ["hyres18_best.pth", "hyres18.pth", "hyres18_best.pt", "hyres18.pt"]
    ckpt_path = None
    for p in preferred:
        if os.path.exists(p):
            ckpt_path = p
            break
    if ckpt_path is None:
        candidates = [f for f in os.listdir() if f.endswith(('.pth', '.pt'))]
        for c in candidates:
            if any(x in c.lower() for x in ['hyres', 'res18']):
                ckpt_path = c
                break
        if ckpt_path is None and candidates:
            ckpt_path = candidates[0]

    if not ckpt_path:
        raise FileNotFoundError("No .pth or .pt checkpoint found. Place your model file in the app directory.")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Extract state_dict
    sd = ckpt
    if isinstance(ckpt, dict):
        sd = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt

    # Strip 'module.'
    new_sd = {}
    for k, v in sd.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_sd[new_k] = v

    # Load with fallback
    try:
        model.load_state_dict(new_sd)
        logger.info("Model loaded successfully")
    except Exception as e1:
        logger.warning(f"Direct load failed: {e1}")
        try:
            model.load_state_dict(new_sd, strict=False)
            logger.info("Model loaded with strict=False")
        except Exception as e2:
            logger.warning(f"Partial load failed: {e2}")
            model_sd = model.state_dict()
            filtered = {k: v for k, v in new_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
            model.load_state_dict(filtered, strict=False)
            logger.info(f"Loaded {len(filtered)} matching parameters")

    model.eval()
    logger.info("Model is ready for inference")

# Load on startup
try:
    load_model()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None


# =============================
# Prediction Function
# =============================

def predict_tumor(image_b64: str, pixel_spacing: float = None):
    if model is None:
        raise Exception("Model not loaded")

    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = image.size
        logger.debug(f"Image loaded: {orig_w}x{orig_h}")
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        raise Exception(f"Invalid image: {e}")

    input_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_out, seg_out = model(input_tensor)
        probs = F.softmax(class_out, dim=1)
        top_prob, top_class = torch.max(probs, 1)
        predicted_class = class_names[top_class.item()]
        predicted_prob = float(top_prob.item())

        bbox = length_px = diameter_px = length_mm = diameter_mm = None
        annotated_b64 = None

        if seg_out is not None:
            seg_mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
            seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                scale_x = orig_w / seg_mask.shape[1]
                scale_y = orig_h / seg_mask.shape[0]
                x_orig = int(x * scale_x)
                y_orig = int(y * scale_y)
                w_orig = int(w * scale_x)
                h_orig = int(h * scale_y)
                bbox = [x_orig, y_orig, x_orig + w_orig, y_orig + h_orig]
                length_px = max(w_orig, h_orig)
                diameter_px = min(w_orig, h_orig)

                if pixel_spacing is not None:
                    spacing_x = spacing_y = float(pixel_spacing) if isinstance(pixel_spacing, (int, float)) else pixel_spacing
                    length_mm = length_px * spacing_x
                    diameter_mm = diameter_px * spacing_y

                img_cv = np.array(image)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                cv2.rectangle(img_cv, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)
                _, buffer = cv2.imencode(".png", img_cv)
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "class": predicted_class,
        "probability": round(predicted_prob, 4),
        "bbox": bbox,
        "length_px": length_px,
        "diameter_px": diameter_px,
        "length_mm": round(length_mm, 2) if length_mm else None,
        "diameter_mm": round(diameter_mm, 2) if diameter_mm else None,
        "image_size": [orig_w, orig_h],
        "annotated_image": annotated_b64
    }


# =============================
# Flask Routes
# =============================

items = []

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "HyRes18 Tumor API Ready"})

@app.route("/items", methods=["POST"])
def create_item():
    item = request.args.get("item")
    if not item:
        return jsonify({"error": "Item parameter required"}), 400
    items.append(item)
    return jsonify({"item": item, "id": len(items)-1})

@app.route("/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    try:
        return jsonify({"item": items[item_id]})
    except IndexError:
        return jsonify({"error": "Item not found"}), 404

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        image_b64 = request.form.get("image_b64")
        pixel_spacing = request.form.get("pixel_spacing")
        pixel_spacing = float(pixel_spacing) if pixel_spacing else None

        if not image_b64:
            return jsonify({"error": "image_b64 is required"}), 400

        result = predict_tumor(image_b64, pixel_spacing)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)