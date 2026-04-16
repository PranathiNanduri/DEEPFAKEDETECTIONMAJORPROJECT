from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from explainability.gradcam_visualization import (
    generate_gradcam,
    overlay_heatmap_on_image
)

MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "spatial_model.pth"

HEATMAP_DIR = PROJECT_ROOT / "data" / "heatmaps"
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CLASSES = ["fake", "real", "spoof"]


# ==========================================
# PREPROCESS
# ==========================================
def preprocess(face_bgr):

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    return tfm(rgb).unsqueeze(0)


# ==========================================
# LOAD MODEL
# ==========================================
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = models.resnet18(weights=None)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        3
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            class_names = checkpoint.get(
                "class_names",
                DEFAULT_CLASSES
            )

        else:

            model.load_state_dict(
                checkpoint
            )

            class_names = DEFAULT_CLASSES

    else:

        model.load_state_dict(
            checkpoint
        )

        class_names = DEFAULT_CLASSES

    model = model.to(DEVICE)

    model.eval()

    return model, class_names


# ==========================================
# GRADCAM TARGET
# ==========================================
def get_gradcam_target_layer(model):

    return model.layer4[-1]


# ==========================================
# INFERENCE
# ==========================================
def infer_spatial(
    face_bgr,
    save_heatmap_name=None
):

    model, class_names = load_model()

    x = preprocess(face_bgr).to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))

    pred_label = class_names[pred_idx]

    confidence = float(probs[pred_idx])

    heatmap_path = None

    if save_heatmap_name:

        try:

            cam = generate_gradcam(
                model,
                x,
                get_gradcam_target_layer(model),
                class_idx=pred_idx
            )

            heatmap_path = HEATMAP_DIR / f"{save_heatmap_name}.jpg"

            overlay_heatmap_on_image(
                face_bgr,
                cam,
                save_path=heatmap_path
            )

        except Exception as e:

            print(f"GradCAM Error: {e}")

    return {
        "label": pred_label,
        "confidence": confidence,
        "probs": probs.tolist(),
        "class_names": class_names,
        "heatmap_path": str(heatmap_path) if heatmap_path else None
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)

    args = parser.parse_args()

    img = cv2.imread(args.image)

    result = infer_spatial(img)

    print(result)