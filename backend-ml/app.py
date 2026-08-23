import sys
import os
import base64

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_model.pth"
)


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 256


# --------------------------------------------------
# Load model ONCE when server starts
# --------------------------------------------------

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()


print("Model loaded successfully!")
print("Checkpoint:", CHECKPOINT_PATH)
print("Device:", DEVICE)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "FastAPI is working!"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return {
            "error": "Invalid image file"
        }

    # Keep original dimensions
    original_height, original_width = image.shape[:2]

    # Convert BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize exactly like the training script
    resized = resized.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    resized = (resized - mean) / std

    # HWC -> CHW
    tensor = torch.tensor(
        resized,
        dtype=torch.float32
    ).permute(2, 0, 1)

    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(DEVICE)

    # Run model
    with torch.no_grad():
        logits = model(tensor)

        probabilities = torch.sigmoid(logits)

        probability_mask = probabilities[
            0, 0
        ].cpu().numpy()

    # Convert probability -> binary mask
    binary_mask = (
        probability_mask > 0.5
    ).astype(np.uint8) * 255

    # Resize mask back to original image size
    binary_mask = cv2.resize(
        binary_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------
    # Create overlay
    # --------------------------------------------------

    overlay = image.copy()

    # Red color for predicted oil spill
    overlay[binary_mask > 0] = (0, 0, 255)

    blended = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    # --------------------------------------------------
    # Convert outputs to base64
    # --------------------------------------------------

    _, mask_encoded = cv2.imencode(
        ".png",
        binary_mask
    )

    _, overlay_encoded = cv2.imencode(
        ".png",
        blended
    )

    mask_base64 = base64.b64encode(
        mask_encoded.tobytes()
    ).decode("utf-8")

    overlay_base64 = base64.b64encode(
        overlay_encoded.tobytes()
    ).decode("utf-8")

    return {
        "filename": file.filename,
        "mask": mask_base64,
        "overlay": overlay_base64
    }