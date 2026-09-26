import os
import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import SegformerForSemanticSegmentation


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "nvidia/segformer-b1-finetuned-ade-512-512"

CHECKPOINT_PATH = (
    r"C:\Users\DELL\Downloads\capybara 2.0"
    r"\checkpoints\best_segformer_oil_spill.pth"
)

IMG_SIZE = 256

# Your model has:
# Class 0 -> Background
# Class 1 -> Oil Spill
NUM_CLASSES = 2
OIL_CLASS = 1

# Pixel-level segmentation threshold
THRESHOLD = 0.5

# Minimum percentage of the image that must be
# classified as oil spill before we report detection.
DETECTION_PERCENTAGE = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CrudeControl Oil Spill Detection API",
    description="SegFormer-based oil spill segmentation and detection API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("==========================================")
print("     CRUDECONTROL MODEL INITIALIZATION")
print("==========================================")
print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print()

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


print("Creating SegFormer B1 model...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

print("Base model created.")
print("Loading trained checkpoint...")


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


# Some checkpoints are stored as:
# {
#     "state_dict": {...}
# }
#
# Your checkpoint is a direct state_dict, but this
# makes the backend compatible with either format.

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint


# ------------------------------------------------------------
# Remove DataParallel prefix if present
# ------------------------------------------------------------

state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}


# ------------------------------------------------------------
# Load trained weights
# ------------------------------------------------------------

model.load_state_dict(
    state_dict,
    strict=True
)


# ------------------------------------------------------------
# Move model to device
# ------------------------------------------------------------

model.to(DEVICE)

model.eval()


print()
print("==========================================")
print("       MODEL LOADED SUCCESSFULLY")
print("==========================================")
print(f"Model: SegFormer B1")
print(f"Classes: {NUM_CLASSES}")
print(f"Oil class: {OIL_CLASS}")
print(f"Input resolution: {IMG_SIZE}x{IMG_SIZE}")
print(f"Segmentation threshold: {THRESHOLD}")
print(f"Detection percentage: {DETECTION_PERCENTAGE}%")
print(f"Device: {DEVICE}")
print("==========================================")
print()


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

# Same ImageNet normalization used during training.

IMAGE_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGE_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image):

    # Save original dimensions
    original_width, original_height = image.size

    # Ensure RGB
    image = image.convert("RGB")

    # Resize to training resolution
    image_resized = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    # Convert to NumPy
    image_array = np.asarray(
        image_resized,
        dtype=np.float32
    )

    # Scale from [0, 255] -> [0, 1]
    image_array = image_array / 255.0

    # ImageNet normalization
    image_array = (
        image_array - IMAGE_MEAN
    ) / IMAGE_STD

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # NumPy -> PyTorch
    tensor = torch.from_numpy(
        image_array
    ).float()

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move to CPU/GPU
    tensor = tensor.to(DEVICE)

    return (
        tensor,
        original_width,
        original_height
    )


# ============================================================
# OIL SPILL PREDICTION
# ============================================================

@torch.no_grad()
def predict(image: Image.Image):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    tensor, original_width, original_height = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    outputs = model(
        pixel_values=tensor
    )

    logits = outputs.logits

    # --------------------------------------------------------
    # Resize logits to input resolution
    # --------------------------------------------------------

    logits = F.interpolate(
        logits,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = F.softmax(
        logits,
        dim=1
    )

    # --------------------------------------------------------
    # Extract oil-spill probability
    #
    # Class 0 = Background
    # Class 1 = Oil Spill
    # --------------------------------------------------------

    oil_probability = probabilities[
        0,
        OIL_CLASS
    ]

    # --------------------------------------------------------
    # Apply segmentation threshold
    # --------------------------------------------------------

    mask = (
        oil_probability >= THRESHOLD
    ).cpu().numpy().astype(np.uint8)

    # --------------------------------------------------------
    # Resize mask back to original image dimensions
    # --------------------------------------------------------

    mask_original = cv2.resize(
        mask,
        (
            original_width,
            original_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Calculate oil spill percentage
    # --------------------------------------------------------

    oil_pixels = int(
        np.sum(mask_original == 1)
    )

    total_pixels = int(
        mask_original.size
    )

    oil_percentage = (
        oil_pixels / total_pixels
    ) * 100.0

    # Explicit Python float
    oil_percentage = float(
        oil_percentage
    )

    return (
        mask_original,
        oil_percentage
    )


# ============================================================
# CONVERT IMAGE TO BASE64
# ============================================================

def image_to_base64(image: Image.Image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "CrudeControl Oil Spill Detection API",
        "model": "SegFormer B1",
        "classes": NUM_CLASSES,
        "oil_class": OIL_CLASS,
        "device": str(DEVICE)
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "SegFormer B1",
        "device": str(DEVICE)
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:
        return {
            "error": "Uploaded file is empty."
        }

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        return {
            "error": "The uploaded file is not a valid image."
        }

    # --------------------------------------------------------
    # Run model
    # --------------------------------------------------------

    mask, oil_percentage = predict(
        image
    )

    # ========================================================
    # CREATE MASK IMAGE
    # ========================================================

    mask_image = Image.fromarray(
        (
            mask * 255
        ).astype(np.uint8)
    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    original = np.array(
        image
    ).copy()

    overlay = original.copy()

    # Find oil-spill pixels
    oil_region = mask == 1

    # Mark detected oil as red
    overlay[oil_region] = (
        255,
        0,
        0
    )

    # Blend original image and detection overlay
    blended = cv2.addWeighted(
        original,
        0.6,
        overlay,
        0.4,
        0
    )

    overlay_image = Image.fromarray(
        blended
    )


    # ========================================================
    # DETECTION DECISION
    # ========================================================

    # IMPORTANT:
    # Convert NumPy boolean to native Python bool.
    #
    # Without bool(), FastAPI can throw:
    #
    # TypeError: 'numpy.bool' object is not iterable
    #
    oil_detected = bool(
        oil_percentage >= DETECTION_PERCENTAGE
    )


    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return {
        "filename": file.filename,

        "oil_spill_detected": oil_detected,

        "oil_spill_percentage": float(
            round(
                oil_percentage,
                2
            )
        ),

        "segmentation_threshold": float(
            THRESHOLD
        ),

        "detection_percentage_threshold": float(
            DETECTION_PERCENTAGE
        ),

        "mask": image_to_base64(
            mask_image
        ),

        "overlay": image_to_base64(
            overlay_image
        )
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
