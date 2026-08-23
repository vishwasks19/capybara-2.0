"""
End-to-end SAR Oil Spill Segmentation Pipeline
================================================
Pretrained U-Net (ResNet34 encoder, ImageNet weights) via segmentation-models-pytorch.

Pipeline stages (all in this one script):
    1. Dataset loading (images + binary masks)
    2. Train / val split
    3. Model definition (pretrained U-Net + ResNet34 encoder)
    4. Training loop with Dice + BCE loss
    5. Validation (IoU / Dice metrics)
    6. Inference on a single image -> saved predicted mask + overlay

Expected data layout (works with the Kaggle "Deep-SAR SOS" dataset or similar):

    data/
      images/   *.png / *.jpg / *.tif   (SAR image, grayscale or RGB)
      masks/    *.png                    (binary mask, same filename as image,
                                           0 = background/sea, 255 or 1 = oil spill)

Install deps:
    pip install torch torchvision segmentation-models-pytorch albumentations opencv-python --break-system-packages

Usage:
    # Train
    python oil_spill_segmentation.py train --data_dir ./data --epochs 20 --out_dir ./checkpoints

    # Run inference on a single image with a trained checkpoint
    python oil_spill_segmentation.py predict --checkpoint ./checkpoints/best_model.pth --image ./data/images/sample.png --out_dir ./predictions
"""

import os
import argparse
import glob
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation-models-pytorch not installed. Run:\n"
        "  pip install segmentation-models-pytorch --break-system-packages"
    )

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBU = True
except ImportError:
    HAS_ALBU = False

IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------
class OilSpillDataset(Dataset):
    """Loads (image, binary mask) pairs from an images/ and masks/ folder.

    Assumes each image in `images/` has a same-named (or same-stem) mask in `masks/`.
    Masks are binarized: any pixel > 0 becomes class 1 (oil spill).
    """

    def __init__(self, images_dir, masks_dir, file_list=None, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        all_images = []
        for e in exts:
            all_images.extend(glob.glob(os.path.join(images_dir, e)))
        self.image_paths = sorted(file_list if file_list else all_images)
        self.augment = augment

        if HAS_ALBU:
            if augment:
                self.transform = A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_paths)

    def _find_mask_path(self, image_path):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = os.path.join(self.masks_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No mask found for image {image_path}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self._find_mask_path(img_path)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # binarize: 1 = oil spill

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = torch.tensor(mask, dtype=torch.float32)

        mask = mask.unsqueeze(0)  # shape [1, H, W]
        return image, mask


# ----------------------------------------------------------------------
# 2. Model: pretrained U-Net with ResNet34 encoder
# ----------------------------------------------------------------------
def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,  # raw logits; apply sigmoid manually
    )
    return model.to(DEVICE)


# ----------------------------------------------------------------------
# 3. Loss + metrics
# ----------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        return bce_loss + dice_loss


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ----------------------------------------------------------------------
# 4. Training loop
# ----------------------------------------------------------------------
def train(args):
    images_dir = os.path.join(args.data_dir, "images")
    masks_dir = os.path.join(args.data_dir, "masks")

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    all_images = []
    for e in exts:
        all_images.extend(glob.glob(os.path.join(images_dir, e)))
    all_images = sorted(all_images)

    if len(all_images) == 0:
        raise RuntimeError(f"No images found in {images_dir}")

    random.seed(42)
    random.shuffle(all_images)
    split = int(0.85 * len(all_images))
    train_files, val_files = all_images[:split], all_images[split:]

    train_ds = OilSpillDataset(images_dir, masks_dir, file_list=train_files, augment=True)
    val_ds = OilSpillDataset(images_dir, masks_dir, file_list=val_files, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model()
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)
                val_iou += iou_score(logits, masks) * images.size(0)
        val_loss /= max(len(val_ds), 1)
        val_iou /= max(len(val_ds), 1)

        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_IoU={val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved new best checkpoint: {ckpt_path}")

    print("Training complete.")


# ----------------------------------------------------------------------
# 5. Inference on a single image
# ----------------------------------------------------------------------
def predict(args):
    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    if HAS_ALBU:
        transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        input_tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
    else:
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0
        input_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

    binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
    binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]

    mask_path = os.path.join(args.out_dir, f"{base}_mask.png")
    cv2.imwrite(mask_path, binary_mask)

    # overlay: red = predicted oil spill
    overlay = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    overlay[binary_mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.7, overlay, 0.3, 0)
    overlay_path = os.path.join(args.out_dir, f"{base}_overlay.png")
    cv2.imwrite(overlay_path, blended)

    print(f"Saved mask -> {mask_path}")
    print(f"Saved overlay -> {overlay_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SAR Oil Spill Segmentation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the segmentation model")
    train_parser.add_argument("--data_dir", type=str, required=True,
                               help="Directory containing images/ and masks/ subfolders")
    train_parser.add_argument("--out_dir", type=str, default="./checkpoints")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-4)

    predict_parser = subparsers.add_parser("predict", help="Run inference on a single image")
    predict_parser.add_argument("--checkpoint", type=str, required=True)
    predict_parser.add_argument("--image", type=str, required=True)
    predict_parser.add_argument("--out_dir", type=str, default="./predictions")

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()
