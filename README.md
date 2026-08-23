# Oil Spill Detection & Vessel Attribution

A software-only pipeline that detects oil spills from SAR satellite imagery, forecasts/hindcasts how the slick drifts, and attributes the spill to a likely source vessel using historic AIS data.

Built as a hackathon demo (SIH-style) — 2-day rough demo, 15-day full build.

## Overview

1. **Detection** — A SAR image is fed into a segmentation model that outputs a pixel-level oil spill mask.
2. **Drift modeling** — The spill's geometry and age are used to hindcast (backtrack) its likely origin point and forecast its future spread, using oceanographic/meteorological data via [OpenDrift](https://opendrift.github.io/).
3. **Attribution** — Historic AIS vessel tracks near the hindcast origin are scored on proximity, trajectory match, and behavioral anomalies (e.g. AIS signal loss, erratic course changes) to rank the most likely source vessel.
4. **UI** — A web app lets a user upload a SAR image and see the detected spill overlay and the top vessel match together, automatically, in one flow.

## Architecture

```
SAR image
   |
   v
[Segmentation model]  ---->  spill mask + geometry
   |
   v
[OpenDrift hindcast/forecast]  ---->  origin point + drift trajectory
   |
   v
[AIS correlation & scoring]  ---->  ranked vessel candidates
   |
   v
[Frontend]  ---->  overlay + top vessel match shown together
```

## Tech Stack

| Layer | Tech |
|---|---|
| Segmentation model | PyTorch, `segmentation-models-pytorch` (U-Net, pretrained ResNet encoder) |
| Drift modeling | OpenDrift |
| Backend (model serving) | FastAPI |
| Backend (attribution/AIS) | Node.js |
| Frontend | React |
| Design | Figma |

## Repo Structure

```
.
├── model/
│   └── oil_spill_segmentation.py   # end-to-end training + inference pipeline
├── backend-ml/                     # FastAPI service wrapping the segmentation model
├── backend-attribution/            # Node service: AIS scoring + vessel attribution
├── frontend/                       # React app
└── data/
    ├── images/                     # SAR image patches
    └── masks/                      # ground-truth binary masks
```

## Getting Started

### 1. Segmentation model

```bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python --break-system-packages
```

Place your dataset as:

```
data/
  images/   # SAR image patches (.png/.jpg/.tif)
  masks/    # binary masks, same filename as matching image
```

Train:

```bash
python model/oil_spill_segmentation.py train --data_dir ./data --epochs 20 --out_dir ./checkpoints
```

Run inference on a single image:

```bash
python model/oil_spill_segmentation.py predict --checkpoint ./checkpoints/best_model.pth --image ./data/images/sample.png --out_dir ./predictions
```

**Dataset used:** [Deep-SAR SOS Oil Spill Detection Dataset](https://www.kaggle.com/datasets/bitsandlayers/sar-oil-spill-segmentation-dataset-sos) (free, Kaggle). Alternative: [Sentinel-1 SAR Oil Spill dataset on Zenodo](https://zenodo.org/records/8346860).

### 2. Backend — model serving

```bash
cd backend-ml
uvicorn app:app --reload
```

Exposes a `/predict` endpoint that accepts an uploaded SAR image and returns the predicted spill mask/overlay.

### 3. Backend — AIS attribution

```bash
cd backend-attribution
npm install
npm start
```

Exposes an `/attribute` endpoint that takes a spill location/time and returns ranked candidate vessels scored on proximity, trajectory match, and AIS behavioral anomalies.

### 4. Frontend

```bash
cd frontend
npm install
npm start
```

Upload a SAR image — detection and vessel attribution run automatically, and both results (spill overlay + top vessel match) are shown together on the results screen.

## Team

| Name | Role |
|---|---|
| Partha | Frontend, UI/UX |
| Pushkar | Backend (Node, AIS attribution logic) |
| Sanju | Design (Figma/Canva) |
| Vishwas | ML engineer (segmentation, drift modeling) |
| Udisha | Presentation |
| Ananya | Backend (model serving) |

## Status

🚧 Hackathon demo in progress.

## License

TBD.
