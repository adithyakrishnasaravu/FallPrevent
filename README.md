# ElderWatch — On-Device Elderly Fall Monitoring Agent

Real-time fall detection and response system using pose estimation, LSTM classification, and fine-tuned Gemma models. All inference runs on-device — no data leaves the room.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Video Stream                          │
│                  (webcam / RTSP)                         │
└──────────────┬──────────────────────────────────────────┘
               │
   ┌───────────▼───────────┐
   │   Loop 1: Perception  │  MediaPipe Pose (33 keypoints)
   │       ~15ms/frame     │  + YOLO bounding box
   └───────────┬───────────┘
               │ keypoint vectors
   ┌───────────▼───────────┐
   │  Loop 2: LSTM Fall    │  Sliding window of 30 frames
   │    Detection ~10ms    │  Trained on URFall dataset
   └───────────┬───────────┘
               │ fall_probability > threshold
   ┌───────────▼───────────────────────────────────────┐
   │           Loop 3: Context Assessment              │
   │                                                    │
   │  Fast Path (< 1s)         Slow Path (3-8s)       │
   │  Gemma 270M               Gemma 3n E4B           │
   │  → alert_caregiver()      → scene description    │
   │  → escalate()             → consciousness check  │
   │  → call_emergency()       → hazard assessment    │
   └───────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup

```bash
cd elderwatch
pip install -r requirements.txt
```

### 2. Download URFall Dataset

```bash
python scripts/download_urfall.py --include-csv
```

### 3. Extract Poses

```bash
python scripts/extract_poses.py \
    --urfall-dir data/urfall \
    --output-dir data/poses
```

### 4. Prepare Training Data

```bash
python scripts/prepare_dataset.py \
    --poses-dir data/poses \
    --output-dir data/processed
```

### 5. Train LSTM

```bash
python scripts/train_lstm.py --config configs/config.yaml
```

### 6. Generate Gemma Fine-tuning Data

```bash
python scripts/generate_gemma_data.py --output data/gemma_270m_finetune.jsonl
```

### 7. Fine-tune Gemma (on GCP with GPU)

```bash
pip install transformers peft datasets accelerate bitsandbytes
python scripts/finetune_gemma.py --data data/gemma_270m_finetune.jsonl
```

### 8. Run Demo

```bash
# With webcam
python scripts/agent_loop.py --source 0

# With video file
python scripts/agent_loop.py --source path/to/video.avi

# With URFall test video
python scripts/agent_loop.py --source data/urfall/fall/fall-01-cam0-rgb.avi
```

### 9. Enable Phone Alerts (SMS)

```bash
cp .env.example .env
# fill TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
```

Set `agent.caregiver_phone` in `configs/config.yaml` (already set in this repo), then run the agent.  
If Twilio credentials are present, `alert_caregiver` sends SMS; every alert also writes a patient log entry.

## Project Structure

```
elderwatch/
├── configs/
│   └── config.yaml            # All hyperparameters and paths
├── scripts/
│   ├── download_urfall.py     # Dataset download
│   ├── extract_poses.py       # MediaPipe pose extraction
│   ├── prepare_dataset.py     # Windowing + augmentation
│   ├── train_lstm.py          # LSTM model + training loop
│   ├── generate_gemma_data.py # Synthetic fine-tuning data
│   ├── finetune_gemma.py      # LoRA fine-tuning on GCP
│   └── agent_loop.py          # Main agent (3 loops)
├── data/                      # Created by scripts
│   ├── urfall/                # Raw videos
│   ├── poses/                 # Extracted keypoints
│   └── processed/             # Train/val/test splits
├── models/                    # Trained model checkpoints
├── requirements.txt
└── README.md
```

## Pipeline Steps

| Step | Script | Input | Output | Time |
|------|--------|-------|--------|------|
| Download | `download_urfall.py` | URL | 70 .avi files | ~10min |
| Poses | `extract_poses.py` | .avi videos | .npy keypoint arrays | ~20min |
| Prepare | `prepare_dataset.py` | .npy arrays | train/val/test .npz | ~1min |
| Train | `train_lstm.py` | .npz splits | .pt model | ~5min (GPU) |
| Gemma Data | `generate_gemma_data.py` | — | .jsonl (300 examples) | instant |
| Fine-tune | `finetune_gemma.py` | .jsonl | LoRA adapter | ~30min (GPU) |

## Key Design Decisions

**Why BiLSTM + Attention?** Falls have a temporal signature — rapid descent followed by stillness. The bidirectional LSTM sees both the approach and the aftermath. Attention lets it focus on the actual fall moment rather than averaging over the whole window.

**Why augment to 500+?** URFall has only 70 sequences. Mirroring, speed variation, noise, and keypoint dropout bring us to enough training data for a small LSTM. The augmentations also simulate real-world conditions (partially occluded subjects, varying fall speeds).

**Why dual Gemma models?** Speed vs. richness tradeoff. The 270M controller fires in <1s for the immediate alert. The 3n VLM takes 3-8s but provides the scene context that makes the alert actionable.

**Why LoRA?** Full fine-tuning of even a 270M model is overkill for 300 examples. LoRA trains in minutes and the adapter is tiny (~2MB), which matters for on-device deployment.

## Test Local Alert + Patient Log

### Quick Local Test (No camera required)

```bash
cd /Users/adithyasaravu/Desktop/code/FallPrevent/intalily-hackathon
python3 scripts/test_local_alert_flow.py
```

Expected:
- Terminal prints an in-app alert message generated in Gemma-style format.
- A new JSON line is appended to `data/patient_events.jsonl`.

### Full Pipeline Test (LSTM + Event Routing)

```bash
cd /Users/adithyasaravu/Desktop/code/FallPrevent/intalily-hackathon
python3 scripts/agent_loop.py --source data/fall-01-cam0.mp4 --config configs/config.yaml
```

With `agent.alert_mode: "local"`, alerts stay in-app and no SMS is attempted.
Score bands are configurable in `configs/config.yaml`:
- `agent.not_fall_max_score` (default `0.6`)
- `agent.possible_fall_max_score` (default `0.8`)

## Hero UI Page

A clean hero page that explains the architecture and simulates the incident flow is available in `ui/`.

Run it locally:

```bash
cd /Users/adithyasaravu/Desktop/code/FallPrevent/intalily-hackathon/ui
python3 -m http.server 8080
```

Then open: `http://localhost:8080`
