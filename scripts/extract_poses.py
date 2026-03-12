"""
Pose Extraction Pipeline
========================
Extracts 33 MediaPipe pose keypoints from every frame of URFall videos.
Saves as numpy arrays for LSTM training.

Uses the NEW MediaPipe Tasks API (PoseLandmarker) which works with
mediapipe >= 0.10.22.

Requires downloading the pose landmarker model:
  wget -O models/pose_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

Output per video: (num_frames, 33, 4) array
  - 33 keypoints
  - 4 values each: x, y, z, visibility
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

NUM_KEYPOINTS = 33
KEYPOINT_DIMS = 4  # x, y, z, visibility

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
DEFAULT_MODEL_PATH = "models/pose_landmarker.task"


def ensure_model(model_path: str = DEFAULT_MODEL_PATH):
    """Download pose landmarker model if not present."""
    if os.path.exists(model_path):
        return model_path

    print(f"Downloading pose landmarker model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Done.")
    return model_path


def extract_poses_from_video(video_path: str,
                              model_path: str = DEFAULT_MODEL_PATH
                              ) -> tuple[np.ndarray, dict]:
    """
    Extract pose keypoints from every frame of a video.

    Returns:
        poses: (num_frames, 33, 4) numpy array
        metadata: dict with video info
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up PoseLandmarker in VIDEO mode
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_poses = []
    frames_with_pose = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Timestamp in milliseconds
            timestamp_ms = int(frame_idx * 1000 / fps) if fps > 0 else frame_idx * 33

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                keypoints = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in landmarks
                ], dtype=np.float32)
                frames_with_pose += 1
            else:
                keypoints = np.zeros((NUM_KEYPOINTS, KEYPOINT_DIMS), dtype=np.float32)

            all_poses.append(keypoints)
            frame_idx += 1

    cap.release()

    poses = np.array(all_poses, dtype=np.float32)

    metadata = {
        "video": str(video_path),
        "fps": fps,
        "total_frames": total_frames,
        "extracted_frames": len(all_poses),
        "frames_with_pose": frames_with_pose,
        "pose_detection_rate": frames_with_pose / max(len(all_poses), 1),
        "resolution": [width, height],
    }

    return poses, metadata


def process_dataset(urfall_dir: str, output_dir: str,
                     model_path: str = DEFAULT_MODEL_PATH):
    """Process all URFall videos and save pose arrays."""
    urfall_path = Path(urfall_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ensure model is downloaded
    ensure_model(model_path)

    # Support both .mp4 and .avi
    fall_videos = sorted(
        list((urfall_path / "fall").glob("*.mp4")) +
        list((urfall_path / "fall").glob("*.avi"))
    )
    adl_videos = sorted(
        list((urfall_path / "adl").glob("*.mp4")) +
        list((urfall_path / "adl").glob("*.avi"))
    )

    print(f"Found {len(fall_videos)} fall videos, {len(adl_videos)} ADL videos")

    all_metadata = []

    # Process falls
    print("\n--- Processing Fall Videos ---")
    for video_path in tqdm(fall_videos, desc="Falls"):
        try:
            poses, meta = extract_poses_from_video(
                video_path, model_path=model_path
            )
            meta["label"] = "fall"
            meta["label_id"] = 1

            stem = video_path.stem
            np.save(output_path / f"{stem}_poses.npy", poses)
            all_metadata.append(meta)

        except Exception as e:
            print(f"  ERROR processing {video_path.name}: {e}")

    # Process ADLs
    print("\n--- Processing ADL Videos ---")
    for video_path in tqdm(adl_videos, desc="ADL"):
        try:
            poses, meta = extract_poses_from_video(
                video_path, model_path=model_path
            )
            meta["label"] = "adl"
            meta["label_id"] = 0

            stem = video_path.stem
            np.save(output_path / f"{stem}_poses.npy", poses)
            all_metadata.append(meta)

        except Exception as e:
            print(f"  ERROR processing {video_path.name}: {e}")

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Summary
    fall_count = sum(1 for m in all_metadata if m["label"] == "fall")
    adl_count = sum(1 for m in all_metadata if m["label"] == "adl")
    avg_detection = np.mean([m["pose_detection_rate"] for m in all_metadata])

    print(f"\n{'='*60}")
    print(f"Extraction complete:")
    print(f"  Falls processed:  {fall_count}")
    print(f"  ADLs processed:   {adl_count}")
    print(f"  Avg pose detection rate: {avg_detection:.1%}")
    print(f"  Output directory: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract poses from URFall")
    parser.add_argument("--urfall-dir", default="data/urfall")
    parser.add_argument("--output-dir", default="data/poses")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Path to pose_landmarker.task model file")
    args = parser.parse_args()

    process_dataset(args.urfall_dir, args.output_dir, args.model_path)