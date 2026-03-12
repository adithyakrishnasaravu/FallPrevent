"""
Dataset Preparation
===================
Converts raw pose sequences into sliding-window samples for LSTM training.
Handles augmentation: mirroring, speed variation, noise injection, keypoint dropout.

Output: train/val/test splits as .npz files with sequences and labels.
"""

import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml


# MediaPipe left/right landmark pairs for mirroring
MIRROR_PAIRS = [
    (1, 4), (2, 5), (3, 6),          # eyes
    (7, 8),                            # ears
    (9, 10),                           # mouth
    (11, 12), (13, 14), (15, 16),     # shoulders, elbows, wrists
    (17, 18), (19, 20), (21, 22),     # hands
    (23, 24), (25, 26), (27, 28),     # hips, knees, ankles
    (29, 30), (31, 32),               # feet
]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_sliding_windows(poses: np.ndarray, seq_len: int,
                            stride: int) -> np.ndarray:
    """
    Create sliding window sequences from a pose array.

    Args:
        poses: (num_frames, 33, 4)
        seq_len: window size (e.g., 30 frames = 1 second)
        stride: step between windows

    Returns:
        windows: (num_windows, seq_len, 132)  — flattened keypoints
    """
    num_frames = poses.shape[0]
    if num_frames < seq_len:
        # Pad short sequences
        pad = np.zeros((seq_len - num_frames, *poses.shape[1:]),
                       dtype=poses.dtype)
        poses = np.concatenate([poses, pad], axis=0)
        num_frames = seq_len

    # Flatten keypoints: (frames, 33, 4) -> (frames, 132)
    flat = poses.reshape(num_frames, -1)

    windows = []
    for start in range(0, num_frames - seq_len + 1, stride):
        windows.append(flat[start:start + seq_len])

    return np.array(windows, dtype=np.float32)


def augment_mirror(poses: np.ndarray) -> np.ndarray:
    """Mirror pose by swapping left/right landmarks and flipping x."""
    mirrored = poses.copy()
    for left, right in MIRROR_PAIRS:
        mirrored[:, left], mirrored[:, right] = (
            poses[:, right].copy(), poses[:, left].copy()
        )
    # Flip x coordinate (column 0)
    mirrored[:, :, 0] = 1.0 - mirrored[:, :, 0]
    return mirrored


def augment_speed(poses: np.ndarray, factor: float) -> np.ndarray:
    """Resample poses to simulate speed change."""
    num_frames = poses.shape[0]
    new_len = int(num_frames / factor)
    if new_len < 2:
        return poses

    indices = np.linspace(0, num_frames - 1, new_len).astype(int)
    return poses[indices]


def augment_noise(poses: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to keypoint coordinates (not visibility)."""
    noisy = poses.copy()
    noise = np.random.randn(*poses.shape).astype(np.float32) * std
    noise[:, :, 3] = 0  # Don't add noise to visibility
    noisy += noise
    return noisy


def augment_dropout(poses: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
    """Randomly zero out entire keypoints to simulate occlusion."""
    dropped = poses.copy()
    mask = np.random.random((poses.shape[0], poses.shape[1])) < drop_prob
    dropped[mask] = 0
    return dropped


def prepare_dataset(poses_dir: str, output_dir: str,
                     config_path: str = "configs/config.yaml"):
    """
    Full dataset preparation pipeline:
    1. Load all pose files
    2. Apply augmentations
    3. Create sliding windows
    4. Split into train/val/test
    5. Save as .npz
    """
    cfg = load_config(config_path)
    lstm_cfg = cfg["lstm"]
    aug_cfg = cfg["augmentation"]

    poses_path = Path(poses_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(poses_path / "metadata.json") as f:
        metadata = json.load(f)

    all_windows = []
    all_labels = []

    print("Preparing dataset...")
    for meta in tqdm(metadata):
        video_name = Path(meta["video"]).stem
        pose_file = poses_path / f"{video_name}_poses.npy"

        if not pose_file.exists():
            print(f"  Skipping {video_name} — pose file not found")
            continue

        poses = np.load(pose_file)  # (frames, 33, 4)
        label = meta["label_id"]    # 1=fall, 0=adl

        # Collect all augmented versions of this sequence
        variants = [poses]

        if aug_cfg.get("mirror", True):
            variants.append(augment_mirror(poses))

        for factor in aug_cfg.get("speed_factors", [1.0]):
            if factor != 1.0:
                variants.append(augment_speed(poses, factor))
                if aug_cfg.get("mirror", True):
                    variants.append(augment_speed(
                        augment_mirror(poses), factor
                    ))

        # For each variant, optionally add noise/dropout, then window
        for variant in variants:
            # Original
            windows = create_sliding_windows(
                variant,
                lstm_cfg["sequence_length"],
                lstm_cfg["stride"]
            )
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), label))

            # Noisy version
            if aug_cfg.get("noise_std", 0) > 0:
                noisy = augment_noise(variant, aug_cfg["noise_std"])
                windows = create_sliding_windows(
                    noisy,
                    lstm_cfg["sequence_length"],
                    lstm_cfg["stride"]
                )
                all_windows.append(windows)
                all_labels.append(np.full(len(windows), label))

            # Dropout version
            if aug_cfg.get("random_drop_prob", 0) > 0:
                dropped = augment_dropout(variant, aug_cfg["random_drop_prob"])
                windows = create_sliding_windows(
                    dropped,
                    lstm_cfg["sequence_length"],
                    lstm_cfg["stride"]
                )
                all_windows.append(windows)
                all_labels.append(np.full(len(windows), label))

    # Concatenate
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"\nTotal samples: {len(X)}")
    print(f"  Falls: {(y == 1).sum()}")
    print(f"  ADL:   {(y == 0).sum()}")
    print(f"  Shape: {X.shape}")

    # Stratified split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} (falls: {(y_train==1).sum()})")
    print(f"  Val:   {len(X_val)} (falls: {(y_val==1).sum()})")
    print(f"  Test:  {len(X_test)} (falls: {(y_test==1).sum()})")

    # Compute normalization stats from training set
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1.0  # avoid division by zero

    # Save
    np.savez_compressed(
        output_path / "train.npz", X=X_train, y=y_train
    )
    np.savez_compressed(
        output_path / "val.npz", X=X_val, y=y_val
    )
    np.savez_compressed(
        output_path / "test.npz", X=X_test, y=y_test
    )
    np.savez(
        output_path / "norm_stats.npz", mean=mean, std=std
    )

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LSTM dataset")
    parser.add_argument("--poses-dir", default="data/poses")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    prepare_dataset(args.poses_dir, args.output_dir, args.config)
