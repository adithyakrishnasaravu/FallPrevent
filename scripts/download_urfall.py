"""
URFall Dataset Downloader & Organizer
=====================================
Downloads the UR Fall Detection Dataset from:
https://fenix.ur.edu.pl/mkepski/ds/uf.html

Dataset contains:
  - 30 fall sequences (fall-01 to fall-30)
  - 40 ADL sequences (adl-01 to adl-40)
  - RGB data as MP4 videos (cam0 front view)
  - Also available: depth ZIPs, CSV sync/accel data

We download MP4 videos (easiest to process with OpenCV/MediaPipe).
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path
from tqdm import tqdm


# Correct base URL (university domain changed from univ.rzeszow.pl)
BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"

# MP4 video files (cam0 = front-facing camera)
FALL_VIDEOS = [f"fall-{i:02d}-cam0.mp4" for i in range(1, 31)]
ADL_VIDEOS = [f"adl-{i:02d}-cam0.mp4" for i in range(1, 41)]

# Sync/label CSV files
FALL_DATA_CSV = [f"fall-{i:02d}-data.csv" for i in range(1, 31)]
ADL_DATA_CSV = [f"adl-{i:02d}-data.csv" for i in range(1, 41)]

# Accelerometer CSV files
FALL_ACC_CSV = [f"fall-{i:02d}-acc.csv" for i in range(1, 31)]
ADL_ACC_CSV = [f"adl-{i:02d}-acc.csv" for i in range(1, 41)]

# Pre-extracted features (optional but useful for validation)
FEATURE_FILES = ["urfall-cam0-falls.csv", "urfall-cam0-adls.csv"]


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file with progress bar. Skip if exists."""
    if dest.exists() and not force:
        print(f"  {dest.name} — already exists, skipping")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                  desc=dest.name) as t:
            urllib.request.urlretrieve(url, filename=str(dest),
                                       reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_urfall(data_dir: str, force: bool = False,
                     include_csv: bool = False,
                     include_features: bool = False,
                     use_zip_rgb: bool = False):
    """
    Download the URFall dataset.

    By default downloads MP4 videos (compact, easy to process).
    Use --use-zip-rgb to download ZIP archives of PNG frames instead.
    """
    data_path = Path(data_dir)
    fall_dir = data_path / "fall"
    adl_dir = data_path / "adl"

    fall_dir.mkdir(parents=True, exist_ok=True)
    adl_dir.mkdir(parents=True, exist_ok=True)

    all_files = []

    if use_zip_rgb:
        # ZIP archives of PNG image sequences
        for i in range(1, 31):
            fname = f"fall-{i:02d}-cam0-rgb.zip"
            all_files.append((f"{BASE_URL}/{fname}", fall_dir / fname, "fall-rgb"))
        for i in range(1, 41):
            fname = f"adl-{i:02d}-cam0-rgb.zip"
            all_files.append((f"{BASE_URL}/{fname}", adl_dir / fname, "adl-rgb"))
    else:
        # MP4 videos (recommended — smaller, works with OpenCV directly)
        for fname in FALL_VIDEOS:
            all_files.append((f"{BASE_URL}/{fname}", fall_dir / fname, "fall-mp4"))
        for fname in ADL_VIDEOS:
            all_files.append((f"{BASE_URL}/{fname}", adl_dir / fname, "adl-mp4"))

    if include_csv:
        # Sync data CSVs (contain frame labels for falls)
        for fname in FALL_DATA_CSV:
            all_files.append((f"{BASE_URL}/{fname}", fall_dir / fname, "fall-csv"))
        for fname in ADL_DATA_CSV:
            all_files.append((f"{BASE_URL}/{fname}", adl_dir / fname, "adl-csv"))

        # Accelerometer CSVs
        for fname in FALL_ACC_CSV:
            all_files.append((f"{BASE_URL}/{fname}", fall_dir / fname, "fall-acc"))
        for fname in ADL_ACC_CSV:
            all_files.append((f"{BASE_URL}/{fname}", adl_dir / fname, "adl-acc"))

    if include_features:
        for fname in FEATURE_FILES:
            all_files.append((f"{BASE_URL}/{fname}", data_path / fname, "features"))

    print(f"Downloading {len(all_files)} files to {data_path}")
    print(f"Source: {BASE_URL}")
    print("=" * 60)

    success, failed = 0, 0
    for url, dest, category in all_files:
        if download_file(url, dest, force):
            success += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Done: {success} downloaded, {failed} failed")

    # Verify
    verify_dataset(data_path, use_zip_rgb)


def verify_dataset(data_path: Path, use_zip_rgb: bool = False):
    """Check what we have."""
    fall_dir = data_path / "fall"
    adl_dir = data_path / "adl"

    ext = "*.zip" if use_zip_rgb else "*.mp4"
    fall_vids = list(fall_dir.glob(ext))
    adl_vids = list(adl_dir.glob(ext))
    fall_csvs = list(fall_dir.glob("*-data.csv"))
    adl_csvs = list(adl_dir.glob("*-data.csv"))

    print(f"\nDataset verification:")
    print(f"  Fall videos: {len(fall_vids)}/30")
    print(f"  ADL videos:  {len(adl_vids)}/40")
    if fall_csvs or adl_csvs:
        print(f"  Fall CSVs:   {len(fall_csvs)}/30")
        print(f"  ADL CSVs:    {len(adl_csvs)}/40")

    total_expected = 70
    total_got = len(fall_vids) + len(adl_vids)
    if total_got < total_expected:
        print(f"  WARNING: {total_expected - total_got} videos missing. "
              f"Re-run with --force to retry.")
    else:
        print("  All videos present!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URFall dataset")
    parser.add_argument("--data-dir", default="data/urfall",
                        help="Where to save (default: data/urfall)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing files")
    parser.add_argument("--include-csv", action="store_true",
                        help="Also download sync + accelerometer CSVs")
    parser.add_argument("--include-features", action="store_true",
                        help="Also download pre-extracted feature CSVs")
    parser.add_argument("--use-zip-rgb", action="store_true",
                        help="Download ZIP PNG sequences instead of MP4 videos")
    args = parser.parse_args()

    download_urfall(
        args.data_dir,
        force=args.force,
        include_csv=args.include_csv,
        include_features=args.include_features,
        use_zip_rgb=args.use_zip_rgb,
    )