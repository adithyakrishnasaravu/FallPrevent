"""
LSTM Fall Detection Model
=========================
Bidirectional LSTM classifier that takes a window of 30 pose frames
and outputs a fall probability.

Architecture:
  Input: (batch, 30, 132)  — 30 frames × 33 keypoints × 4 dims
  → 2-layer BiLSTM (hidden=128)
  → Attention pooling over timesteps
  → FC → Dropout → FC → Sigmoid
  Output: (batch, 1)  — fall probability
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
import matplotlib.pyplot as plt
import yaml
import argparse
import json
from tqdm import tqdm


# ─── Dataset ────────────────────────────────────────────────────────

class FallDataset(Dataset):
    """Loads pre-windowed pose sequences."""

    def __init__(self, npz_path: str, norm_stats_path: str = None):
        data = np.load(npz_path)
        self.X = torch.FloatTensor(data["X"])
        self.y = torch.FloatTensor(data["y"])

        # Normalize if stats provided
        if norm_stats_path:
            stats = np.load(norm_stats_path)
            mean = torch.FloatTensor(stats["mean"])
            std = torch.FloatTensor(stats["std"])
            self.X = (self.X - mean) / std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Model ──────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Attention over LSTM timesteps to focus on the fall moment."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output: (batch, seq_len, hidden)
        weights = self.attention(lstm_output)           # (batch, seq_len, 1)
        weights = torch.softmax(weights, dim=1)
        weighted = (lstm_output * weights).sum(dim=1)   # (batch, hidden)
        return weighted


class FallDetectorLSTM(nn.Module):
    """
    Bidirectional LSTM with temporal attention for fall detection.
    """

    def __init__(self, input_size: int = 132, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = TemporalAttention(hidden_size * 2)  # *2 for BiLSTM

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            prob: (batch,)  — fall probability
        """
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden*2)
        pooled = self.attention(lstm_out)     # (batch, hidden*2)
        logit = self.classifier(pooled)      # (batch, 1)
        return logit.squeeze(-1)


# ─── Training ───────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(float)

    acc = (preds == all_labels).mean()
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

    return total_loss / len(all_labels), acc, auc, all_probs, all_labels


def train(config_path: str = "configs/config.yaml"):
    """Full training pipeline."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    lstm_cfg = cfg["lstm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    data_dir = Path(cfg["data"]["processed_dir"])
    norm_path = str(data_dir / "norm_stats.npz")

    train_ds = FallDataset(str(data_dir / "train.npz"), norm_path)
    val_ds = FallDataset(str(data_dir / "val.npz"), norm_path)
    test_ds = FallDataset(str(data_dir / "test.npz"), norm_path)

    # Handle class imbalance with weighted sampling
    labels = train_ds.y.numpy()
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels.astype(int)]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights)
    )

    train_loader = DataLoader(
        train_ds, batch_size=lstm_cfg["batch_size"],
        sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=lstm_cfg["batch_size"],
        shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=lstm_cfg["batch_size"],
        shuffle=False, num_workers=2
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Model
    model = FallDetectorLSTM(
        input_size=lstm_cfg["input_size"],
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Use pos_weight for class imbalance in loss
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lstm_cfg["learning_rate"],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    # Training loop
    best_val_auc = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}
    model_path = Path(cfg["model_paths"]["lstm"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {lstm_cfg['epochs']} epochs...")
    print("=" * 70)

    for epoch in range(1, lstm_cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.3f} | "
              f"LR: {lr:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": lstm_cfg,
                "epoch": epoch,
                "val_auc": val_auc,
            }, model_path)
            print(f"  → Saved best model (AUC: {val_auc:.4f})")

    # Test evaluation
    print("\n" + "=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_auc, test_probs, test_labels = eval_epoch(
        model, test_loader, criterion, device
    )

    test_preds = (test_probs > lstm_cfg["fall_threshold"]).astype(int)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test AUC: {test_auc:.3f}")
    print(f"\nClassification Report (threshold={lstm_cfg['fall_threshold']}):")
    print(classification_report(
        test_labels, test_preds,
        target_names=["ADL (no fall)", "Fall"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # Save training history
    with open(model_path.parent / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    plot_training(history, model_path.parent / "training_curves.png")

    return model, history


def plot_training(history: dict, save_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_acc"])
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(history["val_auc"])
    axes[2].set_title("Validation AUC")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM fall detector")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    train(args.config)
