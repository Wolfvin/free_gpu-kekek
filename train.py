"""
Sample training script for free-gpu-trainer.
Replace this with your actual training code.

This script demonstrates checkpoint save/resume functionality
that works with the auto-rotation system.
"""

import argparse
import json
import os
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with checkpoint support")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save/load checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    return parser.parse_args()


def save_checkpoint(checkpoint_dir: str, epoch: int, loss: float, best_loss: float,
                    model_state: dict = None, extra: dict = None):
    """Save a training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "timestamp": time.time(),
    }
    if extra:
        state.update(extra)
    if model_state:
        state["model_state"] = model_state

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.json")
    with open(ckpt_path, "w") as f:
        json.dump(state, f, indent=2)

    # Save latest pointer
    latest_path = os.path.join(checkpoint_dir, "latest.json")
    with open(latest_path, "w") as f:
        json.dump({"latest_checkpoint": ckpt_path, **state}, f, indent=2)

    # Update training state for free-gpu-trainer rotation
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    with open(state_path, "w") as f:
        json.dump({
            "total_epochs": 100,  # your total epochs
            "completed_epochs": epoch,
            "current_loss": loss,
            "best_loss": best_loss,
            "last_checkpoint": time.time(),
        }, f, indent=2)

    print(f"[Checkpoint] Saved epoch {epoch}, loss={loss:.4f}")


def load_checkpoint(checkpoint_dir: str) -> dict:
    """Load the latest checkpoint."""
    latest_path = os.path.join(checkpoint_dir, "latest.json")
    if not os.path.exists(latest_path):
        return {}
    with open(latest_path) as f:
        return json.load(f)


def train():
    args = parse_args()

    print("=" * 60)
    print("Free GPU Trainer — Sample Training Script")
    print("=" * 60)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Resume from: {args.resume or 'scratch'}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA: {torch.version.cuda}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            print("WARNING: No GPU detected, running on CPU")
    except ImportError:
        print("WARNING: PyTorch not installed")

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float("inf")
    resume_dir = args.resume or args.checkpoint_dir

    ckpt = load_checkpoint(resume_dir)
    if ckpt:
        start_epoch = ckpt.get("completed_epochs", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resuming from epoch {start_epoch}, best_loss={best_loss:.4f}")

    print()
    print("Training started...")
    print("-" * 40)

    # ── Replace this with your actual training loop ──
    for epoch in range(start_epoch, args.epochs):
        # Simulated training step
        loss = 1.0 / (epoch + 1) + (0.01 * (epoch % 5))  # fake loss curve
        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | Loss: {loss:.4f} | Best: {best_loss:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                args.checkpoint_dir,
                epoch=epoch,
                loss=loss,
                best_loss=best_loss,
            )

        # Simulate training time
        time.sleep(0.1)

    print("-" * 40)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    train()
