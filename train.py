"""Example training script for FamilyGPU Orchestrator.

This script demonstrates the checkpoint contract that training
scripts must follow to work with the orchestrator:

  1. save_checkpoint(path) — Save model and optimizer state
  2. load_checkpoint(path) — Resume from a checkpoint
  3. Accept --checkpoint-dir and --resume-from arguments

The orchestrator will:
  - Start this script on the selected provider
  - Monitor the lease and heartbeat
  - Trigger checkpoint before lease expiry
  - Resume from checkpoint on failover
"""

import os
import argparse
import json
import time
from pathlib import Path


def save_checkpoint(path: str, epoch: int, model_state: dict, optimizer_state: dict):
    """Save training checkpoint.

    Args:
        path: Directory to save checkpoint files
        epoch: Current epoch number
        model_state: Model state dict (simulated)
        optimizer_state: Optimizer state dict (simulated)
    """
    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "timestamp": time.time(),
    }

    ckpt_path = ckpt_dir / "checkpoint.json"
    ckpt_path.write_text(json.dumps(checkpoint, indent=2))
    print(f"[Checkpoint] Saved epoch {epoch} to {ckpt_path}")


def load_checkpoint(path: str) -> dict:
    """Load training checkpoint.

    Args:
        path: Directory containing checkpoint files

    Returns:
        Checkpoint dict with epoch, model_state, optimizer_state
    """
    ckpt_path = Path(path) / "checkpoint.json"
    if not ckpt_path.exists():
        return {}

    checkpoint = json.loads(ckpt_path.read_text())
    print(f"[Checkpoint] Loaded epoch {checkpoint.get('epoch', '?')} from {ckpt_path}")
    return checkpoint


def train(resume_from: str = None, checkpoint_dir: str = "./checkpoints",
          epochs: int = 10, learning_rate: float = 1e-4):
    """Main training loop with checkpoint support.

    Args:
        resume_from: Path to checkpoint directory to resume from
        checkpoint_dir: Directory to save checkpoints
        epochs: Number of training epochs
        learning_rate: Learning rate
    """
    start_epoch = 0
    model_state = {"weights": "random_init", "lr": learning_rate}
    optimizer_state = {"step": 0}

    # Resume from checkpoint if specified
    if resume_from:
        ckpt = load_checkpoint(resume_from)
        if ckpt:
            start_epoch = ckpt.get("epoch", 0) + 1
            model_state = ckpt.get("model_state", model_state)
            optimizer_state = ckpt.get("optimizer_state", optimizer_state)
            print(f"[Train] Resuming from epoch {start_epoch}")
        else:
            print(f"[Train] No checkpoint found at {resume_from}, starting fresh")

    print(f"[Train] Starting training: epochs={epochs}, lr={learning_rate}")
    print(f"[Train] Checkpoint dir: {checkpoint_dir}")
    print(f"[Train] Resume from: {resume_from or 'none'}")

    for epoch in range(start_epoch, epochs):
        # Simulate training step
        loss = 1.0 / (epoch + 1)  # Decreasing loss
        model_state["weights"] = f"epoch_{epoch}_weights"
        optimizer_state["step"] = epoch + 1

        print(f"[Epoch {epoch + 1}/{epochs}] loss={loss:.4f}")

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            save_checkpoint(checkpoint_dir, epoch, model_state, optimizer_state)

        # Simulate training time
        time.sleep(1)

    # Final checkpoint
    save_checkpoint(checkpoint_dir, epochs - 1, model_state, optimizer_state)
    print(f"[Train] Training complete! Final checkpoint saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example training script for FamilyGPU Orchestrator")
    parser.add_argument("--checkpoint-dir", default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume-from", default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")

    args = parser.parse_args()
    train(
        resume_from=args.resume_from,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
