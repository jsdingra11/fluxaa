# examples/train_example.py

import argparse
import json
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs to run.")
    return parser.parse_args()


def load_checkpoint(checkpoint_dir: Path):
    ckpt_path = checkpoint_dir / "checkpoint.json"
    if ckpt_path.exists():
        with ckpt_path.open("r") as f:
            data = json.load(f)
        print(f"[train_example] Loaded checkpoint: {data}")
        return data
    else:
        print("[train_example] No checkpoint found, starting from scratch.")
        return {"epoch": 0}


def save_checkpoint(checkpoint_dir: Path, state: dict):
    ckpt_path = checkpoint_dir / "checkpoint.json"
    with ckpt_path.open("w") as f:
        json.dump(state, f)
    print(f"[train_example] Saved checkpoint: {state}")


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Simulate resuming
    if args.resume:
        state = load_checkpoint(checkpoint_dir)
        start_epoch = state.get("epoch", 0)
    else:
        print("[train_example] Fresh run.")
        start_epoch = 0
        state = {"epoch": 0}

    total_epochs = args.epochs

    # Simulate a crash only on first run
    crash_flag_path = checkpoint_dir / "crashed_once.flag"
    should_crash = not crash_flag_path.exists()

    for epoch in range(start_epoch, total_epochs):
        print(f"[train_example] Epoch {epoch + 1}/{total_epochs} ...")
        # Simulate work
        time.sleep(1.0)

        # Update state + checkpoint every epoch
        state["epoch"] = epoch + 1
        save_checkpoint(checkpoint_dir, state)

        # Simulate a crash around the middle on first run
        if should_crash and epoch == (total_epochs // 2):
            print("[train_example] Simulating crash now!")
            crash_flag_path.write_text("crashed")
            raise RuntimeError("Intentional crash for testing orchestrator.")

    print("[train_example] Training completed successfully.")


if __name__ == "__main__":
    main()
