# examples/train_pytorch_example.py

import argparse
from pathlib import Path

import torch
from torch import nn, optim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--input-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def make_model(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def synthetic_batch(batch_size, input_dim, device):
    x = torch.randn(batch_size, input_dim, device=device)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    return x, y


def save_checkpoint(checkpoint_dir: Path, model, optimizer, epoch):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        ckpt_path,
    )
    print(f"[train_pytorch] Saved checkpoint at epoch {epoch} -> {ckpt_path}")


def load_checkpoint(checkpoint_dir: Path, model, optimizer):
    ckpt_path = checkpoint_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        print("[train_pytorch] No checkpoint found, starting from scratch.")
        return 0

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    epoch = state["epoch"]
    print(f"[train_pytorch] Loaded checkpoint from epoch {epoch}")
    return epoch


def main():
    args = parse_args()
    device = torch.device("cpu")  # CPU-only for now
    checkpoint_dir = Path(args.checkpoint_dir)

    model = make_model(args.input_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    if args.resume:
        start_epoch = load_checkpoint(checkpoint_dir, model, optimizer)
    else:
        start_epoch = 0
        print("[train_pytorch] Fresh training run.")

    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        model.train()
        losses = []
        # 10 synthetic batches per epoch
        for _ in range(10):
            x, y = synthetic_batch(args.batch_size, args.input_dim, device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"[train_pytorch] Epoch {epoch + 1}/{total_epochs} - loss={avg_loss:.4f}")
        save_checkpoint(checkpoint_dir, model, optimizer, epoch + 1)

    print("[train_pytorch] Training completed successfully.")


if __name__ == "__main__":
    main()
