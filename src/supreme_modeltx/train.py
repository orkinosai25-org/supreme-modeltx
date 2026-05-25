from __future__ import annotations

import argparse
from pathlib import Path


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for training. Install a CUDA build from https://pytorch.org/get-started/locally/."
        ) from exc
    return torch


from .config import TrainingConfig, load_config


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    tokens = sorted(set(text))
    stoi = {ch: idx for idx, ch in enumerate(tokens)}
    itos = {idx: ch for ch, idx in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]):
    return [stoi[ch] for ch in text if ch in stoi]


def get_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_batch(torch, token_ids, cfg: TrainingConfig, device):
    max_start = len(token_ids) - cfg.sequence_length - 1
    starts = torch.randint(max_start, (cfg.batch_size,))
    x = torch.stack(
        [token_ids[start : start + cfg.sequence_length] for start in starts]
    ).to(device)
    y = torch.stack(
        [token_ids[start + 1 : start + cfg.sequence_length + 1] for start in starts]
    ).to(device)
    return x, y


def run_training(cfg: TrainingConfig, text: str, requested_device: str = "auto"):
    torch = _require_torch()
    from torch import optim

    from .model import T100LM

    if len(text) < cfg.sequence_length + 2:
        raise ValueError("Training text is too short for configured sequence length.")

    stoi, _ = build_vocab(text)
    if len(stoi) > cfg.vocab_size:
        raise ValueError(
            f"Text has {len(stoi)} unique tokens, above vocab_size={cfg.vocab_size}. Increase vocab_size."
        )

    token_ids = torch.tensor(encode(text, stoi), dtype=torch.long)
    device = torch.device(get_device(torch, requested_device))

    model = T100LM(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    model.train()
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sumotx] training on device={device} cuda_available={torch.cuda.is_available()}")
    print(f"[sumotx] architecture=T100 series, layers={cfg.num_layers}, heads={cfg.num_heads}")

    for step in range(1, cfg.max_steps + 1):
        x, y = sample_batch(torch, token_ids, cfg, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg.use_amp and device.type == "cuda",
        ):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if step % cfg.eval_interval == 0 or step == 1 or step == cfg.max_steps:
            print(f"step={step} loss={loss.item():.4f}")

    ckpt_path = checkpoint_dir / "t100_6layer_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "vocab": stoi,
        },
        ckpt_path,
    )
    print(f"[sumotx] checkpoint saved: {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Supreme Model TX (T100 6-layer baseline)")
    parser.add_argument("--config", type=str, default="configs/t100_6layer.json")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.data:
        text = Path(args.data).read_text(encoding="utf-8")
    else:
        text = (
            "Supreme Model TX is trained for sovereign model development. "
            "This default corpus is only for smoke training and GPU pipeline validation. "
        ) * 80

    run_training(cfg, text=text, requested_device=args.device)


if __name__ == "__main__":
    main()
