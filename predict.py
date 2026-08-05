
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader import build_test_loader
from utils.model import load_DSCrisk
from utils.torchdrug_env import setup_torchdrug_env

setup_torchdrug_env()

# ---------- Internal config ----------
CKPT_PATH = ROOT / "data" / "ckpt.pt"
OUT_DIR = ROOT / "outputs"
DEVICE = "cuda:0"  # fall back to CPU automatically if CUDA is unavailable
CLASS_NAMES = ["Major", "Minor", "Moderate", "Unknown"]


def resolve_device() -> torch.device:
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(DEVICE)
        torch.cuda.set_device(device.index if device.index is not None else 0)
        return device
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    if not CKPT_PATH.is_file():
        raise FileNotFoundError(f"file not found: {CKPT_PATH}")

    device = resolve_device()
    model = load_DSCrisk(CKPT_PATH, device=device)
    loader, df = build_test_loader()

    all_logits = []
    all_labels = []
    for graph_a, graph_b, labels, idx_a, idx_b in tqdm(loader, desc="predict"):
        graph_a = graph_a.to(device)
        graph_b = graph_b.to(device)
        idx_a = idx_a.to(device)
        idx_b = idx_b.to(device)
        logits, _ = model(graph_a, graph_b, idx_a, idx_b)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(dim=-1)

    acc = float((preds == labels).float().mean().item())
    pred_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in preds.tolist()]

    out_df = df[["DDInterID_A", "DDInterID_B"]].copy()
    out_df["pred_class"] = pred_names

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_csv = OUT_DIR / "test_predictions.csv"
    out_df.to_csv(pred_csv, index=False)

    print(f"accuracy: {acc:.6f}")
    print(f"predictions -> {pred_csv}")


if __name__ == "__main__":
    main()
