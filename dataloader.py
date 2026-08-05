
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.torchdrug_env import setup_torchdrug_env

setup_torchdrug_env()

from torchdrug import data  # noqa: E402

# ---------- Internal config ----------
DATASET_DIR = ROOT / "data" / "dataset"
TEST_CSV = DATASET_DIR / "test.csv"
BATCH_SIZE = 64
NUM_WORKERS = 0
SMILE_A_COL = "DDInterID_A_Smile"
SMILE_B_COL = "DDInterID_B_Smile"


def _parse_molecule(smiles: str):
    try:
        mol = data.Molecule.from_smiles(smiles)
    except Exception:
        return None
    if mol is None or int(mol.num_node) <= 0:
        return None
    return mol


class DDIPairDataset(Dataset):
    def __init__(self, pair_csv: Path):
        self.df = pd.read_csv(pair_csv)
        required = [
            "label",
            "drug_idx_a",
            "drug_idx_b",
            SMILE_A_COL,
            SMILE_B_COL,
        ]
        for col in required:
            if col not in self.df.columns:
                raise KeyError(f"pair csv is missing column: {col}")

        # Cache unique SMILES -> Molecule to avoid repeated parsing
        unique_smiles = sorted(
            set(self.df[SMILE_A_COL].astype(str).str.strip())
            | set(self.df[SMILE_B_COL].astype(str).str.strip())
        )
        self.smile_to_mol = {}
        failed = []
        for smi in unique_smiles:
            mol = _parse_molecule(smi)
            if mol is None:
                failed.append(smi)
            else:
                self.smile_to_mol[smi] = mol
        if failed:
            raise ValueError(
                f"failed to parse {len(failed)} SMILES; first={failed[0]!r}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        smi_a = str(row[SMILE_A_COL]).strip()
        smi_b = str(row[SMILE_B_COL]).strip()
        idx_a = int(row["drug_idx_a"])
        idx_b = int(row["drug_idx_b"])
        label = int(row["label"])
        return self.smile_to_mol[smi_a], self.smile_to_mol[smi_b], label, idx_a, idx_b


def collate_ddi_pair(batch):
    mols_a, mols_b, labels, idx_a, idx_b = zip(*batch)
    graph_a = data.Molecule.pack(list(mols_a))
    graph_b = data.Molecule.pack(list(mols_b))
    labels = torch.tensor(labels, dtype=torch.long)
    idx_a = torch.tensor(idx_a, dtype=torch.long)
    idx_b = torch.tensor(idx_b, dtype=torch.long)
    return graph_a, graph_b, labels, idx_a, idx_b


def build_test_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """Build the test DataLoader from test.csv (SMILES + drug indices)."""
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"file not found: {TEST_CSV}")

    dataset = DDIPairDataset(TEST_CSV)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_ddi_pair,
    )
    return loader, dataset.df


def main() -> None:
    loader, df = build_test_loader()
    print(f"test pairs: {len(df)}")
    print(f"batches: {len(loader)} (batch_size={BATCH_SIZE})")
    graph_a, graph_b, labels, idx_a, idx_b = next(iter(loader))
    print(
        f"first batch: graph_a.num_node={int(graph_a.num_node)}, "
        f"labels={tuple(labels.tolist()[:8])}..., "
        f"idx_a={tuple(idx_a.tolist()[:4])}..."
    )


if __name__ == "__main__":
    main()
