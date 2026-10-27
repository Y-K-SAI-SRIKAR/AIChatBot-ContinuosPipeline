"""
IntentDataset: PyTorch Dataset wrapping processed train/val/test splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class IntentDataset(Dataset):
    """
    PyTorch Dataset for intent classification examples.

    Each item from the JSON split is a dict with keys:
        text, intent, label_id, split
    """

    def __init__(self, examples: List[Dict]) -> None:
        self.examples = examples

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: Path) -> "IntentDataset":
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        if not isinstance(examples, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(examples)}")
        return cls(examples)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

    # ------------------------------------------------------------------
    @property
    def texts(self) -> List[str]:
        return [e["text"] for e in self.examples]

    @property
    def labels(self) -> List[int]:
        return [e["label_id"] for e in self.examples]

