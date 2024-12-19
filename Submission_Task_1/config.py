import os

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    cls_weight_path: Path = Path(os.getcwd()) / "saved_weights/effnet_b2_s.pth"
    gan_weight_path: Path = Path(os.getcwd()) / "saved_weights/gan.pth"
    idx2lbl: dict = field(default_factory=lambda: {0: "real", 1: "fake"})
    device: str = "cuda"
    
    gan_missing_rate: float = 0.25
    gan_block_n: int = 32
