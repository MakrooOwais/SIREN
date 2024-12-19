from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    cifar_lbl2idx: dict[str, int] = field(
        default_factory=lambda: {
            "dog": 0,
            "ship": 1,
            "truck": 2,
            "frog": 3,
            "bird": 4,
            "car": 5,
            "horse": 6,
            "airplane": 7,
            "cat": 8,
            "deer": 9,
        }
    )
    cifar_idx2lbl: dict[int, str] = field(
        default_factory=lambda: {
            0: "dog",
            1: "ship",
            2: "truck",
            3: "frog",
            4: "bird",
            5: "car",
            6: "horse",
            7: "airplane",
            8: "cat",
            9: "deer",
        }
    )
    cifar_classes: list[str] = field(
        default_factory=lambda: [
            "dog",
            "ship",
            "truck",
            "frog",
            "bird",
            "car",
            "horse",
            "airplane",
            "cat",
            "deer",
        ]
    )

    # CLIP Parameters
    clip_desc_json_dir: Path = Path(os.getcwd()) / "Descriptors"
    clip_desc_embd_dir: Path = Path(os.getcwd()) / "Descriptors_Embeddings"
    clip_model_name: str = "ViT-B-32"
    clip_pretrain_name: str = "laion2b_s34b_b79k"
    clip_n: int = 5

    # ViT Parameters
    vit_model_name: str = "google/vit-base-patch16-224-in21k"
    vit_lora_dir: Path = None
    vit_lora_dir: Path = Path(os.getcwd()) / "saved_weights/lora_weights"
    vit_idx2lbl_dir: Path = Path(os.getcwd()) / "idx2label"
    vit_threshold: float = 0.5

    # ConvNeXt Parameters
    convnext_model_name: str = "convnext_large.fb_in22k_ft_in1k_384"
    convnext_weight_path: Path = Path(os.getcwd()) / "saved_weights/convnext.pt"
    convnext_num_classes: int = 10
    convnext_use_imagenet_weights: bool = False

    # Segmentator Parameters
    seg_model_name: str = "CIDAS/clipseg-rd64-refined"
    seg_n_masks: int = 3
