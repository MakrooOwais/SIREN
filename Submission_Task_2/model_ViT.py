import os
import json
import torch
import torch.nn as nn
from transformers import ViTModel

from PEFT_ViT import ViTFeatureExtractor
from pathlib import Path


class ViTLoRAMultilabel(nn.Module):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        lora_dir: Path | str | None = None,
        idx2lbl_dir: Path | str | None = None,
        threshold: float = 0.5,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        # save dirs
        if lora_dir is not None:
            lora_dir = Path(lora_dir) if isinstance(lora_dir, str) else lora_dir
        if idx2lbl_dir is not None:
            idx2lbl_dir = (
                Path(idx2lbl_dir) if isinstance(idx2lbl_dir, str) else idx2lbl_dir
            )
        else:
            raise AttributeError("Need to have idx2lbl_dir")
        self.lora_dir = lora_dir
        self.idx2lbl_dir = idx2lbl_dir
        self.device_ = device

        num_labels_arts, cifar_classes, self.lbl2idx, self.idx2lbl = [], [], {}, {}
        for cls in os.listdir(self.idx2lbl_dir):
            cls_name = cls[: -len(".json")]
            with open(self.idx2lbl_dir / cls, "r") as f:
                self.idx2lbl[cls_name] = json.load(f)
            self.idx2lbl[cls_name] = {
                int(key): val for key, val in self.idx2lbl[cls_name].items()
            }
            self.lbl2idx[cls_name] = {
                val: key for key, val in self.idx2lbl[cls_name].items()
            }
            num_labels_arts.append(len(self.idx2lbl[cls_name]))
            cifar_classes.append(cls_name)

        # Load base ViT model
        vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        vit.eval()
        for param in vit.parameters():
            param.requires_grad = False
        self.vit = ViTFeatureExtractor(vit)

        # Add custom classification head
        self.classifiers = nn.ModuleDict()
        for num_lables, name in zip(num_labels_arts, cifar_classes):
            self.classifiers[name] = nn.Linear(self.vit.config.hidden_size, num_lables)
        self.sigmoid = nn.Sigmoid()

        # load pretrained weights
        if lora_dir is not None:
            self.vit.load_all_adapters(lora_dir)
            for cls in cifar_classes:
                try:
                    self.classifiers[cls].load_state_dict(
                        torch.load(lora_dir / cls / "classifier.pt", weights_only=False)
                    )
                except Exception as e:
                    print(e)

        # model hyperparameters
        self.threshold = threshold
        self.cifar_classes = cifar_classes

    def forward(self, images, cls):
        with torch.no_grad(), torch.amp.autocast(self.device_):
            outputs = self.vit(images, cls)
            logits = self.sigmoid(
                self.classifiers[cls](outputs.last_hidden_state[:, 0, :])
            )

        preds = (logits > self.threshold).float()
        indices = preds.nonzero().flatten().cpu().tolist()

        return (
            [self.idx2lbl[cls][idx] for idx in indices],
            sorted(
                list(
                    zip(
                        self.idx2lbl[cls].values(),
                        [round(x, 3) for x in logits.flatten().cpu().tolist()],
                    )
                ),
                key=lambda x: x[1],
                reverse=True,
            ),
        )
