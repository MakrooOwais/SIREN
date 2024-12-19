import json
import open_clip
import os
import torch

from pathlib import Path
from torch import nn

from config import Config


class CLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained_name: str = "laion2b_s34b_b79k",
        desc_emb_path: str | Path | None = None,
        desc_json_path: str | Path | None = None,
        n: int = 3,
        device: str | torch.device = "cuda",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Save the model and weights parameters
        self.model_name = model_name
        self.pretrained_name = pretrained_name
        self.device_ = device

        # Setup clip model
        self.model, _, self.img_preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained_name
        )
        self.model.eval()
        self.txt_tokenizer = open_clip.get_tokenizer(self.model_name)

        # Get the configs
        self.config = Config()

        # Setup paths
        if desc_emb_path is not None:
            if isinstance(desc_emb_path, str):
                desc_emb_path = Path(desc_emb_path)
        if desc_json_path is not None:
            if isinstance(desc_json_path, str):
                desc_json_path = Path(desc_json_path)

        self.n = n
        self.desc_emd_path = desc_emb_path
        self.desc_json_path = desc_json_path
        self.json_files = {}

    def get_desc_embeddings(self, cls: str | int, art: str):
        if isinstance(cls, int):
            cls = self.config.idx2lbl[cls]

        if cls not in self.json_files:
            with open(self.desc_json_path / (cls + ".json"), "r") as f:
                self.json_files[cls] = json.load(f)

        if not os.path.exists(self.desc_emd_path / cls):
            os.makedirs(self.desc_emd_path / cls)
        if not os.path.exists(self.desc_emd_path / cls / (art + ".pth")):
            txt = self.json_files[cls][art]["descriptors"]
            txt = self.txt_tokenizer(txt).to(self.device_)
            with torch.no_grad(), torch.amp.autocast(self.device_):
                embd = self.model.encode_text(txt)
                embd /= embd.norm(dim=-1, keepdim=True)

            torch.save(embd, self.desc_emd_path / cls / (art + ".pth"))
        return (
            torch.load(self.desc_emd_path / cls / (art + ".pth"), weights_only=True).to(
                self.device_
            ),
            self.json_files[cls][art]["descriptors"],
        )

    def forward(self, x: torch.Tensor, cls: str | int, arts: list[str]):
        if isinstance(cls, int):
            cls = self.config.cifar_idx2lbl[cls]

        results = {}
        logits = {}

        with torch.no_grad(), torch.amp.autocast("cuda"):
            embd = self.model.encode_image(x)
            embd /= embd.norm(dim=-1, keepdim=True)

        for art in arts:
            art_embd, descs = self.get_desc_embeddings(cls, art)
            with torch.amp.autocast("cuda"):
                text_probs = torch.softmax(embd @ art_embd.T * 100.00, dim=-1)
            results[art] = [
                descs[i.item()]
                for i in torch.topk(text_probs, min(self.n, len(text_probs))).indices[0]
            ]
            logits[art] = sorted(
                list(
                    zip(descs, [round(x.cpu().item(), 3) for x in text_probs.flatten()])
                ),
                key=lambda x: x[1],
                reverse=True,
            )

        return results, logits
