import torch
import json
import os

from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image
from torch import nn
from torchvision import transforms


from model_ConvNeXt import ConvNeXt
from model_ViT import ViTLoRAMultilabel
from model_CLIP import CLIP
from model_CLIPSeg import CLIPSeg

from config import Config


class SIRENa(nn.Module):
    def __init__(self, device: torch.device | str = "cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conf = Config()
        self.device_ = device

        self.convnext_model = ConvNeXt(
            self.conf.convnext_model_name,
            self.conf.convnext_weight_path,
            self.conf.convnext_num_classes,
            self.conf.convnext_use_imagenet_weights,
            self.conf.cifar_idx2lbl,
            self.device_,
        ).to(self.device_)

        self.vit_model = ViTLoRAMultilabel(
            self.conf.vit_model_name,
            self.conf.vit_lora_dir,
            self.conf.vit_idx2lbl_dir,
            self.conf.vit_threshold,
            self.device_,
        ).to(self.device_)

        self.clip_model = CLIP(
            self.conf.clip_model_name,
            self.conf.clip_pretrain_name,
            self.conf.clip_desc_embd_dir,
            self.conf.clip_desc_json_dir,
            self.conf.clip_n,
            self.device_,
        ).to(self.device_)

        self.seg_model = CLIPSeg(self.conf.seg_model_name, self.device_)

        self.convnext_model.eval()
        self.vit_model.eval()
        self.clip_model.eval()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (32, 32),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def open_img(self, path):
        return Image.open(path)

    def trans_img(self, img):
        return self.transform(img).unsqueeze(0).to(self.device_)

    def forward(self, img_path, save_dir="results"):
        o_img = self.open_img(img_path)
        t_img = self.trans_img(o_img)

        try:
            cifar_scores, cls = self.convnext_model(t_img)
            arts, art_logits = self.vit_model(t_img, cls)
            results_descs, desc_logits = self.clip_model(t_img, cls, arts)

            arts_mask = sorted(art_logits, key=lambda x: x[1], reverse=True)
            arts_reduced = [x for x in arts_mask if x[1] > self.conf.vit_threshold]
            desc_mask = {
                x: ", ".join([i.lower() for i, _ in desc_logits[x][: self.conf.clip_n]])
                for x, _ in arts_reduced
            }

            masks = self.seg_model(
                o_img,
                list(desc_mask.values())[: self.conf.clip_n],
            )
        except:
            print(cls, arts)

        results = {
            "CIFAR_Scores": cifar_scores,
            "Artifact_Logits": art_logits,
            "Desc_Logits": desc_logits,
            "Results_Descs": results_descs,
        }

        save_dir = Path(save_dir) if not isinstance(save_dir, Path) else save_dir
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir / "results.json", "w") as f:
            json.dump(results, f, indent=4)
        for idx, mask in enumerate(masks[:self.conf.seg_n_masks]):
            plt.imshow(mask)
            plt.imsave(save_dir / (str(idx) + ".png"), mask)

        return desc_mask

if __name__ == "__main__":
    import json
    import os
    path = "/home/owais/The_DumbOne/InterIIT/Final_Inference/perturbed_images_32"
    with open("/home/owais/The_DumbOne/InterIIT/Submission/Submission_Task_1/results.json", "r") as f:
        cls_res = json.load(f)
    model = SIRENa()
    results = []
    from tqdm import tqdm
    for img in tqdm(cls_res):
        if cls_res[img] == "fake":
            format_res = {"index": int(img[:-len(".png")])}
            new_dict  = {}
            res = model(path + "/" + img)
            for art in res:
                new_dict[art.capitalize()] = res[art]
            format_res["explanation"] = new_dict
            results.append(format_res)
    with open("results.json", "w") as f:
        json.dump(results, f, indent = 4)