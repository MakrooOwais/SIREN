import torch

from model import Classifier
from model_gan import GAN
from torchvision import transforms
from PIL import Image

from config import Config


def infer(path):
    conf = Config()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_cls = transforms.Compose([transforms.Resize((224, 224), antialias=True)])
    transform_gan = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    img = Image.open(path).convert("RGB")
    model = Classifier().to(conf.device)
    gan_model = GAN(conf.device)
    model.eval()
    gan_model.eval()
    with torch.no_grad():
        # img = gan_model(transform_gan(img).unsqueeze(0).to(conf.device)).squeeze()
        scores = model(transform_cls(transform_gan(img)).unsqueeze(0).to(conf.device)).squeeze().cpu()
    cls = conf.idx2lbl[scores.argmax().cpu().item()]

    return cls, scores.tolist()


if __name__ == "__main__":
    import os

    path = "/home/owais/The_DumbOne/InterIIT/Final_Inference/perturbed_images_32"
    results = {}
    from tqdm import tqdm
    for img in tqdm(os.listdir(path)):
        results[img] = infer(path + "/" + img)[0]
    import json

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
