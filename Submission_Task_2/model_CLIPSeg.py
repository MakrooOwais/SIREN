from transformers import CLIPProcessor, CLIPSegForImageSegmentation
import torch

from torch import nn

from config import Config


class CLIPSeg(nn.Module):
    def __init__(
        self,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: str | torch.device = "cuda",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Save the model and weights parameters
        self.model_name = model_name
        self.device_ = device

        # Setup CLIPSeg model
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self.model_name)
        self.model.eval()

        # Get the configs
        self.config = Config()

    def forward(self, x: torch.Tensor, descs: list[str]):
        x = self.processor(
            text=descs,
            images=[x] * len(descs),
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad(), torch.amp.autocast(self.device_):
            out = self.model(**x)
        preds = out.logits.unsqueeze(1)

        return [torch.sigmoid(preds[i][0].cpu()) for i in range(len(descs))]
