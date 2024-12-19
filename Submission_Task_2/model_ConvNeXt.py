import timm
import torch
from torch import nn


class ConvNeXt(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_large_384_in22ft1k",
        pretrained_weights_path: str | None = None,
        num_classes: int = 10,
        use_imagenet_weights: bool = False,
        idx2label: dict | None = None,
        device: str | torch.device = "cuda",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.pretrained_weights_path = pretrained_weights_path
        self.num_classes = num_classes
        self.use_imagenet_weights = use_imagenet_weights
        self.idx2label = idx2label
        self.device_ = device

        self.model = timm.create_model(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=self.use_imagenet_weights,
        )

        self.model.load_state_dict(
            torch.load(self.pretrained_weights_path, weights_only=False)
        )

    def forward(self, x):
        with torch.no_grad(), torch.amp.autocast(self.device_):
            pred = nn.functional.softmax(self.model(x), dim=-1)
        if self.idx2label is None:
            return pred, None
        return (
            sorted(
                list(zip(self.idx2label.values(), pred[0].cpu().tolist())),
                key=lambda x: x[1],
                reverse=True,
            ),
            self.idx2label[pred.argmax().item()],
        )
