import torch

from torch import nn
from torchvision.models import efficientnet_v2_s
from torchmetrics import Accuracy

from config import Config


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conf = Config()

        self.accuracy = Accuracy("binary")
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.backbone = efficientnet_v2_s()

        n_input = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(
            in_features=n_input, out_features=2, bias=True
        )

        self.load_state_dict(torch.load(self.conf.cls_weight_path, weights_only=True), strict=False)

    def forward(self, img):
        img = self.backbone(img)

        return nn.functional.softmax(img, -1)
