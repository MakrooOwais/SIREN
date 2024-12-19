import numpy as np
import torch

from torch import nn

from atop.model.networks import Generator
from config import Config


class GAN(nn.Module):
    def __init__(self, device):
        super(GAN, self).__init__()

        self.conf = Config()
        sd_path = None
        self.generator = Generator(
            cnum_in=5, cnum=48, return_flow=False, checkpoint=sd_path
        ).to(device)
        self.generator.load_state_dict(torch.load(self.conf.gan_weight_path, weights_only=True)["G"], strict=False)
        self.generator.train()

    def forward(self, x):
        inputs_gau = x
        nums = np.zeros(int((256**2) / (self.conf.gan_block_n**2)))
        nums[: int(self.conf.gan_missing_rate * (len(nums)))] = 1
        np.random.shuffle(nums)
        mask = nums.reshape(
            (1, 1, int(256 / self.conf.gan_block_n), int(256 / self.conf.gan_block_n))
        )
        mask = np.repeat(mask, self.conf.gan_block_n, axis=2)
        mask = np.repeat(mask, self.conf.gan_block_n, axis=3)

        mask = torch.from_numpy(mask).type(torch.float).to(self.conf.device)

        batch_incomplete = inputs_gau * (1.0 - mask)
        ones_x = torch.ones_like(batch_incomplete)[:, 0:1, :, :].to(self.conf.device)
        x = torch.cat([batch_incomplete, ones_x, ones_x * mask], axis=1)
        _, x2 = self.generator(x, mask, inputs_gau)

        return x2
