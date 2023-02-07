import torch
import torchvision.transforms
from torch import nn
from torchvision.transforms import transforms

from models.Unet.unet_tools import UNetDown, UNetUp, ConvSig


class UNet(nn.Module):
    """
        Args:
            inp_ch (int): Number of input channels
            kernel_size (int): Size of the convolutional kernels
            skip (bool, default=True): Use skip connections
    """

    def __init__(self, inp_ch, kernel_size=3, skip=True):
        super(UNet, self).__init__()
        self.skip = skip
        self.enc1 = UNetDown(inp_ch, 32, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
        self.enc2 = UNetDown(32, 64, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc3 = UNetDown(64, 128, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc4 = UNetDown(128, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc5 = UNetDown(256, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.dec4 = UNetUp(256, skip * 256, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(256, skip * 128, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(128, skip * 64, 64, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(64, skip * 32, 32, 2, batch_norm=True, kernel_size=kernel_size)
        self.out = ConvSig(32)

    def forward(self, inp, conf_map=None):
        """
        Args:
            inp (tensor): Tensor of input Minibatch
            conf_map: Tensor of confidence map

        Returns:
            (tensor): Change detection output
        """
        d1 = self.enc1(inp)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        d5 = self.enc5(d4)
        if self.skip:
            if conf_map is None:
                conf_map = torch.ones(size=[inp.shape[0], 1, inp.shape[2], inp.shape[3]], dtype=torch.float)
                conf_map = conf_map.to(inp.device)
            resize = transforms.Resize([d4.shape[2], d4.shape[3]])
            u4 = self.dec4(d5, d4*resize(conf_map))  # 32
            resize = transforms.Resize([d3.shape[2], d3.shape[3]])
            u3 = self.dec3(u4, d3*resize(conf_map))  # 64
            resize = transforms.Resize([d2.shape[2], d2.shape[3]])
            u2 = self.dec2(u3, d2*resize(conf_map))  # 128
            resize = transforms.Resize([d1.shape[2], d1.shape[3]])
            u1 = self.dec1(u2, d1*resize(conf_map))  # 256
        else:
            u4 = self.dec4(d5)
            u3 = self.dec3(u4)
            u2 = self.dec2(u3)
            u1 = self.dec1(u2)

        cd_out = self.out(u1)
        return cd_out

    def __str__(self):
        return 'UNet'


