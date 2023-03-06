import torch.nn as nn

from models.FgSegNet.model_tools import SegNetDown, SegNetUp, ConvSig, M_FPM, SELayer


class FgSegNet(nn.Module):
    """
    Args:
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    def __init__(self, inp_ch, kernel_size=3):
        super().__init__()
        self.model = nn.Sequential()

        # VGG16
        self.enc1 = SegNetDown(inp_ch, 64, 2, batch_norm=False, kernel_size=kernel_size, maxpool=False, dropout=False)
        self.enc2 = SegNetDown(64, 128, 2, batch_norm=False, kernel_size=kernel_size, maxpool=True, dropout=False)
        self.enc3 = SegNetDown(128, 256, 3, batch_norm=False, kernel_size=kernel_size, maxpool=True, dropout=False)
        self.enc4 = SegNetDown(256, 512, 3, batch_norm=False, kernel_size=kernel_size, maxpool=False, dropout=False)

        # FPM module
        self.fpm = M_FPM(512, 64, kernel_size=kernel_size)

        # Decoder
        self.dec3 = SegNetUp(in_ch=320, res_ch=128, out_ch=64, inst_norm=True, kernel_size=kernel_size)
        self.dec2 = SegNetUp(in_ch=64, res_ch=64, out_ch=64, inst_norm=True, kernel_size=kernel_size)
        self.dec1 = SegNetUp(in_ch=64, res_ch=None, out_ch=64, inst_norm=True, kernel_size=kernel_size)
        self.out = ConvSig(64)

        # Attention
        self.att1 = SELayer(64)
        self.att2 = SELayer(64)

        self.frozenLayers = [self.enc1, self.enc2, self.enc3]
        self.apply(self.weight_init)

    def forward(self, inp):
        """
        """
        # Encoder
        e1 = self.enc1(inp)  # 64, 256, 256
        e2 = self.enc2(e1)  # 128, 128, 128
        e3 = self.enc3(e2)  # 256, 64, 64
        e4 = self.enc4(e3)  # 512, 64, 64

        # FPM
        e5 = self.fpm(e4)  # 320, 64, 64

        # Decoder
        d3 = self.dec3(e5, e2, self.att2, conv1d=True, upSampling=True)  # 64, 128, 128
        d2 = self.dec2(d3, e1, self.att1, conv1d=False, upSampling=True)  # 64, 256, 256
        d1 = self.dec1(d2, conv1d=False, upSampling=False)  # 64, 256, 256

        # Classifier
        cd_out = self.out(d1)
        return cd_out

    def __str__(self):
        return "FgSegNet"



