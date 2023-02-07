import torch
import torch.utils.data
from numpy import sqrt
from torchvision.transforms import transforms

from utils.loss_func import BinaryFocalLoss


def testG(input_G, groundtruth, background, Generator):
    loss_func = torch.nn.BCELoss()
    loss_focal = BinaryFocalLoss()

    Generator.eval()
    Generator.requires_grad_(False)

    torch.cuda.empty_cache()
    conf_map = input_G - background
    predict = Generator(torch.cat([input_G, background], dim=1))
    loss = loss_func(predict, groundtruth)

    return loss.item()

