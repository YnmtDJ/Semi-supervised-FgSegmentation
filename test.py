import os
import sys

import torch
import torch.utils.data
from numpy import sqrt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.modelUtils import initNetWork
from options.BaseOptions import BaseOptions
from utils.dataset import GeneratorDataset
from utils.loss_func import BinaryFocalLoss


def testG(input_G, groundtruth, background, Generator):
    loss_func = torch.nn.BCELoss()
    loss_focal = BinaryFocalLoss()

    Generator.eval()
    Generator.requires_grad_(False)

    torch.cuda.empty_cache()
    predict = Generator(torch.cat([input_G, background], dim=1))
    # predict = Generator(input_G)
    loss = loss_func(predict, groundtruth)
    predict = torch.round(predict)
    error = (predict - groundtruth).abs().sum()/(input_G.shape[0]*input_G.shape[2]*input_G.shape[3])

    return loss.item(), error.item()


if __name__ == '__main__':
    opt = BaseOptions()
    if not opt.initialized:
        opt.initialize()
    opt = opt.parser.parse_args()
    # 加载网络
    generator, discriminator = initNetWork(opt)
    # 数据集
    dataset = GeneratorDataset(os.path.join(opt.dataRoot, "unlabel"), opt.inputSize)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=True)

    sumLoss = 0.
    sumError = 0.
    i = 0
    for data in dataloader:
        input = data[0].cuda()
        groundtruth = data[1].cuda()
        background = data[2].cuda()
        loss, error = testG(input, groundtruth, background, generator)
        sumLoss += loss
        sumError += error

        print("\r", end="")
        print("Testing progress: {}%: ".format(100*i/len(dataloader)), "▋" * (int(100*i/len(dataloader)) // 2), end="")
        sys.stdout.flush()
        i += 1

    print(f"\nAvgLoss:{sumLoss/len(dataloader)}  AvgError:{sumError/len(dataloader)}")




