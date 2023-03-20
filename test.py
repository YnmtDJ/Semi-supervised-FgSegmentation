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


def calAccuracy(predict, groundtruth):
    batch, _, h, w = groundtruth.shape
    accuracy = predict.eq(groundtruth).sum()/(batch*h*w)
    return accuracy.item()


def calPrecision(predict, groundtruth):
    mask = predict.eq(1)
    precision = torch.masked_select(groundtruth, mask).sum()/mask.sum()
    return precision.item()


def calRecall(predict, groundtruth):
    mask = groundtruth.eq(1)
    recall = torch.masked_select(predict, mask).sum()/mask.sum()
    return recall.item()


def calF1_Score(predict, groundtruth):
    precision = calPrecision(predict, groundtruth)
    recall = calRecall(predict, groundtruth)
    return 2*precision*recall/(precision + recall)


def calMIoU(predict, groundtruth):
    mask1 = predict.eq(1)
    mask2 = groundtruth.eq(1)
    MIoU = 0.5*(mask1 & mask2).sum() / (mask1 | mask2).sum()
    mask1 = predict.eq(0)
    mask2 = groundtruth.eq(0)
    MIoU += 0.5*(mask1 & mask2).sum() / (mask1 | mask2).sum()
    return MIoU.item()


def testG(input_G, groundtruth, background, Generator):
    """
    Test the model, using different metrics.
    Returns: loss, accuracy, precision, recall, f1_score, MIoU
    """
    loss_func = torch.nn.BCELoss()
    loss_focal = BinaryFocalLoss()

    Generator.eval()
    Generator.requires_grad_(False)

    torch.cuda.empty_cache()
    predict = Generator(torch.cat([input_G, background], dim=1))
    # predict = Generator(input_G)
    loss = loss_func(predict, groundtruth).item()
    predict = torch.round(predict)
    accuracy = calAccuracy(predict, groundtruth)
    precision = calPrecision(predict, groundtruth)
    recall = calRecall(predict, groundtruth)
    f1_score = calF1_Score(predict, groundtruth)
    MIoU = calMIoU(predict, groundtruth)

    return loss, accuracy, precision, recall, f1_score, MIoU


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
    sumAcc = 0.
    sumPre = 0.
    sumRec = 0.
    sumF1 = 0.
    sumMIoU = 0.
    i = 0
    for data in dataloader:
        input = data[0].cuda()
        groundtruth = data[1].cuda()
        background = data[2].cuda()
        loss, accuracy, precision, recall, f1_score, MIoU = testG(input, groundtruth, background, generator)
        sumLoss += loss
        sumAcc += accuracy
        sumPre += precision
        sumRec += recall
        sumF1 += f1_score
        sumMIoU += MIoU
        print("\r", end="")
        print("Testing progress: {}%: ".format(100*i/len(dataloader)), "▋" * (int(100*i/len(dataloader)) // 2), end="")
        sys.stdout.flush()
        i += 1

    print(f"\nAvgLoss:{sumLoss/len(dataloader)}  AvgAcc:{sumAcc/len(dataloader)}  "
          f"AvgPre:{sumPre/len(dataloader)}  AvgRec:{sumRec/len(dataloader)}  "
          f"AvgF1:{sumF1/len(dataloader)}  AvgMIoU:{sumMIoU/len(dataloader)}")




