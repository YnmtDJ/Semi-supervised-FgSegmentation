import os

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from models.modelUtils import initNetWork
from options.BaseOptions import BaseOptions
from test import testG
from utils.dataset import GeneratorDataset


def trainD(input_G, groundtruth, background, Generator, Discriminator, opt):
    loss_func = torch.nn.MSELoss()

    Generator.eval()
    Generator.requires_grad_(False)
    Discriminator.train()
    Discriminator.requires_grad_(True)

    torch.cuda.empty_cache()
    output_real = Discriminator(groundtruth)
    label_real = torch.ones(size=output_real.shape, dtype=torch.float).cuda()
    loss_real = loss_func(output_real, label_real)

    transform = transforms.Grayscale()
    predict = Generator((transform(input_G)-transform(background)).abs())
    output_fake = Discriminator(predict)
    label_fake = torch.zeros(size=output_fake.shape, dtype=torch.float).cuda()
    loss_fake = loss_func(output_fake, label_fake)
    loss = (loss_real + loss_fake)/2
    loss.backward()

    return loss.item()


def trainD2(input_G, background, Generator, Discriminator, opt):
    loss_func = torch.nn.MSELoss()

    Generator.eval()
    Generator.requires_grad_(False)
    Discriminator.train()
    Discriminator.requires_grad_(True)

    torch.cuda.empty_cache()
    transform = transforms.Grayscale()
    predict = Generator((transform(input_G) - transform(background)).abs())
    output_fake = Discriminator(predict)
    label_fake = torch.zeros(size=output_fake.shape, dtype=torch.float).cuda()
    loss_fake = loss_func(output_fake, label_fake)
    loss = loss_fake/2
    loss.backward()

    return loss.item()


def trainG(input_G, groundtruth, background, Generator, Discriminator, opt):
    loss_func = torch.nn.MSELoss()
    loss_bce = torch.nn.BCELoss()

    Generator.train()
    Generator.requires_grad_(True)
    # Freeze layers
    for layer in Generator.frozenLayers:
        for param in layer.parameters():
            param.requires_grad = False
    Discriminator.eval()
    Discriminator.requires_grad_(False)

    torch.cuda.empty_cache()
    transform = transforms.Grayscale()
    predict = Generator((transform(input_G) - transform(background)).abs())
    lossG = loss_bce(predict, groundtruth)
    # lossG = 0.
    output_fake = Discriminator(predict)
    label_real = torch.ones(size=output_fake.shape, dtype=torch.float).cuda()
    lossD = loss_func(output_fake, label_real)
    loss = lossG + opt.lambdaGAN1*lossD
    loss.backward()

    return loss.item()


def trainG2(input_G, background, Generator, Discriminator, opt):
    loss_func = torch.nn.MSELoss()
    loss_bce = torch.nn.BCELoss()

    Generator.train()
    Generator.requires_grad_(True)
    # Freeze layers
    for layer in Generator.frozenLayers:
        for param in layer.parameters():
            param.requires_grad = False
    Discriminator.eval()
    Discriminator.requires_grad_(False)

    torch.cuda.empty_cache()
    transform = transforms.Grayscale()
    predict = Generator((transform(input_G) - transform(background)).abs())
    output_fake = Discriminator(predict)
    label = predict.clone().detach()
    label = torch.round(label)
    mask = output_fake.gt(0.2)
    if mask.sum().item() > 0:
        predict = torch.masked_select(predict, mask)
        label = torch.masked_select(label, mask)
        lossG = loss_bce(predict, label)
    else:
        lossG = 0.
    label_real = torch.ones(size=output_fake.shape, dtype=torch.float).cuda()
    lossD = loss_func(output_fake, label_real)
    loss = opt.lambdaSemi*lossG + opt.lambdaGAN2*lossD
    loss.backward()

    return loss.item()


if __name__ == "__main__":
    opt = BaseOptions()
    if not opt.initialized:
        opt.initialize()
    opt = opt.parser.parse_args()
    # 加载网络
    generator, discriminator = initNetWork(opt)
    # 数据集
    datasetLabel = GeneratorDataset(os.path.join(opt.dataRoot, "label"), opt.inputSize)
    datasetUnLabel = GeneratorDataset(os.path.join(opt.dataRoot, "unlabel"), opt.inputSize)
    dataloaderLabel = DataLoader(dataset=datasetLabel, batch_size=opt.batchSize, shuffle=True)
    dataloaderUnLabel = DataLoader(dataset=datasetUnLabel, batch_size=opt.batchSize, shuffle=True)
    # 优化器
    optimizerD = torch.optim.Adam(discriminator.parameters(), opt.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), opt.lr)
    # tensorboard
    writer = SummaryWriter(opt.logRoot)
    iterLabel = iter(dataloaderLabel)
    iterUnLabel = iter(dataloaderUnLabel)
    for epoch in range(opt.startEpoch, opt.startEpoch+opt.numEpoch):
        print(f"epoch {epoch} begin")
        lossD = 0.
        lossD2 = 0.
        lossG = 0.
        lossG2 = 0.
        for i in range(len(dataloaderLabel)):
            iterNum = epoch*len(dataloaderLabel)+i
            try:
                inputLabel, groundtruth, background = next(iterLabel)
            except:
                iterLabel = iter(dataloaderLabel)
                inputLabel, groundtruth, background = next(iterLabel)
            try:
                inputUnLabel, _, bg = next(iterUnLabel)
            except:
                iterUnLabel = iter(dataloaderUnLabel)
                inputUnLabel, _, bg = next(iterUnLabel)
            if opt.cuda:
                inputLabel = inputLabel.cuda()
                groundtruth = groundtruth.cuda()
                background = background.cuda()
                inputUnLabel = inputUnLabel.cuda()
                bg = bg.cuda()

            optimizerD.zero_grad()
            optimizerG.zero_grad()
            if iterNum <= opt.preIterNum:
                lossD = trainD(inputLabel, groundtruth, background, generator, discriminator, opt)
                optimizerD.step()
                lossG = trainG(inputLabel, groundtruth, background, generator, discriminator, opt)
                optimizerG.step()
            else:
                lossD = trainD(inputLabel, groundtruth, background, generator, discriminator, opt)
                lossD2 = trainD2(inputUnLabel, bg, generator, discriminator, opt)
                optimizerD.step()
                lossG = trainG(inputLabel, groundtruth, background, generator, discriminator, opt)
                lossG2 = trainG2(inputUnLabel, bg, generator, discriminator, opt)
                optimizerG.step()
                writer.add_scalar("loss/lossD2", lossD2, iterNum)
                writer.add_scalar("loss/lossG2", lossG2, iterNum)

            writer.add_scalar("loss/lossD", lossD, iterNum)
            writer.add_scalar("loss/lossG", lossG, iterNum)

            if iterNum % 5 == 0:
                lossTest = testG(inputLabel, groundtruth, background, generator)
                writer.add_scalar("loss/lossTest", lossTest, iterNum)

            if iterNum % 100 == 0:
                print("Saving models")
                torch.save(generator.state_dict(), os.path.join(opt.modelRoot, str(generator) + ".pth"))
                torch.save(discriminator.state_dict(), os.path.join(opt.modelRoot, str(discriminator) + ".pth"))

