import os

import torch
import torch.utils.data
from numpy import sqrt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


from models.modelUtils import initNetWork
from options.BaseOptions import BaseOptions
from test import testG
from utils.dataset import GeneratorDataset


def trainD(input_G, groundtruth, background, Generator, Discriminator, transform,  opt):
    loss_func = torch.nn.MSELoss()

    Generator.eval()
    Generator.requires_grad_(False)
    Discriminator.train()
    Discriminator.requires_grad_(True)

    torch.cuda.empty_cache()
    output_real = Discriminator(groundtruth)
    label_real = torch.ones(size=output_real.shape, dtype=torch.float).cuda()
    loss_real = loss_func(output_real, label_real)

    predict = Generator(torch.cat([input_G, background], dim=1))
    # predict = Generator(input_G)
    output_fake = Discriminator(predict)
    label_fake = torch.zeros(size=output_fake.shape, dtype=torch.float).cuda()
    loss_fake = loss_func(output_fake, label_fake)
    loss = (loss_real + loss_fake)/2
    loss.backward()

    return loss.item()


def trainD2(input_G, background, Generator, Discriminator, transform, opt):
    loss_func = torch.nn.MSELoss()

    Generator.eval()
    Generator.requires_grad_(False)
    Discriminator.train()
    Discriminator.requires_grad_(True)

    torch.cuda.empty_cache()
    predict = Generator(torch.cat([input_G, background], dim=1))
    # predict = Generator(input_G)
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
    Discriminator.eval()
    Discriminator.requires_grad_(False)

    torch.cuda.empty_cache()
    predict = Generator(torch.cat([transform_w(input_G), transform_w(background)], dim=1))
    # predict = Generator(input_G)
    # mask = groundtruth.eq(0) | groundtruth.eq(1)
    # predict = torch.masked_select(predict, mask)
    # groundtruth = torch.masked_select(groundtruth, mask)
    lossG = loss_bce(predict, groundtruth)
    # lossG = 0.
#     output_fake = Discriminator(predict)
#     label_real = torch.ones(size=output_fake.shape, dtype=torch.float).cuda()
#     lossD = loss_func(output_fake, label_real)
    loss = lossG
    loss.backward()

    return loss.item()


def trainG2(input_G, background, Generator, Discriminator, opt):
    loss_func = torch.nn.MSELoss()
    loss_bce = torch.nn.BCELoss(reduction='none')

    Generator.train()
    Generator.requires_grad_(True)
    Discriminator.eval()
    Discriminator.requires_grad_(False)

    torch.cuda.empty_cache()
    conf_map = (input_G - background).abs().sum(dim=1, keepdim=True).to(input_G.device)
    predict = Generator(torch.cat([transform_w(input_G), transform_w(background)], dim=1))
    # predict = Generator(input_G)
    # output_fake = Discriminator(predict)
    mask = predict.gt(0.95) | predict.lt(0.05)
    mask2 = conf_map.gt(1) | conf_map.lt(0.05)
    mask = mask & mask2
    label = torch.round(predict).clone().detach()
    predict2 = Generator(torch.cat([transform_s(input_G), transform_s(background)], dim=1))
    # predict2 = Generator(transform(input_G))
    lossG = (loss_bce(predict2, label) * mask).mean()
    # label_real = torch.ones(size=output_fake.shape, dtype=torch.float).cuda()
    # lossD = loss_func(output_fake, label_real)
    loss = opt.lambdaSemi*lossG
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
    N = len(datasetLabel)
    datasetLabel, datasetTest = random_split(datasetLabel, [int(N*0.7), N-int(N*0.7)])
    datasetUnLabel = GeneratorDataset(os.path.join(opt.dataRoot, "unlabel"), opt.inputSize)
    dataloaderLabel = DataLoader(dataset=datasetLabel, batch_size=opt.batchSize, shuffle=True)
    dataloaderTest = DataLoader(dataset=datasetTest, batch_size=opt.batchSize, shuffle=True)
    dataloaderUnLabel = DataLoader(dataset=datasetUnLabel, batch_size=opt.batchSize, shuffle=True)
    # 优化器
    optimizerD = torch.optim.Adam(discriminator.parameters(), opt.lr)
    optimizerG = torch.optim.Adam(generator.parameters(), opt.lr)
    # transformer
    transform_s = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
    transform_w = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    # tensorboard
    writer = SummaryWriter(opt.logRoot)
    iterLabel = iter(dataloaderLabel)
    iterTest = iter(dataloaderTest)
    iterUnLabel = iter(dataloaderUnLabel)
    for epoch in range(opt.startEpoch, opt.startEpoch+opt.numEpoch):
        print(f"epoch {epoch} begin")
        for i in range(len(dataloaderLabel)):
            iterNum = epoch*len(dataloaderLabel)+i
            try:
                inputLabel, groundtruth, background = next(iterLabel)
            except:
                iterLabel = iter(dataloaderLabel)
                inputLabel, groundtruth, background = next(iterLabel)
            if opt.cuda:
                inputLabel = inputLabel.cuda()
                groundtruth = groundtruth.cuda()
                background = background.cuda()

            optimizerD.zero_grad()
            optimizerG.zero_grad()
            if iterNum <= opt.preIterNum:  # train with label
                # lossD = trainD(inputLabel, groundtruth, background, generator, discriminator, transform, opt)
                # optimizerD.step()
                lossG = trainG(inputLabel, groundtruth, background, generator, discriminator, opt)
                optimizerG.step()
            else:  # train with label and unlabel
                try:
                    inputUnLabel, _, bg = next(iterUnLabel)
                except:
                    iterUnLabel = iter(dataloaderUnLabel)
                    inputUnLabel, _, bg = next(iterUnLabel)
                if opt.cuda:
                    inputUnLabel = inputUnLabel.cuda()
                    bg = bg.cuda()
                # lossD = trainD(inputLabel, groundtruth, background, generator, discriminator, transform, opt)
                # lossD2 = trainD2(inputUnLabel, bg, generator, discriminator, opt)
                # optimizerD.step()
                lossG = trainG(inputLabel, groundtruth, background, generator, discriminator, opt)
                lossG2 = trainG2(inputUnLabel, bg, generator, discriminator, opt)
                optimizerG.step()
                # writer.add_scalar("loss/lossD2", lossD2, iterNum)
                writer.add_scalar("loss/lossG2", lossG2, iterNum)

            # writer.add_scalar("loss/lossD", lossD, iterNum)
            writer.add_scalar("loss/lossG", lossG, iterNum)

            if iterNum % 10 == 0:  # test generator
                try:
                    inputTest, gtTest, bgTest = next(iterTest)
                except:
                    iterTest = iter(dataloaderTest)
                    inputTest, gtTest, bgTest = next(iterTest)
                if opt.cuda:
                    inputTest = inputTest.cuda()
                    gtTest = gtTest.cuda()
                    bgTest = bgTest.cuda()
                lossTest, accuracy, _, _, _, _ = testG(inputTest, gtTest, bgTest, generator)
                writer.add_scalar("loss/lossTest", lossTest, iterNum)
                writer.add_scalar("metric/accTest", accuracy, iterNum)

            if iterNum % 100 == 0:  # save models
                print("Saving models")
                torch.save(generator.state_dict(), os.path.join(opt.modelRoot, str(generator) + ".pth"))
                torch.save(discriminator.state_dict(), os.path.join(opt.modelRoot, str(discriminator) + ".pth"))

    print("training models finished")
