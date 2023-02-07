import os

import torch
import torch.utils.data
import torchvision
from numpy import sqrt
from torchvision.transforms import transforms
from torchvision.utils import make_grid

from models.modelUtils import initNetWork
from options.BaseOptions import BaseOptions
from utils.dataset import GeneratorDataset


def generateImageLabel(generator, opt):
    if not os.path.isdir(os.path.join(opt.dataRoot, 'label')):
        os.makedirs(os.path.join(opt.dataRoot, 'label'))

    dataset = GeneratorDataset(os.path.join(opt.dataRoot, 'label'), opt.inputSize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    num = 0
    generator.eval()
    generator.requires_grad_(False)
    for input, groundtruth, background in dataloader:
        input = input.cuda()
        groundtruth = groundtruth.cuda()
        background = background.cuda()
        conf_map = input - background
        output = generator(torch.cat([input, background], dim=1))
        output = output.round()

        input = make_grid(input, nrow=opt.batchSize)
        background = make_grid(background, nrow=opt.batchSize)
        groundtruth = make_grid(groundtruth, nrow=opt.batchSize)
        # conf_map = make_grid(conf_map, nrow=opt.batchSize)
        output = make_grid(output, nrow=opt.batchSize)
        img_grid = torch.cat((input, background, groundtruth, output), 1)
        path = os.path.join(opt.resultRoot, 'label', str(num).zfill(6) + ".png")
        num += 1
        torchvision.utils.save_image(img_grid, path)

    print(f"Fake Image Generated")


def generateImageUnLabel(generator, opt):
    if not os.path.isdir(os.path.join(opt.dataRoot, 'unlabel')):
        os.makedirs(os.path.join(opt.dataRoot, 'unlabel'))

    dataset = GeneratorDataset(os.path.join(opt.dataRoot, 'unlabel'), opt.inputSize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    num = 0
    generator.eval()
    generator.requires_grad_(False)
    for input, _, background in dataloader:
        input = input.cuda()
        background = background.cuda()
        conf_map = input - background
        output = generator(torch.cat([input, background], dim=1))
        output = output.round()

        input = make_grid(input, nrow=opt.batchSize)
        background = make_grid(background, nrow=opt.batchSize)
        # conf_map = make_grid(conf_map, nrow=opt.batchSize)
        output = make_grid(output, nrow=opt.batchSize)
        img_grid = torch.cat((input, background, output), 1)
        path = os.path.join(opt.resultRoot, 'unlabel', str(num).zfill(6) + ".png")
        num += 1
        torchvision.utils.save_image(img_grid, path)

    print(f"Fake Image Generated")


if __name__ == '__main__':
    opt = BaseOptions()
    if not opt.initialized:
        opt.initialize()
    opt = opt.parser.parse_args()
    opt.modelRoot = '../save_models'
    opt.dataRoot = '../dataset'
    opt.resultRoot = '../result'
    generator, discriminator = initNetWork(opt)

    generateImageLabel(generator, opt)  # 保存图像结果
    generateImageUnLabel(generator, opt)

