import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataRoot', default='./dataset',
                                 help='path to images (should have input, groundtruth, background)')
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')
        self.parser.add_argument('--cuda', type=bool, default=True, help='using gpu or cpu')
        self.parser.add_argument('--startEpoch', type=int, default=0, help='epoch begin from this number')
        self.parser.add_argument('--numEpoch', type=int, default=7, help='epoch num')
        self.parser.add_argument('--inputSize', type=tuple, default=(256, 256), help='scale images to this size')
        self.parser.add_argument('--resultRoot', type=str, default='./result', help='images result save path')
        self.parser.add_argument('--modelRoot', type=str, default='./save_models', help='trained models save path')
        self.parser.add_argument('--logRoot', type=str, default='./log', help='training log save path')
        self.parser.add_argument('--preIterNum', type=int, default=-1,
                                 help='training label first for this number iterations')
        self.parser.add_argument('--lambdaGAN1', type=float, default=0, help='training label GAN loss weight')
        self.parser.add_argument('--lambdaGAN2', type=float, default=0, help='training unlabel GAN loss weight')
        self.parser.add_argument('--lambdaSemi', type=float, default=1, help='training unlabel Semi loss weight')
        self.parser.add_argument('--generator', type=str, default='FgSegNet',
                                 help='the type of generator(FgSegNet, UNet)')
        self.parser.add_argument('--discriminator', type=str, default='PatchDiscriminator',
                                 help='the type of discriminator(FCDiscriminator, PathDiscriminator)')
        self.initialized = True
