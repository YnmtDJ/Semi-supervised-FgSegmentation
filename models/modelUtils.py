import os
import torch
import torchvision.models

from models.FgSegNet.model import FgSegNet
from models.GAN.Discriminator import FCDiscriminator2, PatchDiscriminator
from models.Unet.unet import UNet


def initNetWork(opt):
    generator = None
    if opt.generator == 'FgSegNet':
        generator = initFgSegNet(opt)
    elif opt.generator == 'UNet':
        generator = initUNet(opt)
    discriminator = initDiscriminator(opt)
    if opt.cuda:
        return generator.cuda(), discriminator.cuda()
    else:
        return generator, discriminator


def initFgSegNet(opt):
    # 生成器
    generator = FgSegNet(6)
    # Freeze layers
    for layer in generator.frozenLayers:
        for param in layer.parameters():
            param.requires_grad = False

    # Load VGG-16 weights
    vgg_path = os.path.join(opt.modelRoot, 'vgg16-397923af.pth')
    if not os.path.exists(vgg_path):
        vgg16 = torchvision.models.vgg16(pretrained=True)
        torch.save(vgg16.state_dict(), vgg_path)
    vgg16_weights = torch.load(vgg_path)
    mapped_weights = {}
    for layer_count, (k_vgg, k_segnet) in enumerate(zip(vgg16_weights.keys(), generator.state_dict().keys())):
        # Last layer of VGG16 is not used in encoder part of the model
        if layer_count == 20:
            break
        if "features" in k_vgg:
            mapped_weights[k_segnet] = vgg16_weights[k_vgg]
            # print("Mapping {} to {}".format(k_vgg, k_segnet))

    try:
        generator.load_state_dict(mapped_weights)
        print("Loaded VGG-16 weights in SegNet !")
    except:
        pass  # Ignore missing keys

    G_path = os.path.join(opt.modelRoot, str(generator) + ".pth")
    if os.path.exists(G_path):
        generator.load_state_dict(torch.load(G_path))
    else:
        torch.save(generator.state_dict(), G_path)

    return generator


def initUNet(opt):
    # Generator
    generator = UNet(4)
    G_path = os.path.join(opt.modelRoot, str(generator) + ".pth")
    if os.path.exists(G_path):
        generator.load_state_dict(torch.load(G_path))
    else:
        torch.save(generator.state_dict(), G_path)

    return generator


def initDiscriminator(opt):
    # Discriminator
    discriminator = None
    if opt.discriminator == 'PatchDiscriminator':
        discriminator = PatchDiscriminator(1)
    elif opt.discriminator == 'FCDiscriminator2':
        discriminator = FCDiscriminator2(1, 64, opt.inputSize)
    D_path = os.path.join(opt.modelRoot, str(discriminator) + ".pth")
    if os.path.exists(D_path):
        discriminator.load_state_dict(torch.load(D_path))
    else:
        torch.save(discriminator.state_dict(), D_path)

    return discriminator

