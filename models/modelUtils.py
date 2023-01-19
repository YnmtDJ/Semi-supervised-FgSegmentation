import os
import torch
from models.FgSegNet.model import FgSegNet
from models.GAN.Discriminator import FCDiscriminator2


def initNetWork(opt):
    # 生成器
    generator = FgSegNet(1)
    # Freeze layers
    for layer in generator.frozenLayers:
        for param in layer.parameters():
            param.requires_grad = False

    # Load VGG-16 weights
    vgg16_weights = torch.load(os.path.join(opt.modelRoot, 'vgg16-397923af.pth'))
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

    # 判别器
    discriminator = FCDiscriminator2(1, 64, opt.inputSize)
    D_path = os.path.join(opt.modelRoot, str(discriminator) + ".pth")
    if os.path.exists(D_path):
        discriminator.load_state_dict(torch.load(D_path))
    else:
        torch.save(discriminator.state_dict(), D_path)

    if opt.cuda:
        return generator.cuda(), discriminator.cuda()
    else:
        return generator, discriminator

