import torch
import torch.utils.data
from torchvision.transforms import transforms


def testG(input_G, groundtruth, background, Generator):
    loss_func = torch.nn.BCELoss()

    Generator.eval()
    Generator.requires_grad_(False)

    torch.cuda.empty_cache()
    transform = transforms.Grayscale()
    predict = Generator((transform(input_G) - transform(background)).abs())
    loss = loss_func(predict, groundtruth)

    return loss.item()

