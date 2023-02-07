import torch
from torch import nn


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.size_average = size_average
        assert alpha < 1
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)
        batchSize = inputs.shape[0]
        inputs = inputs.view(batchSize, -1)
        targets = targets.view(batchSize, -1)
        loss_pos = -torch.log(inputs)*torch.pow(1-inputs, self.gamma)*targets*self.alpha
        loss_neg = -torch.log(1-inputs)*torch.pow(inputs, self.gamma)*(1-targets)*(1-self.alpha)
        loss = loss_pos + loss_neg
        if self.size_average:
            return loss.mean()
        return loss.sum()
