import torch
from torch import nn


class LabelSmoothingKLLoss(nn.Module):
    """
    Implement label smoothing. Basically this switches the original label distribution
    (which is effectively a dirac delta) to a mixture of the original and a uniform distribution.
    This helps the network to generalize better (and thus increase accuracy), but in turn will increase the perplexity.
    q'(k) = (1-smoothing)*q(k) + smoothing*uniform(k) = (1-smoothing)*q(k) + smoothing * 1/output_size
    """

    def __init__(self, size, padding_idx, smoothing=0.0, batch_multiplier=1.):
        super(LabelSmoothingKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.batch_multiplier = batch_multiplier

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and mask.size(-1) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach().requires_grad_(False)) / self.batch_multiplier
