import torch
from torch import nn
import torch.nn.functional as F


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
        true_dist.fill_(self.smoothing / (self.size - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and mask.size(-1) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach().requires_grad_(False)) / self.batch_multiplier


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, factor=1.0, batch_multiplier=1.):
        super(LabelSmoothingLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.factor = factor
        self.batch_multiplier = batch_multiplier

    def mean_ds(self, x: torch.Tensor, dim=None) -> torch.Tensor:
        return x.float().mean().type_as(x) if dim is None else x.float().mean(dim).type_as(x)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None):
        if masks is not None:
            outputs = outputs[masks]
            targets = targets[masks]

        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            losses = F.nll_loss(logits, targets, reduction="none")

        else:  # soft-labels
            losses = F.kl_div(logits, targets, reduction="none")
            losses = losses.float().sum(-1).type_as(losses)

        nll_loss = self.mean_ds(losses)
        if self.label_smoothing > 0:
            loss = nll_loss * (1 - self.label_smoothing) - self.mean_ds(logits) * self.label_smoothing
        else:
            loss = nll_loss

        return loss * self.factor
