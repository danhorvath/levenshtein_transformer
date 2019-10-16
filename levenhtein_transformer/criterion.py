import torch
from torch import nn
import torch.nn.functional as F


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

        return loss * self.factor / self.batch_multiplier
