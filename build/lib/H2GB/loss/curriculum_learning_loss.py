import torch
import torch.nn.functional as F
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_loss

import math
def training_scheduler(lam, t, T, scheduler='linear'):
    if scheduler == 'linear':
        return min(1, lam + (1 - lam) * t / T)
    elif scheduler == 'root':
        return min(1, math.sqrt(lam ** 2 + (1 - lam ** 2) * t / T))
    elif scheduler == 'geom':
        return min(1, 2 ** (math.log2(lam) - math.log2(lam) * t / T))


@register_loss('curriculum_learning_loss')
def curriculum_learning_loss(pred, true, epoch):
    """Curriculum Learning from https://github.com/LARS-research/CLGNN/tree/main.
    """
    if cfg.model.loss_fun == 'curriculum_learning_loss':
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            loss = F.nll_loss(pred, true, reduction="none")
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(), reduction="none")

        _, indices = torch.sort(loss, descending=False)
        epoch = 500 if epoch is None else epoch
        size = training_scheduler(0.5, epoch, 500, scheduler='linear')
        num_large_losses = int(len(loss) * size)
        selected_idx = indices[:num_large_losses]

        pred_, true_ = pred[selected_idx], true[selected_idx]
        # multiclass
        if pred.ndim > 1:
            loss = F.nll_loss(pred_, true_)
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred_, true_.float())

        return loss, pred
