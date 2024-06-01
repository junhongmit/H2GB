import torch
import torch.nn as nn
import torch.nn.functional as F

import H2GB.graphgym.register as register
from H2GB.graphgym.config import cfg


def compute_loss(pred, true, epoch=None):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Grou

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    bce_loss_no_red = nn.BCEWithLogitsLoss(reduction='none')
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true, epoch)
        if value is not None:
            return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1 and true.ndim == 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary or multilabel
        else:
            # num_positives = true.sum().item()
            # num_negatives = len(true) - num_positives

            # # Calculate the weight for each class
            # weight_for_0 = num_positives / len(true)# * 3
            # weight_for_1 = num_negatives / len(true)

            # # Create a tensor of weights with the same shape as your labels
            # weights = true * weight_for_1 + (1 - true) * weight_for_0
            # weights = weights.to(pred.device)

            # true = true.float()

            # loss = bce_loss_no_red(pred, true)
            # loss = (loss * weights).mean()
            # return loss, torch.sigmoid(pred)
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    else:
        raise ValueError('Loss func {} not supported'.format(
            cfg.model.loss_fun))
