import torch
import torch.nn.functional as F
from . import dice_loss


def compute_loss(loss_name, logits, labels):
    if loss_name == 'cross_entropy':
            probs = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(
                probs, labels, reduction='mean', ignore_index=255)

    if loss_name == 'dice':
        probs = F.softmax(logits, dim=1)
        loss = 0.
        for l in labels.unique():
            if l == 255:
                continue
            ind = labels != 255
            loss += dice_loss.dice_loss(probs[:, l][ind],
                                        (labels[ind] == l).long()) 
        return loss