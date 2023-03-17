import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def bce_withlogits_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def bce_loss(output, target, class_weight=[2.57, 0.73]):
    if class_weight:
        weight = torch.zeros_like(target)
        weight[target==0] = class_weight[0]
        weight[target==1] = class_weight[1]
        return F.binary_cross_entropy(input=output, target=target,
                                      weight=weight)
    else:
        return F.binary_cross_entropy(input=output, target=target)