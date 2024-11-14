import torch


import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        # self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        loss = at*(1-pt)**self.gamma * BCE_loss
        
        # pred_sigmoid = torch.sigmoid(logits)
        
        # pt = (1 - pred_sigmoid) * targets + pred_sigmoid * (1 - targets)
        
        # focal_weight = (self.alpha * targets + (1 - self.alpha) *(1 - targets)) * pt.pow(self.gamma)
        # loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * focal_weight
        
        return loss.mean()

class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)
        return loss

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight, neg_weight, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = - (self.pos_weight * targets * torch.log(probs) +
                  self.neg_weight * (1 - targets) * torch.log(1 - probs))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        



