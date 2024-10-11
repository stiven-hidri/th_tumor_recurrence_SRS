import torch


import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, reduction='mean'):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


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


