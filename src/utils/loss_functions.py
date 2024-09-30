import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # Probability of the true class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        

class BinaryCrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(BinaryCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)

        if self.reduction == 'mean':
            return BCE_loss.mean()
        elif self.reduction == 'sum':
            return BCE_loss.sum()
        else:
            return BCE_loss


class WeightedBinaryCrossEntropy(torch.nn.Module):
    def __init__(self, pos_weight = torch.tensor([3.0]), reduction='mean'):
        super(WeightedBinaryCrossEntropy, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)(inputs, targets)

        if self.reduction == 'mean':
            return BCE_loss.mean()
        elif self.reduction == 'sum':
            return BCE_loss.sum()
        else:
            return BCE_loss

