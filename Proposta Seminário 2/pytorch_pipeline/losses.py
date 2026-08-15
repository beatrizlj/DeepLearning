import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para datasets desbalanceados.
    Penaliza menos os exemplos fáceis e foca nos difíceis.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss(loss_name):
    """
    Factory function para a função de perda.
    Opções: 'ce' (CrossEntropy), 'focal' (FocalLoss), 'ce_smooth' (CrossEntropy com Label Smoothing).
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'ce_smooth':
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        valid_losses = ['ce', 'focal', 'ce_smooth']
        raise ValueError(f"Loss '{loss_name}' inválida! Opções válidas: {valid_losses}")
