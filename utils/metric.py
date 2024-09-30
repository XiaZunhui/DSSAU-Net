import torch
import torch.nn.functional as F
import torch.nn as nn


class DSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: (BS,3,512,512)
        :param y_truth: (BS,512,512)
        :return:
        """
        y_pred_f = F.one_hot(y_pred.argmax(dim=1).long(), 3)
        y_pred_f = torch.flatten(y_pred_f, start_dim=0, end_dim=2)

        y_truth_f = F.one_hot(y_truth.long(), 3)
        y_truth_f = torch.flatten(y_truth_f, start_dim=0, end_dim=2)

        dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
                y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
                y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

        dice1.requires_grad_(False)
        dice2.requires_grad_(False)
        return (dice1 + dice2) / 2

class Acc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: (BS,2)
        :param y_truth: (BS,)
        :return:
        """
        y_pred_t = y_pred.argmax(dim=1)
        cnt = torch.sum(y_pred_t == y_truth)

        return cnt
