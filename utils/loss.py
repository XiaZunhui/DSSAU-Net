import torch
import torch.nn.functional as F
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: one-hot encoded predictions
        :param y_truth: one-hot encoded predictions
        :return:
        """
        y_pred_f = torch.flatten(y_pred, start_dim=0, end_dim=2)
        y_truth_f = torch.flatten(y_truth, start_dim=0, end_dim=2)
        # print(y_pred_f.shape,y_truth_f.shape)
        dice1 = (2. * ((y_pred_f[:, 1:2] * y_truth_f[:, 1:2]).sum()) + self.smooth) / (
                y_pred_f[:, 1:2].sum() + y_truth_f[:, 1:2].sum() + self.smooth)
        dice2 = (2. * ((y_pred_f[:, 2:] * y_truth_f[:, 2:]).sum()) + self.smooth) / (
                y_pred_f[:, 2:].sum() + y_truth_f[:, 2:].sum() + self.smooth)

        dice1.requires_grad_(True)
        dice2.requires_grad_(True)
        return 1 - (dice1 + dice2) / 2


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_truth):
        """
        :param y_pred: (B,C,H,W)
        :param y_truth: (B,C,H,W)
        :return:
        """
        y_pred_f = y_pred.reshape(-1,1)
        y_truth_f = y_truth.reshape(-1,1)
        m = y_pred_f.shape[0]
        mse = (((y_pred_f-y_truth_f)**2).sum()) / (2.0*m)
        mse.requires_grad_(True)
        return mse


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, y_pred, y_truth):
        """
        :param y_pred:   one-hot
        :param y_truth:  one-hot
        :return:
        """
        y_pred_f = torch.flatten(y_pred.permute(0, 2, 3, 1), start_dim=0, end_dim=2)

        target_t = torch.flatten(y_truth, start_dim=0, end_dim=2)
        loss_ce = F.cross_entropy(y_pred_f, target_t.argmax(dim=1))
        return loss_ce

