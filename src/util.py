import random
import sklearn.metrics
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import random_split


class RandomChoiceAngleRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def random_split_with_ratio(dataset, ratio):
    len_1 = int(len(dataset) * ratio)
    len_2 = len(dataset) - len_1
    return random_split(dataset, (len_1, len_2))


def pixel_label_to_patch_label(x, patch_size=16, threshold=0.25):
    weight = torch.full((1, 1, patch_size, patch_size), 1 / patch_size**2).to(x)
    x = F.conv2d(x, weight, stride=patch_size)
    return torch.where(x > threshold, 1., 0.)


def dice_loss_with_logits(y_pred, y_true, smooth=1.):
    y_pred = y_pred.sigmoid()
    intersection = torch.sum(y_true * y_pred, axis=(1, 2, 3))
    dice = 2. * (intersection + smooth) / (torch.sum(y_true, axis=(1, 2, 3)) + torch.sum(y_pred, axis=(1, 2, 3)) + smooth)
    return 1. - dice.mean()


def f1_score_with_logits(y_pred, y_true):
    y_pred = torch.where(y_pred > 0, 1., 0.)
    batch_size = y_pred.shape[0]
    return sum(
        sklearn.metrics.f1_score(y_true[i].detach().cpu().flatten(), y_pred[i].detach().cpu().flatten())
        for i in range(batch_size)
    ) / batch_size


def accuracy_score_with_logits(y_pred, y_true):
    y_pred = torch.where(y_pred > 0, 1., 0.)
    batch_size = y_pred.shape[0]
    return sum(
        sklearn.metrics.accuracy_score(y_true[i].detach().cpu().flatten(), y_pred[i].detach().cpu().flatten())
        for i in range(batch_size)
    ) / batch_size