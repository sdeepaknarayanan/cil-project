import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms.functional import normalize
from torchvision.utils import make_grid

from util import *


class ResNetModel(pl.LightningModule):

    def __init__(self, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.resnet = fcn_resnet50(num_classes=1, pretrained_backbone=True)
        self.pixel_to_patch = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=11, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=11, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=16, stride=16, padding=0),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=5, stride=1, padding='same'),
        )

    def forward(self, x):
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = self.resnet(x)['out']
        return x, self.pixel_to_patch(x.sigmoid())

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        x, y_pixel = batch
        y_patch = pixel_label_to_patch_label(y_pixel)
        y_pred_pixel, y_pred_patch = self(x)

        loss_patch = dice_loss_with_logits(y_pred_patch, y_patch)
        if optimizer_idx == 0:
            loss_pixel = dice_loss_with_logits(y_pred_pixel, y_pixel)
            loss = .5 * loss_pixel + .5 * loss_patch
            self.log('train_loss_pixel', loss_pixel)
            self.log('train_loss_patch', loss_patch)
            self.log('train_loss', loss)
            return loss
        else:
            return loss_patch

    def validation_step(self, batch, batch_idx):
        x, y_pixel = batch
        y_patch = pixel_label_to_patch_label(y_pixel)
        y_pred_pixel, y_pred_patch = self(x)
        y_pred_patch_from_pixel = pixel_label_to_patch_label(torch.where(y_pred_pixel > 0, 1., 0.))
        batch_size = x.shape[0]

        if batch_idx < 5:
            self.logger.experiment.add_image(f'val_batch_{batch_idx}', make_grid([
                make_grid(x, nrow=batch_size),
                make_grid(y_pixel, nrow=batch_size),
                make_grid(y_pred_pixel.sigmoid(), nrow=batch_size),
                make_grid(F.interpolate(y_patch, 400), nrow=batch_size),
                make_grid(F.interpolate(y_pred_patch.sigmoid(), 400), nrow=batch_size),
                make_grid(F.interpolate(y_pred_patch_from_pixel, 400), nrow=batch_size),
            ], nrow=1), self.current_epoch)

        accuracy_pixel = accuracy_score_with_logits(y_pred_pixel, y_pixel)
        f1_pixel = f1_score_with_logits(y_pred_pixel, y_pixel)
        accuracy_patch = accuracy_score_with_logits(y_pred_patch, y_patch)
        f1_patch = f1_score_with_logits(y_pred_patch, y_patch)
        self.log('val_accuracy_pixel', accuracy_pixel)
        self.log('val_f1_pixel', f1_pixel)
        self.log('val_accuracy_patch', accuracy_patch)
        self.log('val_f1_patch', f1_patch)
        return {'accuracy_pixel': accuracy_pixel, 'f1_pixel': f1_pixel, 'accuracy_patch': accuracy_patch, 'f1_patch': f1_patch}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        y_pred_pixel, y_pred_patch = self(x)
        y_pred_patch_from_pixel = pixel_label_to_patch_label(torch.where(y_pred_pixel > 0, 1., 0.))
        batch_size = x.shape[0]

        if batch_idx < 5:
            self.logger.experiment.add_image(f'pred_batch_{batch_idx}', make_grid([
                make_grid(x, nrow=batch_size),
                make_grid(y_pred_pixel.sigmoid(), nrow=batch_size),
                make_grid(F.interpolate(y_pred_patch.sigmoid(), 400), nrow=batch_size),
                make_grid(F.interpolate(y_pred_patch_from_pixel, 400), nrow=batch_size),
            ], nrow=1), self.current_epoch)

        return self(batch)[1]
    
    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return (
            torch.optim.Adam(self.resnet.parameters(), lr=self.hparams.learning_rate),
            torch.optim.Adam(self.pixel_to_patch.parameters(), lr=self.hparams.learning_rate)
        )
