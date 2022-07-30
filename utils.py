import cv2
import matplotlib.pyplot as plt
import albumentations as A
import os
import glob
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import copy
import numpy as np
import pickle as pkl
from sklearn.metrics import f1_score

class RoadSegmentData(Dataset):
    def __init__(self, image_names, image_path, mask_path, transform = None, resize = None):
        self.image_names = image_names
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.num_data = len(self.image_names)
        if resize is not None:
            self.resize = resize
        else:
            self.resize = None

    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        image_filename = self.image_names[idx]
        image = cv2.imread(os.path.join(self.image_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.mask_path, image_filename), cv2.IMREAD_UNCHANGED,
        )

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if self.resize is not None:
            image = cv2.resize(image, (self.resize, self.resize))
            mask = cv2.resize(mask, (self.resize, self.resize))
        
        mask = mask.reshape(tuple(list(mask.shape)+[1]))

        if self.transform is not None:
            transformed_image = self.transform(image = image,
                                              mask = mask)
            image = transformed_image['image']
            mask = transformed_image['mask']
        return image, mask
    
def visualize_augmentations(dataset, idx=0, samples=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    figure, ax = plt.subplots(nrows=samples, ncols=2, figsize=(10, 24))
    for i in range(samples):
        image, mask = dataset[idx]
        ax[i, 0].imshow(image)
        ax[i, 1].imshow(mask, interpolation="nearest")
        ax[i, 0].set_title("Augmented image")
        ax[i, 1].set_title("Augmented mask")
        ax[i, 0].set_axis_off()
        ax[i, 1].set_axis_off()
    plt.tight_layout()
    plt.show()
    

def restore(image):
    shape = image.shape
    new_arr = np.zeros((image.shape[1], image.shape[2], 3)).astype(np.uint8)
    for i in range(3):
        new_arr[:, :, i] = image[i, :, :]
    return new_arr

def metric_fn(y_pred, y_true, sigmoid = False):
    y_true = y_true.detach().cpu().numpy().reshape(-1)
    y_pred = y_pred.detach().cpu().numpy().reshape(-1)
    if sigmoid == True:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    y_true[y_true>0] = 1
    return f1_score(y_true, y_pred, average = 'micro')



class RoadSegmentTestData(Dataset):
    def __init__(self, image_names, image_path, transform = None, resize = None):
        self.image_names = image_names
        self.image_path = image_path
        self.transform = transform
        self.num_data = len(self.image_names)
        if resize is not None:
            self.resize = resize
        else:
            self.resize = None

    def __len__(self):
        return len(self.image_names)
        
    def __getitem__(self, idx):
        image_filename = self.image_names[idx]
        image = cv2.imread(os.path.join(self.image_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.resize is not None:
            image = cv2.resize(image, (self.resize, self.resize))
        
        if self.transform is not None:
            transformed_image = self.transform(image = image)
            image = transformed_image['image']
        return image

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1.0 - dsc