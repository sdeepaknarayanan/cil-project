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
import argparse

from resnet import UNetWithResnet50Encoder
from utils import *
from parser import parser


args = parser.parse_args()

print("Arguments: ", args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if args.valid > 0:
    train_images = glob.glob(args.train_image_path + '/*')
    indices = np.arange(len(train_images))
    train_ixs = np.random.choice(indices, int((1 - args.valid) * len(train_images)), replace = False)
    valid_ixs = np.array(list(set(indices) - set(train_ixs)))

    final_train_list = []
    for ix in train_ixs:
        img = train_images[ix]
        final_train_list.append(img.split('/')[-1])

    final_valid_list = []
    for ix in valid_ixs:
        img = train_images[ix]
        final_valid_list.append(img.split('/')[-1])

train_transform = A.Compose(
    [
        A.VerticalFlip(),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ToTensorV2(transpose_mask=True),
    ]
)

train_dataset = RoadSegmentData(image_names = final_train_list, 
                       image_path = args.train_image_path + '/',
                       mask_path = args.train_mask_path + '/',
                       transform = train_transform,
                      resize = args.resize)

valid_dataset = RoadSegmentData(image_names = final_valid_list, 
                       image_path = args.train_image_path + '/',
                       mask_path = args.train_mask_path + '/',
                       transform = None,
                      resize = args.resize)


train_loader = DataLoader(train_dataset, 
    batch_size = args.batch_size, 
    shuffle = True, 
    num_workers = 1)

valid_loader = DataLoader(valid_dataset, 
    batch_size = args.batch_size, 
    shuffle = True, 
    num_workers = 1)


model = UNetWithResnet50Encoder().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()



# Training
best_val = -1
for epoch in range(args.epochs):
    epoch_loss = 0
    model.train()
    for i, (images, target) in enumerate(train_loader):
        images = images.float().cuda()
        images = images / 255
        target = target.float().cuda()
        target = target / 255
        output = model(images)
        loss = criterion(output, target)
        epoch_loss+=loss.detach().cpu().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Loss for Epoch: ", epoch, " is ", epoch_loss/(i+1))
    # Validation
    model.eval()
    val_acc = 0.
    with torch.no_grad():
        for ix, (images, target) in enumerate(valid_loader):
            images = images.float().cuda()
            images = images / 255
            target = target.float().cuda()
            target = target / 255
            output = model(x_val)
            val_acc += metric_fn(output, target)
        val_acc /= (ix+1)
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Validation Accuracy Improved!!! Now Validation Accuracy: {best_val}")
