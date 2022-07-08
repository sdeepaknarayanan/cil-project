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

from resunet import build_resunet
from utils import *
from parser import parse


args = parser.parse_args()

print("Arguments: ", args)

save_path = f"{args.name}_Lr_{args.lr}_Ls_{args.loss}_Ep_{args.epochs}_BaSi_{args.batch_size}_Re_{args.resize}_Va_{args.valid}_Se_{args.seed}_Aug_{args.augment}"

if os.path.exists(save_path):
    print("Path Already Exists!!! Overwriting!!!")
else:
    os.makedirs(save_path)

f = open(save_path+"/log.txt", "w")
f.write(f"Model Loaded from Location: {args.load_path}\n")
f.flush()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Making Train/Val Lists")

if args.valid > 0:
    train_images = glob.glob(args.train_image_path + '/*')
    indices = np.arange(len(train_images))
    train_ixs = np.random.choice(indices, int((1 - args.valid) * len(train_images)), replace = False)
    valid_ixs = np.array(list(set(indices) - set(train_ixs)))

    final_train_list = []
    for ix in train_ixs:
        img = train_images[ix]
        final_train_list.append(img.split('/')[-1])
    print("Train/ Lists Done!!")

    final_valid_list = []
    for ix in valid_ixs:
        img = train_images[ix]
        final_valid_list.append(img.split('/')[-1])
    print("Valid/ Lists Done")

if args.augment is not None:

    train_transform = A.Compose(
    [
        A.VerticalFlip(),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ToTensorV2(transpose_mask=True),
    ]
)

else:

    train_transform = A.Compose(
        [
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
                       transform = ToTensorV2(transpose_mask=True),
                      resize = args.resize)

print("Dataset Class Created....")
train_loader = DataLoader(train_dataset, 
    batch_size = args.batch_size, 
    shuffle = True, 
    num_workers = 1)

valid_loader = DataLoader(valid_dataset, 
    batch_size = args.batch_size, 
    shuffle = True, 
    num_workers = 1)

print("Data Loaders Created!!!")
model = build_resunet().cuda()
print("Waiting for model to load.....")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

is_sig = False
if args.loss == 'dice':
    criterion = DiceLoss()
    sigmoid = nn.Sigmoid()
    is_sig = True
else:
    criterion = nn.BCEWithLogitsLoss()


# Training
best_val = -1

for epoch in range(args.epochs):
    model.train()

    epoch_loss = 0
    for i, (images, target) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.float().cuda()
        images = images / 255
        target = target.float().cuda()
        target = target / 255
        output = model(images)
        if is_sig == True:
            output = sigmoid(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()

    print(f"Loss for Epoch: {epoch} is {epoch_loss / (i + 1)}")

    f.write(f"Loss for Epoch: {epoch} is {epoch_loss / (i + 1)}\n")
    f.flush()

    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for ix, (x_val, y_val) in enumerate(valid_loader):
            output = model(x_val.float().cuda()/255)
            if is_sig == True:
                output = sigmoid(output)
            val_loss += criterion(output, y_val.float().cuda()/255)
            val_acc += metric_fn(output, y_val.float().cuda()/255, is_sig)
        val_acc /= (ix+1)
        val_loss/=(ix+1)

        val_loss = val_loss.detach().cpu().item()

        if val_acc > best_val:
            best_val = val_acc
            print(f"Validation Improved!!! Now Validation Acc: {best_val} Loss: {val_loss}")
            f.write(f"Validation Improved!!! Now Validation Acc: {best_val} Loss: {val_loss}\n")
            f.flush()
            torch.save(model.state_dict(), save_path+f"/model_epoch_{epoch}")

        else:
            print(f"Validation Accuracy: {val_acc} Val Loss: {val_loss} Best so far: {best_val}")
            f.write(f"Validation Accuracy: {val_acc} Val Loss: {val_loss} Best so far: {best_val}\n")
            f.flush()

torch.save(model.state_dict(), save_path+f"/model_last_{epoch}")
f.close()