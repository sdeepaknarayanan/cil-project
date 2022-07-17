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
from models import build_resnet, build_resunet, build_resunet_symmetric
from utils import *
from parser import parser
import json

args = parser.parse_args()

print("Arguments: ", args)

save_path = f"exps/{args.name}_Lr_{args.lr}_Ls_{args.loss}_Ep_{args.epochs}_BaSi_{args.batch_size}_Re_{args.resize}_Va_{args.valid}_Se_{args.seed}_Aug_{args.augment}"

if os.path.exists(save_path):
    print("Path Already Exists!!! Overwriting!!!")
else:
    os.makedirs(save_path)

with open(save_path+'/config.json', 'w') as fp:
    json.dump(args.__dict__, fp)

f = open(save_path+"/finetune_log.txt", "w")

np.random.seed(args.seed)
torch.manual_seed(args.seed)

print("Making Train/Val Lists")


train_images = glob.glob(args.train_image_path + '/*')
indices = np.arange(len(train_images))
train_ixs = np.random.choice(indices, int((1 - args.valid) * len(train_images)), replace = False)

if args.valid > 0:
    valid_ixs = np.array(list(set(indices) - set(train_ixs)))

final_train_list = []
for ix in train_ixs:
    img = train_images[ix]
    final_train_list.append(img.split('/')[-1])
print("Train/ Lists Done!!")

if args.valid > 0:
    final_valid_list = []
    for ix in valid_ixs:
        img = train_images[ix]
        final_valid_list.append(img.split('/')[-1])
    print("Valid/ Lists Done")

if args.augment is not None:

    train_transform = A.Compose(
    [
        A.VerticalFlip(),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.6),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.RandomRotate90(p=0.8),
        ToTensorV2(transpose_mask=True)

    ]
)

else:

    train_transform = A.Compose(
        [
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True),
        ]
    )

train_dataset = RoadSegmentData(image_names = final_train_list, 
                       image_path = args.train_image_path + '/',
                       mask_path = args.train_mask_path + '/',
                       transform = train_transform,
                      resize = args.resize)
if args.valid > 0:
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

if args.valid > 0:
    valid_loader = DataLoader(valid_dataset, 
        batch_size = args.batch_size, 
        num_workers = 1)

model = torch.load(args.load_path).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

is_sig = False
if args.loss == 'dice':
    criterion = DiceLoss()
    sigmoid = nn.Sigmoid()
    is_sig = True
else:
    criterion = nn.BCEWithLogitsLoss()


# Training
best_val = np.inf
early_stop_count = 0

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
    if args.valid > 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ix, (x_val, y_val) in enumerate(valid_loader):
                output = model(x_val.float().cuda()/255)
                if is_sig == True:
                    output = sigmoid(output)
                val_loss += criterion(output, y_val.float().cuda()/255)
            val_loss/=(ix+1)

            val_loss = val_loss.detach().cpu().item()

            if val_loss > best_val:
                early_stop_count+=1
                if early_stop_count == 30:
                    print("No Improvement for 30 epochs! Exiting!")
                    torch.save(model, save_path + f"/last_ft_early_stop_{epoch}")
                    sys.exit(0)

            if val_loss <= best_val:
                best_val = val_loss
                early_stop_count = 0
                print(f"Validation Improved!!! New Loss: {val_loss}")
                f.write(f"Validation Improved!!! New Loss: {val_loss}\n")
                f.flush()
                torch.save(model, save_path+f"/best_ft_model_{epoch}")

            else:
                print(f"Val Loss: {val_loss} Best so far: {best_val}")
                f.write(f"Val Loss: {val_loss} Best so far: {best_val}\n")
                f.flush()

    if epoch % 10 == 0:
        torch.save(model, save_path+f"/model_ft_current_{epoch}")

torch.save(model, save_path+f"/model_ft_last_{epoch}")
f.close()
