import torch
from unet_vanilla import UNet, RoadSegmentData, DiceLoss
from unetpp import UNetPP
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob, os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser(description='Finetuning script')
parser.add_argument('model_path', type=str, help='Save file of model to be finetuned')
parser.add_argument('-s','--save_path',type=str, default= './', help= 'Path where the model is to be saved')
parser.add_argument('-e','--epochs', type=int, default = 500, help= 'Number of epochs to run')
parser.add_argument('--seed', type=int, default=0, help='Seed for Psuedo-RNG')
parser.add_argument('-i','--train_image_path', type=str, default='./Vanilla Dataset/training/images/', help='Path to train data images')
parser.add_argument('-l', '--mask_image_path', type=str, default='./Vanilla Dataset/training/groundtruth/', help='Path to train data groundtruth')
parser.add_argument('-b','--batch_size', type=int, default = 8, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate to use')

args = parser.parse_args()

save_path = args.save_path

if os.path.exists(save_path):
    print("Path Already Exists. Files with the same name will be overwritten")
else:
    os.makedirs(save_path)

# Setting seed
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Data Loading/Preprocessing
img_transform = A.Compose(
        [
            A.VerticalFlip(),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.6),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.RandomRotate90(p=0.8),
            ToTensorV2(transpose_mask=True)
        ]
    )

image_names = [img.split('/')[-1] for img in glob.glob(args.train_image_path+'/*')]
image_path = args.train_image_path
mask_path = args.mask_image_path

train_image_names, validate_image_names = train_test_split(image_names, train_size=0.8)
train_data = RoadSegmentData(train_image_names, image_path, mask_path, img_transform)
validate_data = RoadSegmentData(validate_image_names, image_path, mask_path, img_transform)

# Hyperparameters
learning_rate = args.lr
batch_size = args.batch_size
epochs = args.epochs

# Train loop
best_val = np.inf
early_stop_count = 0

# Fine-Tuning
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_data, batch_size=batch_size)
model = torch.load(args.model_path)
model.cuda()
loss_fn = DiceLoss()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    model.train()
    for batch, (x, y_true) in enumerate(train_dataloader):
        x = x.float().cuda()
        y_true = y_true.float().cuda()
        y_pred = model(x)
        assert y_true.numel() == y_pred.numel()
        loss = loss_fn(y_true, y_pred)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch, (x, y_true) in enumerate(validate_dataloader):
            x = x.float().cuda()
            y_true = y_true.float().cuda()
            y_pred = model(x)
            validation_loss += loss_fn(y_true, y_pred).item()
    validation_loss /= (batch+1)
    if validation_loss > best_val:
        early_stop_count+=1
        if early_stop_count == 50:
            print("No Improvement for 50 epochs! Exiting!")
            torch.save(model, save_path + f"/early_stop_{epoch}")
            sys.exit(0)
    else:
        early_stop_count = 0
        torch.save(model, save_path+f"/best_model_finetuned")
    
    print(f'Epoch {(epoch+1)} Validation Loss: {validation_loss:5.4}')
    if (epoch+1)%5 == 0:
        torch.save(model, save_path + f'/Finetuned Model {epoch+1}')
