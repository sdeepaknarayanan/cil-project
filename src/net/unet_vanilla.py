import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import os
import glob
import albumentations as A
import sys
import random
import argparse


# Model

class UNet(nn.Module):

    @staticmethod
    def UnetLayer(in_ch, out_ch, conv_dim = (3,3), pad = 1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, conv_dim, padding=pad, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, conv_dim, padding = pad, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU()
        )

    
    def __init__(self, chs = [3, 32, 64, 128, 256, 512]):
        super(UNet, self).__init__()
        out_chs = list(reversed(chs))[:-1]
        self.enc_layers = nn.ModuleList([UNet.UnetLayer(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
        self.dec_layers = nn.ModuleList([UNet.UnetLayer(out_chs[i], out_chs[i+1]) for i in range(len(out_chs) - 1)])
        self.final = nn.Sequential(
            nn.Conv2d(out_chs[-1],1,1),
            nn.Sigmoid()
            )
        self.pool = nn.MaxPool2d(2,2)
        # UpConv
        self.un_pool = nn.ModuleList([nn.ConvTranspose2d(out_chs[i], out_chs[i+1],2,2) for i in range(len(out_chs) - 1)])

    def forward(self,x):
        # Enconder
        encls = self.enc_layers
        enc_1 = encls[0](x)
        enc_2 = encls[1](self.pool(enc_1))
        enc_3 = encls[2](self.pool(enc_2))
        enc_4 = encls[3](self.pool(enc_3))

        bottleneck = encls[4](self.pool(enc_4))

        # Decoder
        outls = self.dec_layers
        un_pool = self.un_pool
        dec_1 = torch.cat((un_pool[0](bottleneck),enc_4),dim=1)
        dec_1 = outls[0](dec_1)
        dec_2 = torch.cat((un_pool[1](dec_1),enc_3),dim=1)
        dec_2 = outls[1](dec_2)
        dec_3 = torch.cat((un_pool[2](dec_2),enc_2),dim=1)
        dec_3 = outls[2](dec_3)
        dec_4 = torch.cat((un_pool[3](dec_3),enc_1),dim=1)
        dec_4 = outls[3](dec_4)

        final = self.final(dec_4)
        return final


# Reference: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
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

# Dataset input/preprocessing
class RoadSegmentData(Dataset):
    def __init__(self, image_names, image_path, mask_path, transform = None):
        self.image_names = image_names
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.num_data = len(self.image_names)

    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        image_filename = self.image_names[idx]
        image = cv2.imread(os.path.join(self.image_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, image_filename),  cv2.IMREAD_GRAYSCALE)/255
        mask = mask.reshape(tuple(list(mask.shape)+[1]))
        # print(image.shape, mask.shape)
        if self.transform is not None:
            transformed_image = self.transform(image = image,
                                              mask = mask)
            image = transformed_image['image']
            mask = transformed_image['mask']
        return image, mask

if __name__ =='__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train a UNet based model')
    parser.add_argument('-s','--save_path',type=str, default= './',help= 'Path where the model is to be saved')
    parser.add_argument('-e','--epochs', type=int, default = 200, help= 'Number of epochs to run')
    parser.add_argument('--seed', type=int, default=0, help='Seed for Psuedo-RNG')
    parser.add_argument('-i','--train_image_path', type=str, default='./gmap_data/images/', help='Path to train data images')
    parser.add_argument('-l', '--mask_image_path', type=str, default='./gmap_data/groundtruth/', help='Path to train data groundtruth')
    parser.add_argument('-b','--batch_size', type=int, default = 6, help='Batch size for training')
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

    img_transform = A.Compose(
        [
            A.VerticalFlip(),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True)
        ]
    )

    image_names = [img.split('/')[-1] for img in glob.glob(args.train_image_path+'/*')]
    image_path = args.train_image_path
    mask_path = args.mask_image_path

    train_image_names, test_validate_image_names = train_test_split(image_names, train_size=0.8)
    test_image_names, validate_image_names = train_test_split(test_validate_image_names, test_size=0.5)
    train_data = RoadSegmentData(train_image_names, image_path, mask_path, img_transform)
    test_data = RoadSegmentData(test_image_names, image_path, mask_path, img_transform)
    validate_data = RoadSegmentData(validate_image_names, image_path, mask_path, img_transform)

    # Hyperparameters
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    # Train loop
    best_val = np.inf
    early_stop_count = 0

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=batch_size)
    model = UNet()
    model = model.cuda()
    loss_fn = DiceLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
        scheduler.step(validation_loss)
        if validation_loss > best_val:
            early_stop_count+=1
            if early_stop_count == 50:
                print("No Improvement for 50 epochs! Exiting!")
                torch.save(model, save_path + f"/early_stop_{epoch}")
                sys.exit(0)
        else:
            early_stop_count = 0
            torch.save(model, save_path+f"/best_model")
        print(f'Epoch {(epoch+1)} Validation Loss: {validation_loss:5.4}')
        if (epoch+1)%5 == 0:
            torch.save(model, save_path + f'/Model Epoch {epoch+1}')
