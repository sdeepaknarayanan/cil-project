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


# Model
# Reference: https://github.com/4uiiurz1/pytorch-nested-unet/blob/557ea02f0b5d45ec171aae2282d2cd21562a633e/archs.py

class UNetPP(nn.Module):

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
        super(UNetPP, self).__init__()

        self.conv0_0 = UNetPP.UnetLayer(chs[0], chs[1])
        self.conv1_0 = UNetPP.UnetLayer(chs[1], chs[2])
        self.conv2_0 = UNetPP.UnetLayer(chs[2], chs[3])
        self.conv3_0 = UNetPP.UnetLayer(chs[3], chs[4])
        self.conv4_0 = UNetPP.UnetLayer(chs[4], chs[5])

        self.conv0_1 = UNetPP.UnetLayer(chs[1]+chs[2], chs[1])
        self.conv1_1 = UNetPP.UnetLayer(chs[2]+chs[3], chs[2])
        self.conv2_1 = UNetPP.UnetLayer(chs[3]+chs[4], chs[3])
        self.conv3_1 = UNetPP.UnetLayer(chs[4]+chs[5], chs[4])

        self.conv0_2 = UNetPP.UnetLayer(chs[1]*2+chs[2], chs[1])
        self.conv1_2 = UNetPP.UnetLayer(chs[2]*2+chs[3], chs[2])
        self.conv2_2 = UNetPP.UnetLayer(chs[3]*2+chs[4], chs[3])

        self.conv0_3 = UNetPP.UnetLayer(chs[1]*3+chs[2], chs[1])
        self.conv1_3 = UNetPP.UnetLayer(chs[2]*3+chs[3], chs[2])

        self.conv0_4 = UNetPP.UnetLayer(chs[1]*4+chs[2], chs[1])
        
        self.final = nn.Sequential(
            nn.Conv2d(chs[1],1,1),
            nn.Sigmoid()
            )
        self.pool = nn.MaxPool2d(2,2)
        # Upsample
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,x):
        # Enconder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


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

    img_transform = A.Compose(
        [
            A.VerticalFlip(),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.RandomRotate90(),
            ToTensorV2(transpose_mask=True)
        ]
    )

    image_names = [img.split('/')[-1] for img in glob.glob("./gmap_data/images/*")]
    image_path = './gmap_data/images/'
    mask_path = './gmap_data/groundtruth/'
    if os.name == 'nt':
        image_path = './gmap_data/'
        mask_path = './gmap_data/'

    train_image_names, test_validate_image_names = train_test_split(image_names, train_size=0.8)
    test_image_names, validate_image_names = train_test_split(test_validate_image_names, test_size=0.5)
    train_data = RoadSegmentData(train_image_names, image_path, mask_path, img_transform)
    test_data = RoadSegmentData(test_image_names, image_path, mask_path, img_transform)
    validate_data = RoadSegmentData(validate_image_names, image_path, mask_path, img_transform)

    # Hyperparameters
    learning_rate = 1e-4
    batch_size = 8
    epochs = 100

    # Train loop
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validate_dataloader = DataLoader(validate_data, batch_size=batch_size)
    model = UNetPP()
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
        print(f'Epoch {(epoch+1)} Validation Loss: {validation_loss:5.4}')
        if (epoch+1)%10 == 0:
            torch.save(model, f'Model Epoch {epoch+1}')


    # Test loop
    test_dataloader = DataLoader(test_data, batch_size=1)
    model.eval()
    with torch.no_grad():
        acc = 0
        for batch, (x, y_true) in enumerate(test_dataloader):
            x = x.float().cuda()
            y_true = y_true.float().cuda()
            y_pred = model(x)
            assert y_true.numel() == y_pred.numel()
            acc += (torch.ceil(torch.relu(y_pred - 0.5)) == y_true).sum().item()/y_true.numel()
        np.save('true_img',y_true[0,:,:,:].to('cpu').numpy())
        np.save('pred_img',y_pred[0,:,:,:].to('cpu').numpy())
        acc /= (batch+1)
        print(f"Test accuracy: {(acc):6.5} %")