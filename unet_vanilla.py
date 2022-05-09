import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import os
import glob
import albumentations as A


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

    def _forward_general(self,x):
        # Encoder
        layer_outs = []
        for layer in self.enc_layers:
            x = layer(x)
            layer_outs.append(x)
            x = self.pool(x)

        layer_outs.reverse()
        # Decoder
        for layer, copy, un_pool in zip(self.dec_layers, layer_outs, self.un_pool):
            x = un_pool(x)
            x = torch.cat((x,copy), dim=1)
            x = layer(x)

        x = self.final(x)
        return torch.sigmoid(x)

    def _forward_unrolled(self,x):
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

    # Edit this to change forward pass
    forward = _forward_unrolled

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

img_transform = A.Compose(
    [
        A.RandomRotate90(),
        ToTensorV2(transpose_mask=True)
    ]
)

image_names = [img.split('/')[-1] for img in glob.glob("./mass-data/new_images/*")]
image_path = './mass-data/new_images/'
mask_path = './mass-data/new_labels/'
if os.name == 'nt':
    image_path = './mass-data/'
    mask_path = './mass-data/'

train_image_names, test_validate_image_names = train_test_split(image_names, train_size=0.8)
test_image_names, validate_image_names = train_test_split(test_validate_image_names, test_size=0.5)
train_data = RoadSegmentData(train_image_names, image_path, mask_path, img_transform)
test_data = RoadSegmentData(test_image_names, image_path, mask_path, img_transform)
validate_data = RoadSegmentData(validate_image_names, image_path, mask_path, img_transform)

# Hyperparameters
learning_rate = 1e-4
batch_size = 16
epochs = 1

# Train loop
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_data, batch_size=batch_size)
model = UNet()
model = model.cuda()
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
    print(f"Epoch {(epoch+1)} Validation Loss: {validation_loss:5.4}")



# Test loop
test_dataloader = DataLoader(test_data, batch_size=batch_size)
model.eval()
with torch.no_grad():
    acc = 0
    for batch, (x, y_true) in enumerate(test_dataloader):
        x = x.float().cuda()
        y_true = y_true.float().cuda()
        y_pred = model(x)
        assert y_true.numel() == y_pred.numel()
        acc += (torch.ceil(torch.relu(y_pred - 0.5)) == y_true).sum().item()/y_true.numel()
    acc /= (batch+1)
    print(f"Test accuracy: {(acc*100):6.3} %")