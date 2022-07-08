import torch
from unet_vanilla import UNet, RoadSegmentData, DiceLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob, os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split




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

image_names = [img.split('/')[-1] for img in glob.glob("./Vanilla Dataset/training/images/*")]
image_path = './Vanilla Dataset/training/images/'
mask_path = './Vanilla Dataset/training/groundtruth/'
# if os.name == 'nt':
#     image_path = './mass-data/'
#     mask_path = './mass-data/'

train_image_names, validate_image_names = train_test_split(image_names, train_size=0.8)
train_data = RoadSegmentData(train_image_names, image_path, mask_path, img_transform)
validate_data = RoadSegmentData(validate_image_names, image_path, mask_path, img_transform)

# Hyperparameters
learning_rate = 1e-4
batch_size = 8
epochs = 10

# Fine-Tuning
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validate_dataloader = DataLoader(validate_data, batch_size=batch_size)
model = torch.load('./unet_trained')
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
    print(f'Epoch {(epoch+1)} Validation Loss: {validation_loss:5.4}')
    if (epoch+1)%10 == 0:
        torch.save(model, f'unet_ftm-{epoch+1}')
