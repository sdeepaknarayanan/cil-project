import torch
from unet_vanilla import UNet, RoadSegmentData
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob, os
from torch.utils.data import Dataset, DataLoader
import torchvision
import argparse

parser = argparse.ArgumentParser(description='Evaluate trained model with evaluation time augmentation')
parser.add_argument('model_path', type=str, nargs='+', help="Models to ensemble")
parser.add_argument('-i','--image_path', type=str, default= "./Vanilla Dataset/test/images", help = "Path to image")
parser.add_argument('-s','--save_path', type=str, default='./ensemble_output',help="Path where to save output")
parser.add_argument('-u', '--unet', action='store_true', help="Use this option when evaluating UNet model")
args = parser.parse_args()

img_transform = A.Compose(
        [
            ToTensorV2(transpose_mask=True)
        ]
    )

image_path = args.image_path
image_names = [img.split('/')[-1] for img in glob.glob(image_path+"/*")]
if os.name == 'nt':
    image_names = [img.split('\\')[-1] for img in image_names]
test_data = RoadSegmentData(sorted(image_names), image_path, image_path, img_transform)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

torch.cuda.empty_cache() 
model = torch.load(args.model_path)
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

model.cuda()
model.eval()

with torch.no_grad():
    for batch, (img,(x, _)) in enumerate(zip(image_names, test_dataloader)):
        x = x.float().cuda()
        if not args.unet:
            x = x/255
        y_pred = model(x)
        torchvision.utils.save_image(y_pred[0], save_path + f'/{img}')