import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob, os
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
from models import build_resnet, build_resunet, build_resunet_symmetric, build_xception, UNet
import argparse

parser = argparse.ArgumentParser(description='Evaluate trained model with evaluation time augmentation')
parser.add_argument('model_path', type=str, help="Model to evaluate")
parser.add_argument('-i','--image_path', type=str, default= "./Vanilla Dataset/test/images", help = "Path to image")
parser.add_argument('-s','--save_path', type=str, default='./model_output',help="Path where to save output")
parser.add_argument('-u', '--unet', action='store_true', help="Use this option when evaluating UNet model")
args = parser.parse_args()

class RoadSegmentEvalAugmentData(Dataset):
    def __init__(self, image_names, image_path, transform = None):
        self.image_names = image_names
        self.image_path = image_path
        self.transform = transform
        self.num_data = len(self.image_names)*16

    def __len__(self):
        return self.num_data
        
    def __getitem__(self, idx):
        image_filename = self.image_names[idx//16]
        image = cv2.imread(os.path.join(self.image_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rotate_num = ((idx%16)%4)
        rcode = [0,cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][rotate_num]
        if rotate_num !=0:
            image = cv2.rotate(image,rcode)
        if self.transform is not None:
            transformed_image = self.transform(image = image)
            image = transformed_image['image']
        return image

img_transform = A.Compose(
        [
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10,p=0.75),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
            ToTensorV2(transpose_mask=True)
        ]
    )

image_path = args.image_path
image_names = [img.split('/')[-1] for img in glob.glob(image_path+"/*")]
if os.name == 'nt':
    image_names = [img.split('\\')[-1] for img in image_names]
test_data = RoadSegmentEvalAugmentData(sorted(image_names), image_path, img_transform)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

torch.cuda.empty_cache()
model = torch.load(args.model_path)
save_path = args.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'/inputs')
    os.makedirs(save_path+'/augmented')
    os.makedirs(save_path+'/final')

model.cuda()
model.eval()

image_names_augmented = sum([[f'{img.split(".")[0]}_{i}.png' for i in range(16)] for img in image_names],[])
with torch.no_grad():
    for batch, (img,x) in enumerate(zip(image_names_augmented, test_dataloader)):
        torchvision.utils.save_image(x[0]/255, save_path + f'/inputs/{img}')
        x = x.float().cuda()
        if not args.unet:
            x = x/255
        y_pred = model(x)
        torchvision.utils.save_image(y_pred[0], save_path+f'/augmented/{img}')

for img in image_names:
    final_masks = []
    # Reversing the rotation
    for i in range(16):
        mask = cv2.imread(save_path+f'/augmented/{img.split(".")[0]}_{i}.png',  cv2.IMREAD_GRAYSCALE)/255
        rotate_num = (4-(i%4))%4
        rcode = [0,cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][rotate_num]
        if rotate_num!=0:
            mask = cv2.rotate(mask, rcode)
        final_masks.append(mask)
    # Averaging the masks
    sm = final_masks[0]
    for i in range(15):
        sm+=final_masks[i+1]
    out_mask = (sm/16)
    out_mask = torch.Tensor(out_mask)
    torchvision.utils.save_image(out_mask, save_path + f'/final/{img}')
