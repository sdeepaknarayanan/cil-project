import cv2
import os
import glob
import torchvision
import torch
import argparse

parser = argparse.ArgumentParser(description='Ensemble outputs together')
parser.add_argument('output_paths', type=str, nargs='+', help="Outputs of models to ensemble")
parser.add_argument('-s','--save_path', type=str, default='./ensemble',help="Path where to save output")
args = parser.parse_args()

output_paths = args.output_paths
save_path = args.save_path

if os.path.exists(save_path):
    print("Path Already Exists. Files with the same name will be overwritten")
else:
    os.makedirs(save_path)


image_names = [img.split('/')[-1] for img in glob.glob(output_paths[0]+'/*')]
if os.name == 'nt':
    image_names = [img.split('\\')[-1] for img in image_names]

for img in image_names:
    li = []
    for output_path in output_paths:
        mask = cv2.imread(output_path+f'/{img}',  cv2.IMREAD_GRAYSCALE)/255
        li.append(mask)
    sm = li[0]
    for i in range(len(output_paths)-1):
        sm+=li[i+1]
    out_mask = (sm/len(output_paths))
    out_mask = torch.Tensor(out_mask)
    torchvision.utils.save_image(out_mask, save_path + f'/{img}')