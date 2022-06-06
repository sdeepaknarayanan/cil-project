## Evaluation Script

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

from resnet import UNetWithResnet50Encoder
from utils import *

from PIL import Image

import argparse

parser = argparse.ArgumentParser(description='Parameters for Model.')
parser.add_argument('--test_image_path', type=str, dest='test_image_path')
parser.add_argument('--test_mask_path', type=str, dest='test_mask_path')
parser.add_argument('--model_path', type=str, dest='model_path')
parser.add_argument('--resize', type=int, dest='resize',  default = 256)
parser.add_argument('--batch_size', type=int, dest='batch_size',  default = 1)
parser.add_argument('--cpu', type=int, dest='cpu', default=1)
parser.add_argument('--save_dir', type=str, dest='save_dir', default='preds')

args = parser.parse_args()

model = UNetWithResnet50Encoder()

if args.cpu == 1:
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(args.model_path))

test_images = glob.glob(args.test_image_path + '/*')
indices = np.arange(len(test_images))

final_test_list = []
for ix in indices:
    img = test_images[ix]
    final_test_list.append(img.split('/')[-1])

test_dataset = RoadSegmentData(image_names = final_test_list, 
                       image_path = args.test_image_path + '/',
                       mask_path = args.test_mask_path + '/',
                       transform = ToTensorV2(transpose_mask=True),
                      resize = args.resize)


test_loader = DataLoader(test_dataset, 
    batch_size = args.batch_size, 
    shuffle = False, 
    num_workers = 1)

val_acc = 0.
with torch.no_grad():
    for ix, (images, target) in enumerate(test_loader):
        print(images.shape, target.shape)
        if args.cpu:
            images = images.float()/255
            target = target.float()/255
        else:
            target = target.float().cuda()/255
            images = images.float().cuda()/255
            images = images / 255
        output = model(images)
        val_acc += metric_fn(output, target)

        output = output.detach().cpu().numpy().reshape((256, 256))
        output = cv2.resize(output, (400, 400))
        output = output * 255
        im = output.astype(np.uint8)
        plt.imshow(output)
        plt.show()
        im = Image.fromarray(im)
        im.save(f"preds/{final_test_list[ix]}")
        break
    val_acc /= (ix+1)  
    print(f"Testing Accuracy: {val_acc}")

