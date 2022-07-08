import os
import torch
import torchvision.transforms as T

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

from util import RandomChoiceAngleRotation


def get_all_file_paths(dir):
    filenames = sorted(os.listdir(dir))
    return [os.path.join(dir, filename) for filename in filenames]


class RoadSegmentationDataset(Dataset):

    def __init__(self, image_dir, label_dir=None, augment=False):
        self.image_paths = get_all_file_paths(image_dir)
        self.label_paths = get_all_file_paths(label_dir) if label_dir else None

        if augment:
            self.image_transforms = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            self.affine_transforms = T.Compose([
                RandomChoiceAngleRotation((0, 90, 180, 270)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ])
        else:
            self.image_transforms = None
            self.affine_transforms = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], mode=ImageReadMode.RGB) / 255

        if self.image_transforms:
            image = self.image_transforms(image)

        if self.label_paths:
            label = read_image(self.label_paths[idx], mode=ImageReadMode.GRAY) / 255
            
            if self.affine_transforms:
                # Apply the same transfomation to both image and label.
                stacked = torch.cat((image, label))
                stacked = self.affine_transforms(stacked)
                image, label = torch.split(stacked, (3, 1))
            
            return image, label
        else:
            if self.affine_transforms:
                image = self.affine_transforms(image)

            return image
