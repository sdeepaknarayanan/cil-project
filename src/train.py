import argparse
import pytorch_lightning as pl
import torch
import torchvision

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import *
from model import *
from util import *


def main(args):
    pl.utilities.seed.seed_everything(args.seed)

    if args.model_checkpoint:
        model = ResNetModel.load_from_checkpoint(args.model_checkpoint)
    else:
        model = ResNetModel(learning_rate=args.learning_rate)

    checkpoint_callbacks = [
        ModelCheckpoint(save_top_k=3, monitor='val_f1_patch', mode='max'),
        ModelCheckpoint(save_top_k=3, monitor='val_f1_pixel', mode='max'),
        ModelCheckpoint(save_top_k=3, monitor='train_loss', mode='min'),
        ModelCheckpoint(save_top_k=1, monitor='epoch', mode='max')
    ]

    if args.aug_epochs:
        dataset = RoadSegmentationDataset(args.data_aug_image_dir, args.data_aug_label_dir, augment=True)
        train_dataset, val_dataset = random_split_with_ratio(dataset, args.train_val_ratio)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        trainer = Trainer(
            accelerator='auto',
            max_epochs=args.aug_epochs,
            callbacks=checkpoint_callbacks,
            log_every_n_steps=10,
            fast_dev_run=args.dry_run,
        )
        trainer.fit(model, train_loader, val_loader)

    if args.epochs:
        dataset = RoadSegmentationDataset(args.data_train_image_dir, args.data_train_label_dir, augment=True)
        train_dataset, val_dataset = random_split_with_ratio(dataset, args.train_val_ratio)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        trainer = Trainer(
            accelerator='auto',
            max_epochs=args.epochs,
            callbacks=checkpoint_callbacks,
            log_every_n_steps=10,
            fast_dev_run=args.dry_run
        )
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--aug-epochs', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--train-val-ratio', type=float, default=0.8)
    # Other options
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model-checkpoint', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default='lightning_logs/')
    # Data directories
    parser.add_argument('--data-root-dir', type=str, default='data/')
    parser.add_argument('--data-aug-image-rel-dir', type=str, default='mass-data/new_images/')
    parser.add_argument('--data-aug-label-rel-dir', type=str, default='mass-data/new_labels/')
    parser.add_argument('--data-train-image-rel-dir', type=str, default='cil-road-segmentation-2022/training/images/')
    parser.add_argument('--data-train-label-rel-dir', type=str, default='cil-road-segmentation-2022/training/groundtruth/')

    args = parser.parse_args()
    args.data_aug_image_dir = args.data_root_dir + args.data_aug_image_rel_dir
    args.data_aug_label_dir = args.data_root_dir + args.data_aug_label_rel_dir
    args.data_train_image_dir = args.data_root_dir + args.data_train_image_rel_dir
    args.data_train_label_dir = args.data_root_dir + args.data_train_label_rel_dir

    main(args)
