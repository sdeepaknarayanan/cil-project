import argparse
import csv
import pytorch_lightning as pl
import torch
import torchvision

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataset import *
from model import *
from util import *


def main(args):
    pl.utilities.seed.seed_everything(args.seed)

    model = ResNetModel.load_from_checkpoint(args.model_checkpoint)

    dataset = RoadSegmentationDataset(args.data_test_image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    trainer = Trainer(accelerator='auto')

    pred = trainer.predict(model, loader)
    pred = torch.cat(pred)
    pred = torch.where(pred > 0., 1., 0.)

    with open(args.output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['id', 'prediction'])
        for i in range(len(dataset)):
            for x in range(pred.shape[2]):
                for y in range(pred.shape[3]):
                    writer.writerow([f'{i + 144}_{y * 16}_{x * 16}', int(pred[i, 0, x, y])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--model-checkpoint', type=str, default=None)
    parser.add_argument('--output-file', type=str, default='submission.csv')
    parser.add_argument('--log-dir', type=str, default='lightning_logs/')
    parser.add_argument('--data-test-image-dir', type=str, default='data/cil-road-segmentation-2022/test/images/')

    args = parser.parse_args()
    main(args)
