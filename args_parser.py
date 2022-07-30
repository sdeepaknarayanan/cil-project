import argparse

parser = argparse.ArgumentParser(description='Parameters for Model.')

parser.add_argument('--seed', type=int, dest='seed',default = 0)
parser.add_argument('--valid', dest='valid',type=float,  default = 0.2)
parser.add_argument('--train_image_path', type=str, dest='train_image_path')
parser.add_argument('--train_mask_path', type=str, dest='train_mask_path')
parser.add_argument('--resize', type=int, dest='resize',  default = None)
parser.add_argument('--batch_size', type=int, dest='batch_size',  default = 1)
parser.add_argument('--epochs', type=int, dest='epochs',  default = 200)
parser.add_argument('--load_path', type=str, dest='load_path', default=None)
parser.add_argument('--lr', type=float, dest='lr', default=1e-4)
parser.add_argument('--loss', type=str, dest='loss', default='dice')
parser.add_argument('--name', type=str, dest='name', default='experiment')
parser.add_argument('--model', type=str, dest='model', default='resunet_symmetric')
parser.add_argument('--pretr', type=int, dest='pretr', default=1)
parser.add_argument('--wdecay', type=float, dest='wd', default=0.)
