import argparse

parser = argparse.ArgumentParser(description='Parameters for Model.')
parser.add_argument('--seed', type=int, dest='seed',default = 0)
parser.add_argument('--valid', dest='valid',type=float,  default = 0.2)
parser.add_argument('--train_image_path', type=str, dest='train_image_path')
parser.add_argument('--train_mask_path', type=str, dest='train_mask_path')
parser.add_argument('--resize', type=int, dest='resize',  default = None)
parser.add_argument('--batch_size', type=int, dest='batch_size',  default = 1)
parser.add_argument('--epochs', type=int, dest='epochs',  default = 5)
parser.add_argument('--save_path', type=str, dest='save_path')


# python main_pt.py --train_mask_path ../training/groundtruth --train_image_path ../training/images --resize 256 --save_path ./

# bsub -n 2 -W 4:00 -R "rusage[mem=10000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py