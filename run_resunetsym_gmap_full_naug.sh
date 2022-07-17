python train.py --train_mask_path ../gmap_data/groundtruth --train_image_path ../gmap_data/images --epochs 200 --lr 1e-4 --loss dice --name resunet_sym_gmap_train_no_aug --model resunet_symmetric --aug 0 --batch_size 6 --val 0.2

