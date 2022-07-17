python finetune.py --load_path exps/models_to_load/resnet_gmap --batch_size 4 --epochs 500 --lr 1e-4 --loss dice --name resnet_gmap_finetune --aug 1 --model resnet --pretr 1 --train_image_path ../training/images --train_mask_path ../training/groundtruth

