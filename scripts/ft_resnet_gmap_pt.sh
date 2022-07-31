python finetune.py --train_image_path ../training/images --train_mask_path ../training/groundtruth --batch_size 4 --epochs 500 --loss dice --name resnet_gmap_pt_finetune --model resnet --pretr 1 --load_path best_models_final/gmap/resnet_pt


