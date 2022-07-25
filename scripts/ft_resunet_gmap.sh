python finetune.py --train_image_path ../training/images --train_mask_path ../training/groundtruth --batch_size 4 --epochs 700 --loss dice --name resunet_gmap_finetune --aug 1 --model resunet --pretr 0 --load_path best_models_final/gmap/resunet


