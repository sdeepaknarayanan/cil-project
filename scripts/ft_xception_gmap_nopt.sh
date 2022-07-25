python finetune.py --train_image_path ../training/images --train_mask_path ../training/groundtruth --batch_size 4 --epochs 500 --loss dice --name xception_gmap_nopt_finetune --aug 1 --model xception --pretr 0 --load_path best_models_final/gmap/xception_nopt


