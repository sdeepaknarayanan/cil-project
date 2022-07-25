python finetune.py --train_image_path ../gmap_data/images --train_mask_path ../gmap_data/groundtruth --batch_size 4 --epochs 200 --loss dice --name resunet_nopt_loaded --aug 1 --model resunet --load_path resunet_nopt_model/best_model_197

