## Scripts to run all the major experiments

- The above ```.sh``` files are used to run the individual experiments. Each ```.sh``` file has the hyperparameters used for running the specific experiment. 

### Scripts for Training on Kaggle Data Only (Effect of Pretrained Weights for Initialization)
1. ```run_xception_ethz.sh``` - This script trains the X-UNet on Kaggle Data with pretrained initialization for the encoder. 
2. ```run_xception_ethz_nopt.sh``` - This script trains the X-UNet on Kaggle Data with random initialization for the encoder. 
3. ```run_resnet_ethz.sh``` - This script trains the ResUNet on Kaggle Data with pretrained initialization for the encoder. 
4. ```run_resnet_ethz_nopt.sh``` - This script trains the ResUNet on Kaggle Data with random initialization for the encoder. 


---
### Scripts for Finetuning Experiments

1. ```run_xception_gmap_nopt.sh``` - This script trains the X-UNet on Google Data with random initialization for the encoder. 
2. ```run_xception_gmap.sh``` - This script trains the X-UNet on Google Data with pretrained initialization for the encoder.
3. ```run_resnet_gmap.sh``` - This script trains the ResUNet on Google Data with pretrained initialization for the encoder.
4. ```run_resnet_gmap_nopt.sh``` - This script trains the ResUNet on Google Data with random initialization for the encoder.
5. ```ft_xception_gmap_nopt.sh``` - This script finetunes the X-UNet on Kaggle Data. The model loaded for finetuning was trained on Google Data while being initialized randomly (See 1.).
6. ```ft_xception_gmap_pt.sh``` - This script finetunes the X-UNet on Kaggle Data. The model loaded for finetuning was trained on Google Data while being initialized to pretrained weights (See 2.).
7. ```ft_resnet_gmap_pt.sh``` - This script finetunes the ResUNet on Kaggle Data. The model loaded for finetuning was trained on Google Data with encoder being initialized to pretrained weights (See 3.).
8. ```ft_resnet_gmap_nopt.sh``` - This script finetunes the ResUNet on Kaggle Data. The model loaded for finetuning was trained on Google Data with encoder being initialized randomly (See 4.).

The models loaded for these experiments are available [here](https://drive.google.com/drive/u/0/folders/1vZl5hVb_daQ6rG35uUQby0mRTWAl-640).

