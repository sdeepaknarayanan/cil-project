## Scripts to run all the major experiments

The above ```.sh``` files are used to run the individual experiments. Each ```.sh``` file has the hyperparameters used for running the specific experiment. The only modification required while executing these scripts is to change the path to the data. 

### Scripts for Training on Kaggle Data Only (Effect of Pretrained Weights for Initialization)
- ```run_unet_ethz.sh``` - This script trains the UNet on Kaggle Data.
- ```run_xception_ethz.sh``` - This script trains the X-UNet on Kaggle Data with pretrained initialization for the encoder.
- ```run_xception_ethz_nopt.sh``` - This script trains the X-UNet on Kaggle Data with random initialization for the encoder.
- ```run_resnet_ethz.sh``` - This script trains the ResUNet on Kaggle Data with pretrained initialization for the encoder.
- ```run_resnet_ethz_nopt.sh``` - This script trains the ResUNet on Kaggle Data with random initialization for the encoder. 


---
### Scripts for Finetuning Experiments
- ```run_unet_gmap.sh``` - This script trains the UNet on Google Data.
- ```run_xception_gmap_nopt.sh``` - This script trains the X-UNet on Google Data with random initialization for the encoder. 
- ```run_xception_gmap.sh``` - This script trains the X-UNet on Google Data with pretrained initialization for the encoder.
- ```run_resnet_gmap.sh``` - This script trains the ResUNet on Google Data with pretrained initialization for the encoder.
- ```run_resnet_gmap_nopt.sh``` - This script trains the ResUNet on Google Data with random initialization for the encoder.
- ```ft_unet_gmap.sh``` - This script finetunes the UNet on Kaggle Data. The model loaded for finetuning was trained on Google Data while being initialized randomly.
- ```ft_xception_gmap_nopt.sh``` - This script finetunes the X-UNet on Kaggle Data. The model loaded for finetuning was trained on Google Data while being initialized randomly.
- ```ft_xception_gmap_pt.sh``` - This script finetunes the X-UNet on Kaggle Data. The model loaded for finetuning was trained on Google Data while being initialized to pretrained weights.
- ```ft_resnet_gmap_pt.sh``` - This script finetunes the ResUNet on Kaggle Data. The model loaded for finetuning was trained on Google Data with encoder being initialized to pretrained weights.
- ```ft_resnet_gmap_nopt.sh``` - This script finetunes the ResUNet on Kaggle Data. The model loaded for finetuning was trained on Google Data with encoder being initialized randomly.

---

The models obtained from all the experiments and the ones used for finetuning are available [here](https://drive.google.com/drive/u/0/folders/1vZl5hVb_daQ6rG35uUQby0mRTWAl-640), under the same name as the scripts.

