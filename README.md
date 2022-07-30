# CIL 2022 Road Segementation Project 
Our codebase has the following different components
- Generation of Additional Data From Google Maps API
- Training Different Models (Baselines, UNet, UNet with different Encoders)
- Finetuning a trained model on a different dataset
- Evaluation
  - Standard Evaluation without Augmentation
  - Evaluation with Augmentation
- Ensembling the best models

---
The detailed instructions for each of the components is below. 

But before that we would like to setup the packages needed for utilizing the codebase. 

We use ```Python 3.8.5``` for all our experiments. 

The list of python packages that our codebase carries can be found in the ```requirements.txt``` file in the root directory above.


If you are using a local machine, execute the following commands

```
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

If you are instead using the Euler Cluster, execute the following commands

```
env2lmod
module load gcc/8.2.0 python_gpu/3.8.5
pip3 install albumentations==1.1.0
```
---
### Generation of Additional Data
- Firstly create appropriate directories for images and directories respectively.
- Then set the values for ```sat_image_path``` and ```road_image_path``` in ```Line 51``` and ```Line 52``` of the ```data_gen_gmap.py``` script with the directories you created.  
- Now replace the ```API_KEY = "Put your API Key here"``` in ```Line 8``` of the ```data_gen_gmap.py``` script with your Google Maps API Key.
- Finally execute the command```python data_gen_gmap.py``` script to generate the additional data.
---
### Training Different Models
- To train a model, you need to execute the ```train.py``` script inside the ```src``` directory. 
- The script requires you to provide the following key command line arguments:
  - ```seed``` to set the random seed for reproducibility. Example Value: ```0```
  - ```valid``` to set the fraction of training data to use as validation data. Example Value: ```0.2```
  - ```train_image_path``` - to set the path to the train image. Example Value: ```./gmap_data/images```
  - ```train_mask_path``` - to set the path to the train masks. Example Value: ```./gmap_data/groudtruth```
  - ```batch_size``` - to set the batch size for training. Example Value: ```4```
  - ```epochs``` - to set the number of epochs to train the model. Example Value: ```200```
  - ```lr``` - to set the initial learning rate for training. Example Value: ```1e-4```
  - ```name``` - to set the name for the experiment. Example Value: ```resnet_pretrained```
  - ```model``` - to set the model to train. Example Value: ```resnet```
  - ```pretr``` - flag to decide whether to use pre-trained weights as initialization. Example Value: ```1``` for using pre-trained weights.
  - ```wdecay``` - to include l2 regularization while training the models. Example Value: ```1e-5```

An example command to execute would be the following: 
```
python train.py --seed 0 --valid 0.2 --train_image_path ./images --train_mask_path ./groundtruth --batch_size 4 --epochs 200 --lr 1e-4 --name xception_pretrained --model xception --pretr 1 --wdecay 0
```
For the ease of usage, we provide all the scripts we used for training our models in the ```scripts``` directory. All the hyperparameters are specified in the scripts themselves. 

The ```train.py``` script creates a directory (see ```Lines 23-28``` in ```train.py```) where it logs outputs, stores model hyperparameters and the models. The name to this directory is created from the command line arguments. Therefore uniqueness in commandline arguments such as ```name``` is suggested for better management of experiments. The script also performs early stopping whenever the ```valid``` commandline argument is not zero. The models are also saved every ```10``` epochs. This is stored as ```model_current_{epoch}``` where epoch is the current epoch. The models are also stored as and when the performance on validation set improves. They are stored as ```best_model_{epoch}``` where epoch is the epoch when the validation loss improved. The model that has the overall best validation score is the one stored the last with the largest epoch number.

---
### Finetuning Different Models
- To finetune a model, execute the script ```finetune.py``` inside the ```src``` directory. 
- For this, make sure to provide the arguments ```load_path``` and ```model``` in addition to the others discussed above. 
- The ```finetune.py``` script also creates a directory where it logs outputs, stores model hyperparameters and the models.
- ```load_path``` refers to the path of the model to be loaded. Example Value: ```./exps/resnet_best/model_best_200```.
- Just like the ```train.py``` script, the finetuned models are saved every ```10``` epochs, as ```model_ft_current_{epoch}``` where epoch is current epoch and the best models based on validation set are saved as ```best_ft_model_{epoch}``` where epoch is the epoch where validation loss improved.

An example command to finetune on Kaggle Data after training on Google Data would be the following:
```
python finetune.py --train_image_path ../kaggle/images --train_mask_path ../kaggle/groundtruth --batch_size 4 --epochs 500 --loss dice --name resnet_gmap_nopt_finetune --model resnet --pretr 0 --load_path best_models_final/gmap/resnet_nopt --seed 1
```
---
### Evaluation
