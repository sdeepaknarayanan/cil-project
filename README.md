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
  - ```seed``` to set the random seed for reproducibility. Example Value: ```0```.
  - ```valid``` to set the fraction of training data to use as validation data. Example Value: ```0.2```.
  - ```train_image_path``` - to set the path to the train image. Example Value: ```./gmap_data/images```
  - ```train_mask_path``` - to set the path to the train masks. Example Value: ```./gmap_data/groudtruth```
  - ```batch_size``` - to set the batch size for training. Example Value: ```4```
  - ```epochs``` - to set the number of epochs to train the model. Example Value: ```200```
  - ```lr``` - to set the initial learning rate for training. Example Value: ```1e-4```
  - ```name``` - to set the name for the experiment. Example Value: ```resnet_pretrained_finetune```
  - ```model``` - to set the model to train. Example Value: ```resnet```
  - ```pretr``` - flag to decide whether to use pre-trained weights as initialization. Example Value: ```1``` for using pre-trained weights.
  - ```wdecay``` - to include l2 regularization while training the models. Example Value: ```1e-5```
