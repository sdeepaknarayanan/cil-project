# Instructions for Baselines

To train and evaluate baseline models, run the scripts:
- ```logistic_regression_baseline.py``` for the logistic regression baseline
- ```cnn_baseline.py``` for the CNN baseline

The scripts require you to provide the following key command line arguments:
  - ```train_image_path``` - to set the path to the train images. Example Value: ```../kaggle_data/training/images```
  - ```train_mask_path``` - to set the path to the train masks. Example Value: ```../kaggle_data/training/groudtruth```
  - ```test_image_path``` - to set the path to the test images. Example Value: ```../kaggle_data/test/images```
  - ```name``` - to set the name for the experiment. Example Value: ```cnn```
  
Script ```cnn_baseline.py``` has these additional arguments:
  - ```seed``` to set the random seed for reproducibility. Example Value: ```0```
  - ```batch_size``` - to set the batch size for training. Example Value: ```128```
  - ```epochs``` - to set the number of epochs to train the model. Example Value: ```20```
  - ```lr``` - to set the initial learning rate for training. Example Value: ```1e-3```
  
  Commands to reproduce the baseline results from our report are:
  ```
  python logistic_regression_baseline.py --train_image_path ../kaggle_data/training/images --train_mask_path ../kaggle_data/training/groundtruth --test_image_path ../kaggle_data/test/images --name logreg

  python cnn_baseline.py --train_image_path ../kaggle_data/training/images --train_mask_path ../kaggle_data/training/groundtruth --test_image_path ../kaggle_data/test/images --name cnn --seed 0 --batch_size 128 --epochs 20 --lr 1e-3
  ```
  