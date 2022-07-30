# CIL 2022 Road Segementation Project 
Our codebase has the following different components
- Generation of Additional Data From Google Maps API
- Training Different Models (Baselines, UNet, UNet with different Encoders)
- Evaluation
  - Standard Evaluation without Augmentation
  - Evaluation with Augmentation
- Ensembling the best models

---
The detailed instructions for each of the components is below. 

But before that we would like to setup the packages needed for utilizing the codebase. 

We use ```Python 3.8.5``` for all our experiments. 

The list of python packages that our codebase carries can be found in the ```requirements.txt``` file in the root directory above.


If you are using a local machine, kindly execute the following commands

```
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```



