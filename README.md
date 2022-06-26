# Anomaly detection of pelvis bone prothesis by AutoInpainting 

## set-up

Clone the git project:

```
$ git clone https://github.com/FeliMe/brain_sas_baseline.git
```

Create a virtual environment and install the requirements:

```
$ conda create -f environment.yml
```

Activate the newly created environment:

```
$ conda activate pelvis_unsupervised_anomaly
```

## Usage:

Change the following items in the config file "./config/config.json"

```
"nifti_dir": "PathToNiftiFiles/" : This should be the directory to the Nifti volumes.
"checkpoint_dir": "PathToModelCheckPoints": This should be the directory to the model weights.
"exp_name": "pelvis_anomaly_field_map": A name for the experiments.
```

The following params defines where the image data, logs and result summary to be stored:

```
"write_path_img": "./data/data_2d/image/" :  Dir to store the 2D preprocessed slices.
"write_path_map": "./data/data_2d/map/"   :  Dir to store the 2D preprocessed binary maps.
"write_path_results": "./data/results/"   :  Dir to store the final results. This will include the inpainted images, logs of details, and the final .csv file to present the class labels of predictions.
```

The following params are optimized and may not be changed:

```
"n_slice": "70" : The slices to be analyzed.
"step_size": 1  : The step size between the slices.
"res_thr": 400  : Threshold value.
```

Finally to run the code in the terminal, do:

```
In the main directory, within terminal, type:
python main.py -c ./config/config.json

```

## The model checkpoints can be downloaded from the following link:
(https://drive.google.com/file/d/1IyI7uthpWAHgzDM3R3r99-X6UqOS6Jlr/view?usp=sharing)
