# Anomaly detection of pelvis bone prothesis by AutoInpainting 

This repo contanins two different methods to detect the artifical prothesis in the pelvis region.
While both methods are built upon inpainting strategy, they are fundamentally different from each other.
Both methods can be executed with a single command that is explained in the following.

## set-up

Clone the git project:

```
$ https://github.com/Astarakee/bone_anomaly.git
```

Change the directory into the cloned folder:

```
cd bone_anomaly
```
The repo contains three branches. The one that is Dockerized
and contains both models are named as "kl_docker".
Therefore, swith to the branch by typing:

```
git checkout kl_docker
```


## Usage:

In the main directory type

```
python main.py -h
```
This will return the list of arguments that are required by the user.
Mandatory input is:

```
"--input_dir" : the directory to the input nifti volumes that are going to be processed and analyzed.
```

There exist also some other optional inputs that may not be changed. The current values are the optial ones.

```
"n_slice": "70" : The slices to be analyzed.
"step_size": 1  : The step size between the slices.
"res_thr": 400  : Threshold value.
```


The following optional params defines where the image data, logs and result summary to be stored:

```
"write_path_img": "./data/data_2d/image/" :  Dir to store the 2D preprocessed slices.
"write_path_map": "./data/data_2d/map/"   :  Dir to store the 2D preprocessed binary maps.
"write_path_results": "./data/results/"   :  Dir to store the final results. This will include the inpainted images, logs of details, and the final .csv file to present the class labels of predictions.
```


Finally to run the code in the terminal, do:

```
In the main directory, within terminal, type:
python main.py --input_dir 'dir to nifti data'

```

## The model checkpoints can be downloaded from the following link:
(https://drive.google.com/file/d/1IyI7uthpWAHgzDM3R3r99-X6UqOS6Jlr/view?usp=sharing)
