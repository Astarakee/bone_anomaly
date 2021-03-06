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

To build the docker image, run the following command: 

```
docker build . -t bone-anomaly
```

It will install the required packages and libraries, create the required folders and
download the weights of the trained model to be used for the inference phase.

## Usage:

After the installaiton, to run the code for the inference phase:

```
docker run -v INPUT_FOLDER:/input -v OUTPUT_FOLDER:/data bone-anomaly
```

Here "INPUT_FOLDER" and "OUTPUT_FOLDER" should be modified. In specific, the  "INPUT_FOLDER"
must be a directory containing the nifti files. 
The structure of "INPUT_FOLDER" should be like:

```
/data/INPUT_FOLDER/
        Subject1.nii.gz
        Subject2.nii.gz
        ....
        SubjectN.nii.gz
```
Please note that there is no restriction for the filename of each subjects.

The "OUTPUT_FOLDER" should be an empty folder to store the model outputs.

For example, it can be:

```
docker run -v /home/data/pelvis:/input -v /home/data/pelvis_results:/data bone-anomaly
```

The descrived script by default runs the "sliding window-based autoinpainting model".
To run the "KL-divergance-based autoinpainting model" you just need to add an argument:

```
docker run -v /home/data/pelvis:/input -v /home/data/pelvis_results:/data bone-anomaly --kl_div
```
Please note the computational time of the second approach is much more than first approach.


## Model output:

Within the "OUTPUT_FOLDER" two main directory will be created:

1 - data_2d: This will consists of the prepared 2D images to be analyzed. \
2 - results: This includes the inpainted images as well as the heating maps for each subjects separately. \
Please note within the "results" folder, one .json file and one .csv file will be stored. \
The .json file saves a summary of the model params and the .csv files represent the predicted
class label of processing algorithm. In particular, for each subject it indicated a binary
value [0, 1] which stands for the absence(0) or presence(1) of prothesis.


## Finally, the model checkpoints can be downloaded from the following link:
(https://drive.google.com/file/d/1IyI7uthpWAHgzDM3R3r99-X6UqOS6Jlr/view?usp=sharing)
