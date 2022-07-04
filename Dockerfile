FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update --allow-unauthenticated && apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

## COPY main.py scripts and libs
RUN mkdir /input
RUN mkdir /model_weights
RUN mkdir libs
COPY libs ./libs
COPY main.py main.py

## DOWNLOAD Pre-trained weights
RUN gdown https://drive.google.com/uc?id=1IyI7uthpWAHgzDM3R3r99-X6UqOS6Jlr
RUN unzip model_weight.zip
ENTRYPOINT ["python","main.py"]
