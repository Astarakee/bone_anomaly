import cv2  # image processing and machine vision package
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from libs import self_inpaint
import os, json
import argparse
import skimage
import skimage.morphology
import skimage.filters
from libs.unet_model import InpaintingUnet
from libs.mialabutils import level_set_segmentation


import tensorflow as tf


def save_normailzed_color_image(file_name, image_array):
    image_array = image_array * 255
    if len(image_array.shape) == 3:
        output_array = np.flip(image_array, axis=-1)
    else:
        output_array = np.zeros((image_array.shape[0],image_array.shape[1],3))
        output_array[..., 0] = image_array
        output_array[..., 2] = -image_array
    cv2.imwrite(file_name, output_array)


def save_normailzed_grayscale_image(file_name, image_array):
    image_array = image_array * 255
    if len(image_array.shape) == 3:
        image_array = np.flip(image_array, axis=-1)
    cv2.imwrite(file_name, image_array)


def load_config(filename):
    with open(filename) as config_data:
        config = json.loads(config_data.read())
    return config


def binarize_label(inputmask, threshold_value=0.5):
    thresholded = np.where(inputmask > threshold_value, np.uint8(1), np.uint8(0))
    return thresholded


# parser = argparse.ArgumentParser(description='Test Self-inpaint')
# parser.add_argument('indir', type=str, help='Input dir for images')
# parser.add_argument('outdir', type=str, help='Output dir for image')
# parser.add_argument('config', type=str, help='Config file')
# parser.add_argument('-gpu', type=str, default="0", help="define the gpu_id to use")
# args = parser.parse_args()


# input_path = args.indir
# output_path = args.outdir
# config_file = args.config
# config = load_config(config_file)


# os.system("pwd")
# allfiles = os.listdir(input_path)
# allfiles.sort()

# #Mehdi
# checkpoint_dir = config["ModelFile"]
# latest = tf.train.latest_checkpoint(checkpoint_dir)
# model = InpaintingUnet(conv_layer='gconv', load_weights=latest,  train_bn=False)

DILATION_SIZE = 3


def create_mast_from_kldist(kl_dist, threshold=30, erode=False):
    if len(kl_dist.shape) > 2:
        kl_dist = np.average(kl_dist, axis=-1)
    kl_dist = np.expand_dims(kl_dist, axis=-1)
    mask = binarize_label(kl_dist, threshold)
    if erode is True:
        erode_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask[...,0], erode_kernel, iterations=1)
        mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    dialate_kernel = np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8)
    dialate_kernel[0, 0] = dialate_kernel[0, DILATION_SIZE - 1] = dialate_kernel[DILATION_SIZE - 1, 0] = dialate_kernel[
        DILATION_SIZE - 1, DILATION_SIZE - 1] = 0
    diliated = cv2.dilate(mask, dialate_kernel, iterations=1)
    mask = 1 - diliated
    return mask


def create_mast_from_kldist_with_hysteresis_threshold(kl_dist, t_low, t_high):
    if len(kl_dist.shape) > 2:
        kl_dist = np.average(kl_dist, axis=-1)

    mask = skimage.filters.apply_hysteresis_threshold(kl_dist, t_low, t_high)
    mask = skimage.morphology.binary_opening(mask, selem=skimage.morphology.disk(3))
    mask = skimage.morphology.binary_closing(mask, selem=skimage.morphology.disk(3))
    mask = mask.astype('uint8')
    dialate_kernel = np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8)
    dialate_kernel[0, 0] = dialate_kernel[0, DILATION_SIZE - 1] = dialate_kernel[DILATION_SIZE - 1, 0] = dialate_kernel[
        DILATION_SIZE - 1, DILATION_SIZE - 1] = 0
    diliated = cv2.dilate(mask, dialate_kernel, iterations=1)
    mask = np.expand_dims(diliated, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    mask = 1 - mask
    return mask


def create_mask_by_erosion(mask, erode_size=3):
    mask = 1 - mask
    if len(mask.shape) == 3:
        mask = mask[...,0]
    erode_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, erode_kernel, iterations=1)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    mask = 1 - mask
    return mask



def create_mast_from_paired_kldist(kl_dist, kl_dist_attempt, threshold=30, erode=False):
    if len(kl_dist.shape) > 2:
        kl_dist = np.average(kl_dist, axis=-1)
    kl_dist = np.expand_dims(kl_dist, axis=-1)
    if len(kl_dist_attempt.shape) > 2:
        kl_dist_attempt = np.average(kl_dist_attempt, axis=-1)
    kl_dist_attempt = np.expand_dims(kl_dist_attempt, axis=-1)
    mask = binarize_label(kl_dist, threshold)
    mask[kl_dist_attempt > threshold] = 1
    # mask = np.where((kl_dist+kl_dist_attempt>threshold), np.uint8(1), np.uint8(0))
    if erode is True:
        erode_kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, erode_kernel, iterations=1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    dialate_kernel = np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8)
    dialate_kernel[0, 0] = dialate_kernel[0, DILATION_SIZE - 1] = dialate_kernel[DILATION_SIZE - 1, 0] = dialate_kernel[
        DILATION_SIZE - 1, DILATION_SIZE - 1] = 0
    diliated = cv2.dilate(mask, dialate_kernel, iterations=1)
    mask = 1 - diliated
    return mask


def create_mast_from_level_set_seg(speed_map, init_mask, threshold=30, threshold_window = None, curvature_weight=0.1):
    if threshold_window is None:
        threshold_window = threshold*2
    min_clip = threshold - threshold_window/2.0
    max_clip = threshold + threshold_window/2.0
    speed_map[speed_map > max_clip] = max_clip
    speed_map[speed_map<min_clip] = min_clip
    speed_map = (speed_map-threshold)/(threshold_window/2.0)

    init_mask = 1 - init_mask
    if len(init_mask.shape) > 2:
        init_mask = init_mask[...,0]
    mask = level_set_segmentation(speed_map, init_mask, lower_threshold=0.0, upper_threshold=2.0,
                                  curvature_weight=curvature_weight)
    mask = binarize_label(mask,0)
    #plt.imshow(speed_map, cmap='rainbow')
    #plt.show()
    #plt.imshow(mask, cmap='gray')
    #plt.show()
    threshold_result = binarize_label(speed_map,0)
    #plt.imshow(threshold_result, cmap='gray')
    #plt.show()
    dialate_kernel = np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8)
    dialate_kernel[0, 0] = dialate_kernel[0, DILATION_SIZE - 1] = dialate_kernel[DILATION_SIZE - 1, 0] = dialate_kernel[
        DILATION_SIZE - 1, DILATION_SIZE - 1] = 0
    diliated = cv2.dilate(mask, dialate_kernel, iterations=1)
    diliated = skimage.morphology.binary_closing(diliated, selem=skimage.morphology.disk(3))
    mask = np.expand_dims(diliated, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    mask = 1 - mask
    return mask
