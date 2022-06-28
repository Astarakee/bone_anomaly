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


parser = argparse.ArgumentParser(description='Test Self-inpaint')
parser.add_argument('indir', type=str, help='Input dir for images')
parser.add_argument('outdir', type=str, help='Output dir for image')
parser.add_argument('config', type=str, help='Config file')
parser.add_argument('-gpu', type=str, default="0", help="define the gpu_id to use")
args = parser.parse_args()


input_path = args.indir
output_path = args.outdir
config_file = args.config
config = load_config(config_file)


os.system("pwd")
allfiles = os.listdir(input_path)
allfiles.sort()

#Mehdi
checkpoint_dir = config["ModelFile"]
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = InpaintingUnet(conv_layer='gconv', load_weights=latest,  train_bn=False)

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
    plt.imshow(speed_map, cmap='rainbow')
    plt.show()
    plt.imshow(mask, cmap='gray')
    plt.show()
    threshold_result = binarize_label(speed_map,0)
    plt.imshow(threshold_result, cmap='gray')
    plt.show()
    dialate_kernel = np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8)
    dialate_kernel[0, 0] = dialate_kernel[0, DILATION_SIZE - 1] = dialate_kernel[DILATION_SIZE - 1, 0] = dialate_kernel[
        DILATION_SIZE - 1, DILATION_SIZE - 1] = 0
    diliated = cv2.dilate(mask, dialate_kernel, iterations=1)
    diliated = skimage.morphology.binary_closing(diliated, selem=skimage.morphology.disk(3))
    mask = np.expand_dims(diliated, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    mask = 1 - mask
    return mask


grid_size = config["GridSize"]
overlap_factor = config["OverlapFactor"]
mask_threshold = config["MaskThreshold"]
refind_mask_threshold = config["RefinedMaskThreshold"]
min_std = config["MinimumSTD"]
max_std = config["MaximumSTD"]
num_iteration = config["NumberOfIteration"]
batch_size = config["BatchSize"]
mix_method = config["ChannelMixMethod"]
convert_to_hsv = config["ConvertToHSV"]
suffix = config["FileSuffix"]
relax_factor = 1.0 / num_iteration

for file_name in allfiles:
    input_file = os.path.join(input_path, file_name)
    output_file = config["TestName"] + '-' + file_name
    output_file = os.path.join(output_path, output_file)
    input_image_BGR = cv2.imread(input_file, cv2.IMREAD_COLOR)
    input_image = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2RGB)
    print("Opened file:", input_file)
    image_org = self_inpaint.pre_process_image(input_image)
    image_org_hsv = None
    if convert_to_hsv is True:
        image_org_hsv = cv2.cvtColor(input_image_BGR, cv2.COLOR_BGR2HSV)
        image_org_hsv = self_inpaint.pre_process_image(image_org_hsv)
        print(np.min(image_org_hsv), np.max(image_org_hsv))

    fixed_image = image_org
    cur_mask = np.ones(fixed_image.shape,dtype = np.uint8)
    for iteration in range(0, 3):
        posterior_mean, posterior_std = self_inpaint.create_uncertainty_map(deepcopy(fixed_image),
                                                                            grid_size,
                                                                            overlap_factor,
                                                                            batch_size=batch_size,
                                                                            model=model,
                                                                            mix_method=mix_method,
                                                                            input_hsv=image_org_hsv)
        if iteration == 0:
            merged_posterior_mean, merged_posterior_std = posterior_mean, posterior_std
            mean_std = np.mean(posterior_std)
            median_std = np.median(posterior_std)
            print("std", mean_std, median_std)

        prior_std = np.ones(posterior_mean.shape) * median_std
        prior_mean = np.zeros(posterior_mean.shape)

        kl_dist = self_inpaint.kullback_leible_distance_gaussian(posterior_mean, posterior_std, prior_mean, prior_std)
        print("mean diff", np.mean(posterior_mean), "mean std", np.mean(posterior_std))
        mean_normailization_factor = 1.0 / (np.std(posterior_mean) * 6)
        std_normailization_factor = 1.0 / (np.std(posterior_std) * 6)
        # save_normailzed_color_image(output_file.replace(suffix, "_0_avg_{}".format(iteration) + suffix), posterior_mean * mean_normailization_factor)
        # save_normailzed_grayscale_image(output_file.replace(suffix, "_0_std_{}".format(iteration) + suffix), posterior_std * std_normailization_factor)
        print(output_file)
        print("kl distance:", np.max(kl_dist), np.mean(kl_dist), np.std(kl_dist))
        # save_normailzed_grayscale_image(output_file.replace(suffix, "_0_vkl_{}".format(iteration) + suffix), kl_dist / (12))
        new_mask = create_mast_from_kldist(kl_dist, mask_threshold)
        if iteration > 0:
            added_region = np.where((new_mask==0) & (cur_mask==1), np.uint8(1), np.uint8(0))
            merged_posterior_mean = merged_posterior_mean * (1.0 - added_region[...,0]) + posterior_mean * added_region[...,0]
            merged_posterior_std = merged_posterior_std * (1.0 - added_region[...,0]) + posterior_std * added_region[...,0]
        cur_mask = np.minimum(new_mask, cur_mask)
        fixed_image = self_inpaint.inpaint_with_mask(deepcopy(image_org), cur_mask, model=model)
        # save_normailzed_color_image(output_file.replace(suffix, "_0_xmk_{}".format(iteration) + suffix), cur_mask)
        # save_normailzed_color_image(output_file.replace(suffix, "_0_xed_{}".format(iteration) + suffix), fixed_image)

    # kl_dist = self_inpaint.kullback_leible_distance_gaussian(merged_posterior_mean, merged_posterior_std, prior_mean, prior_std)
    # new_mask = create_mast_from_kldist_with_hysteresis_threshold(kl_dist, mask_threshold*0.75, mask_threshold)
    # fixed_image = self_inpaint.inpaint_with_mask(deepcopy(image_org), new_mask, model=model)
    # cur_mask = skimage.morphology.binary_opening(cur_mask[...,0], selem=skimage.morphology.disk(3))

    prior_mean, prior_std = self_inpaint.create_uncertainty_map(deepcopy(fixed_image),
                                                                grid_size,
                                                                overlap_factor,
                                                                batch_size=batch_size,
                                                                model=model,
                                                                mix_method=mix_method,
                                                                input_hsv=image_org_hsv)

    posterior_mean, posterior_std = merged_posterior_mean, merged_posterior_std

    save_normailzed_color_image(output_file.replace(suffix, "_0_avg_max" + suffix),
                                posterior_mean * mean_normailization_factor)
    save_normailzed_grayscale_image(output_file.replace(suffix, "_0_std_max" + suffix),
                                    posterior_std * std_normailization_factor)
    save_normailzed_grayscale_image(output_file.replace(suffix, "_0_vkl_max" + suffix),
                                    kl_dist / 12)
    save_normailzed_color_image(output_file.replace(suffix, "_0_xmk" + suffix), cur_mask)
    save_normailzed_color_image(output_file.replace(suffix, "_0_xed" + suffix), fixed_image)
    entropy_gain = self_inpaint.entropy_gain_gaussian(posterior_std,prior_std)
    entropy_normailization_factor = 1.0 / (np.std(entropy_gain) * 6)
    print("entropy normal factor", entropy_normailization_factor, np.mean(entropy_gain), np.std(entropy_gain))
    save_normailzed_color_image(output_file.replace(suffix, "_ent" + suffix),
                                entropy_gain * entropy_normailization_factor)
    guarantee_fine_region = 1 - binarize_label(posterior_std, threshold_value=mean_std/2.0)
    if len(guarantee_fine_region.shape)==2:
        guarantee_fine_region = np.expand_dims(guarantee_fine_region, axis=-1)
        guarantee_fine_region = np.concatenate((guarantee_fine_region, guarantee_fine_region, guarantee_fine_region), axis=-1)

    save_normailzed_color_image(output_file.replace(suffix, "_guaranteed_fine_region" + suffix), guarantee_fine_region)

    visual_mask = deepcopy(cur_mask)

    for iteration in range(1, num_iteration):
        if iteration > 1:
            prior_mean, prior_std = self_inpaint.create_uncertainty_map(deepcopy(fixed_image),
                                                                        grid_size,
                                                                        overlap_factor,
                                                                        batch_size=batch_size,
                                                                        model=model,
                                                                        mix_method=mix_method,
                                                                        input_hsv=image_org_hsv)

        save_normailzed_color_image(output_file.replace(suffix, "_avg{}".format(iteration) + suffix),
                                    prior_mean * mean_normailization_factor)
        # std_ref = cv2.GaussianBlur(prior_std, (15, 15), 5.0)
        print("new mean diff", np.mean(prior_mean), "new mean std", np.mean(prior_std))

        save_normailzed_grayscale_image(output_file.replace(suffix, "_std{}".format(iteration) + suffix),
                                    prior_std * std_normailization_factor)
        kl_dist = self_inpaint.kullback_leible_distance_gaussian(posterior_mean, posterior_std, prior_mean, prior_std)
        # kl_dist = self_inpaint.kullback_leible_distance_histogram(posterior_pdf + 0.000001, prior_pdf + 0.000001)
        print("kl distance:", np.max(kl_dist), np.mean(kl_dist), np.std(kl_dist), np.median(kl_dist))
        save_normailzed_grayscale_image(output_file.replace(suffix, "_vkl{}".format(iteration) + suffix),
                                    kl_dist / (12.0))
        mask = create_mast_from_kldist(kl_dist, refind_mask_threshold)
        # mask = create_mask_by_erosion(cur_mask)
        # save_normailzed_grayscale_image(output_file.replace(suffix, "_xmk{}_attempt".format(iteration) + suffix), mask)
        visual_mask[...,1] = mask[...,0]

        fixed_image = self_inpaint.inpaint_with_mask(deepcopy(image_org), mask, model=model)
        attempt_mean, attempt_std = self_inpaint.create_uncertainty_map(deepcopy(fixed_image),
                                                                        grid_size,
                                                                        overlap_factor,
                                                                        batch_size=batch_size,
                                                                        model=model,
                                                                        mix_method=mix_method,
                                                                        input_hsv=image_org_hsv)
        save_normailzed_grayscale_image(output_file.replace(suffix, "_std{}_attempt".format(iteration) + suffix),
                                    attempt_std * std_normailization_factor)

        kl_dist_attempt = self_inpaint.kullback_leible_distance_gaussian(attempt_mean, attempt_std, prior_mean, prior_std)

        entropy_gain = self_inpaint.entropy_gain_gaussian(posterior_std, attempt_std)
        save_normailzed_color_image(output_file.replace(suffix, "_ent{}_attempt_against_posterior".format(iteration) + suffix),
                                    entropy_gain * entropy_normailization_factor)

        entropy_gain = self_inpaint.entropy_gain_gaussian(attempt_std, prior_std)
        save_normailzed_color_image(output_file.replace(suffix, "_ent{}_attempt".format(iteration) + suffix),
                                    entropy_gain * entropy_normailization_factor )

        save_normailzed_grayscale_image(output_file.replace(suffix, "_vkl{}_atempt".format(iteration) + suffix),
                                        (kl_dist_attempt)/ (12.0))

        # mask = create_mast_from_paired_kldist(kl_dist, kl_dist_attempt-entropy_gain*5, refind_mask_threshold)
        # mask = create_mast_from_kldist(kl_dist + kl_dist_attempt, refind_mask_threshold*2.0)
        mask = create_mast_from_level_set_seg(kl_dist+kl_dist_attempt-entropy_gain*5, cur_mask,
                                              threshold=refind_mask_threshold)
        mask = np.maximum(mask, guarantee_fine_region)


        # kl_dist_attempt = self_inpaint.kullback_leible_distance_gaussian(attempt_mean, attempt_std, posterior_mean, posterior_std)
        save_normailzed_color_image(output_file.replace(suffix, "_vkl{}_gain".format(iteration) + suffix),
                                    (kl_dist_attempt - kl_dist) / (12.0))

        save_normailzed_grayscale_image(output_file.replace(suffix, "_vkl{}_speed".format(iteration) + suffix),
                                    (kl_dist+kl_dist_attempt-entropy_gain*5) / 12.0)

        fixed_image = self_inpaint.inpaint_with_mask(deepcopy(image_org), mask, model=model)




        # save_normailzed_grayscale_image(output_file.replace(suffix, "_xmk{}".format(iteration) + suffix), mask)
        save_normailzed_color_image(output_file.replace(suffix, "_xed{}".format(iteration) + suffix), fixed_image)
        visual_mask[..., 2] = mask[..., 0]
        save_normailzed_color_image(output_file.replace(suffix, "_vkl{}_mask1".format(iteration) + suffix), visual_mask)
        save_normailzed_color_image(output_file.replace(suffix, "_vkl{}_mask2".format(iteration) + suffix), mask)
        visual_mask = deepcopy(mask)
        cur_mask = mask
