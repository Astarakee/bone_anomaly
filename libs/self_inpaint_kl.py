import cv2 # image processing and machine vision package
from copy import deepcopy
import numpy as np
from libs.unet_model import InpaintingUnet
from libs.util import GridMaskGenerator
import scipy.stats

def histogram_along_axis(data, n_bins=32, range_limits=(-0.32,0.32)):
    # Setup bins and determine the bin location for each element for the bins
    R = range_limits
    N = data.shape[-1]
    bins = np.linspace(R[0],R[1],n_bins+1)
    data2D = data.reshape(-1,N)
    idx = np.searchsorted(bins, data2D,'right')-1

    # Some elements would be off limits, so get a mask for those
    idx[idx==-1] = 0
    idx[idx==n_bins] = n_bins-1

    # We need to use bincount to get bin based counts. To have unique IDs for
    # each row and not get confused by the ones from other rows, we need to
    # offset each row by a scale (using row length for this).
    scaled_idx = n_bins*np.arange(data2D.shape[0])[:,None] + idx

    limit = n_bins*data2D.shape[0]

    # Get the counts and reshape to multi-dim
    counts = np.bincount(scaled_idx.ravel(),minlength=limit+1)[:-1]
    counts.shape = data.shape[:-1] + (n_bins,)
    return counts.astype(np.float)/N


def create_tiled_image_mask_series(input_image, grid_size=64, step_length=16):
    image_list, paint_mask_list, stitch_mask_list = [], [], []
    assert (len(input_image.shape) == 4)
    mask_generator = GridMaskGenerator(input_image.shape[1],
                                       input_image.shape[2],
                                       grid_size=grid_size,
                                       margin=2)
    n_step_y = n_step_x = int(grid_size * 2 / step_length)
    for y in range(n_step_y):
        for x in range(n_step_x):
            offset = (int(step_length * x), int(step_length * y))
            mask_generator.set_grid_offset(offset)
            mask_paint, mask_stitch = mask_generator.sample()
            mask_paint = np.expand_dims(mask_paint, 0)
            paint_mask_list.append(mask_paint)
            mask_stitch = np.expand_dims(mask_stitch, 0)
            stitch_mask_list.append(mask_stitch)
            image = deepcopy(input_image)
            image[mask_paint == 0] = 1
            image_list.append(image)

            # plt.imshow(image[0, ...])
            # plt.show()
    image_array = np.concatenate(tuple(image_list), axis=0)
    paint_mask_array = np.concatenate(tuple(paint_mask_list), axis=0)
    stitch_mask_array = np.concatenate(tuple(stitch_mask_list), axis=0)
    return image_array, paint_mask_array, stitch_mask_array


def stitchTiledImageMaskSeries(image_array, mask_array, grid_size=64, step_length=16):
    n_step_y = n_step_x = int(grid_size * 2 / step_length)
    new_shape = image_array.shape
    n_stitched_image = int(n_step_x * n_step_y / 4)
    new_shape = (n_stitched_image, new_shape[1], new_shape[2], new_shape[3])
    stitched_images = np.zeros(new_shape)
    for y in range(n_step_y):
        for x in range(n_step_x):
            offset = (int(step_length * x), int(step_length * y))
            stitched_id = (offset[0] % grid_size) / step_length + n_step_x/2*(offset[1] % grid_size) / step_length
            stitched_id = int(stitched_id)
            tiled_id = x + y*n_step_x
            tile = image_array[tiled_id]
            tile[mask_array[tiled_id] == 1] = 0
            stitched_images[stitched_id] += tile

    return stitched_images

def kullback_leible_distance_gaussian(mean_1, std_1, mean_2, std_2):
    #print(np.mean(mean_1), np.mean(mean_2), np.mean(std_1), np.mean(std_2))
    kl_dist = np.square(mean_1 - mean_2) + np.square(std_1) - np.square(std_2)
    #print(np.mean(kl_dist))
    kl_dist /= 2 * np.square(std_2)
    #print(np.mean(kl_dist))
    kl_dist += np.log(std_2 / std_1)
    #print(np.mean(kl_dist))
    return kl_dist

def entropy_gain_gaussian(std_1, std_2):
    constance_variable = 2.0*np.pi*np.e
    entropy_1 = 0.5 * np.log(constance_variable * std_1 * std_1)
    entropy_2 = 0.5 * np.log(constance_variable * std_2 * std_2)
    return entropy_2-entropy_1

def kullback_leible_distance_histogram(input_data1, input_data2):
    kl_dist = scipy.stats.entropy(input_data1, qk=input_data2, axis=-1)
    return kl_dist


def pre_process_image(input_image):
    processed_image = cv2.resize(input_image,(256,256))
    processed_image = deepcopy(processed_image)
    processed_image =  processed_image/255.0
    return processed_image


def create_uncertainty_map(input_image, grid_size, over_sample_factor, batch_size=1,
                           model_file=None, model=None, mix_method='avg', input_hsv=None):
    if input_hsv is not None:
        input_image_rgb = input_image * 255.0
        input_hsv = cv2.cvtColor(input_image_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV) / 255.0
    if model is None:
        #model = PConvUnet(vgg_weights=None, inference_only=True)
        model = InpaintingUnet(conv_layer='gconv', load_weights=model_file,  train_bn=True) #Mehdi
        if model_file is None:
            raise Exception('Inpainting model and model files are missing!')
        #model.load(model_file, train_bn=False)
    input_image = np.expand_dims(input_image, 0)
    images, paint_masks, stitch_masks = create_tiled_image_mask_series(input_image, grid_size=grid_size, step_length=grid_size / over_sample_factor)
    pred_imgs = model.predict([images, paint_masks], batch_size=batch_size, verbose=0)
    stitched_images = stitchTiledImageMaskSeries(pred_imgs, stitch_masks, grid_size=grid_size,
                                                 step_length=grid_size / over_sample_factor)
    diff_slice_all = []
    for i in range(stitched_images.shape[0]):
        diff_slice = stitched_images[i] - input_image[0, ...]
        diff_slice_all.append(diff_slice)
    diff_slice_all = np.array(diff_slice_all)
    diff_mean = np.mean(diff_slice_all, axis=0)
    if mix_method == 'sqrt':
        diff_mean = np.square(diff_mean)
        diff_mean = np.sqrt(np.sum(diff_mean, axis=-1))
    elif mix_method == "avg":
        diff_mean = np.mean(diff_mean, axis=-1)

    diff_std = np.std(diff_slice_all, axis=0)
    if mix_method == 'sqrt':
        diff_std = np.square(diff_std)
        diff_std = np.sqrt(np.sum(diff_std, axis=-1))
    elif mix_method == "avg":
        diff_std = np.mean(diff_std, axis=-1)
    diff_std += np.ones(diff_std.shape) * 0.000001 #to avoid divide by zero

    return diff_mean, diff_std


def inpaint_with_mask(input_image, mask, batch_size=1, model_file=None, model=None):
    input_image = np.expand_dims(input_image, 0)
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    mask = np.expand_dims(mask, 0)
    input_image[mask == 0] = 1
    if model == None:
        #model = PConvUnet(vgg_weights=None, inference_only=True)
        model = InpaintingUnet(conv_layer='gconv', load_weights=model_file,  train_bn=True) #Mehdi
        # if model_file is None:
        #     model_file = r"/mnt/Data/chunliang/develop/pcnn/logs/imagenet_phase1_paperMasks/weights.09-0.16.h5"
        # model.load(model_file, train_bn=False)
    pred_imgs = model.predict([input_image, mask], batch_size=batch_size)
    return pred_imgs[0]
