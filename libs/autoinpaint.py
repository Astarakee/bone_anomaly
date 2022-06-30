import os
import cv2
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy
from libs import self_inpaint
import matplotlib.pyplot as plt
from libs.tools import get_sum_int
from scipy.ndimage import morphology
from libs.img_utils import load_img_array
from libs.unet_model import InpaintingUnet
from libs.anomaly_tools import get_organ_mask
from libs.batch_test_new import create_mast_from_kldist, binarize_label
from libs.paths_dirs_stuff import get_data_list, get_sub_dirs, creat_dir
from libs.batch_test_new import create_mast_from_level_set_seg, save_normailzed_color_image

# GPU memory setting
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)



DILATION_SIZE = 3
MaskThreshold = 16
mask_threshold = MaskThreshold



def auto_inpaint(write_path_results, exp_name, img_path, field_path, \
                 checkpoint_dir, grid_size, overlap_factor, batch_size, \
                     mix_method, image_org_hsv, num_iteration, refind_mask_threshold):

 
    # Set the exp name
    write_path = os.path.join(write_path_results, exp_name)
    write_path_h_map = os.path.join(write_path_results, exp_name+'_heatmap')
        
    # Get data and load model
    img_subjects = get_sub_dirs(img_path)[1:]
    img_field_maps = get_sub_dirs(field_path)[1:]   
    
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    model = InpaintingUnet(conv_layer='gconv', load_weights=checkpoint,  train_bn=True)
    
    
    
    ######## Parameters setting 
    
  
    image_size = 256
    n_subjects = len(img_subjects)
    elapse_time = []
    
    
    for ind in range(n_subjects):
        print('Anomaly detection of subject {} out of {}'.format(ind, n_subjects))
        subject_img_path = img_subjects[ind]
        subject_field_path = img_field_maps[ind]
      
        subject_name = subject_img_path.split('/')[-1]
        subject_write_dir = os.path.join(write_path, subject_name)
        subject_h_map_dir = os.path.join(write_path_h_map, subject_name)
        creat_dir(subject_write_dir)
        creat_dir(subject_h_map_dir)
        
        img_lists = get_data_list(subject_img_path, '.png')  
        field_map_lists = get_data_list(subject_field_path, '.png')  
    
        img_array = load_img_array(img_lists, image_size)
        field_array = load_img_array(field_map_lists, image_size)
        organ_mask_volume = get_organ_mask(img_array)
    
        
        slice_n_base = img_lists[0].split('.png')[0].split('_')[-1]
        slice_number = int(slice_n_base)-1
        #slice_number = int(slice_n_base)
        
    
        initTime=time.time()
        for ix in range(img_array.shape[0]):
            slice_number = slice_number+1
            main_img = img_array[ix]
            main_field_map = field_array[ix]
            organ_mask = organ_mask_volume[ix][:,:,0].astype('int')
            organ_mask = morphology.binary_fill_holes(organ_mask, structure=np.ones((3,3)))
            organ_mask = np.repeat(organ_mask[:, :, np.newaxis], 3, axis=2).astype('float32')
    
            slice_name = subject_name+'_inpaint_'+str(slice_number)+'.png'
            slice_name_heatmap = subject_name+'_0_avg_max'+str(slice_number)+'.png'
            path_to_heat_slice = os.path.join(subject_h_map_dir, slice_name_heatmap)

            field_sum_int = np.sum(main_field_map)
            #print(field_sum_int)
            if field_sum_int>50: # instead of 0 avoid tiny colon ...
            
                fixed_image = main_img
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
                       

                    prior_std = np.ones(posterior_mean.shape) * median_std
                    prior_mean = np.zeros(posterior_mean.shape)

                    kl_dist = self_inpaint.kullback_leible_distance_gaussian(posterior_mean, posterior_std, prior_mean, prior_std)
                 
                    mean_normailization_factor = 1.0 / (np.std(posterior_mean) * 6)
                    std_normailization_factor = 1.0 / (np.std(posterior_std) * 6)
                   
                  
                   
                    new_mask = create_mast_from_kldist(kl_dist, mask_threshold)
                    if iteration > 0:
                        added_region = np.where((new_mask==0) & (cur_mask==1), np.uint8(1), np.uint8(0))
                        merged_posterior_mean = merged_posterior_mean * (1.0 - added_region[...,0]) + posterior_mean * added_region[...,0]
                        merged_posterior_std = merged_posterior_std * (1.0 - added_region[...,0]) + posterior_std * added_region[...,0]
                    cur_mask = np.minimum(new_mask, cur_mask)
                    fixed_image = self_inpaint.inpaint_with_mask(deepcopy(main_img), cur_mask, model=model)

                prior_mean, prior_std = self_inpaint.create_uncertainty_map(deepcopy(fixed_image),
                                                                            grid_size,
                                                                            overlap_factor,
                                                                            batch_size=batch_size,
                                                                            model=model,
                                                                            mix_method=mix_method,
                                                                            input_hsv=image_org_hsv)

                posterior_mean, posterior_std = merged_posterior_mean, merged_posterior_std

                # save_normailzed_color_image(output_file.replace(path_to_heat_slice,
                #                             posterior_mean * mean_normailization_factor)
                
                entropy_gain = self_inpaint.entropy_gain_gaussian(posterior_std,prior_std)
                entropy_normailization_factor = 1.0 / (np.std(entropy_gain) * 6)
                
                guarantee_fine_region = 1 - binarize_label(posterior_std, threshold_value=mean_std/2.0)
                if len(guarantee_fine_region.shape)==2:
                    guarantee_fine_region = np.expand_dims(guarantee_fine_region, axis=-1)
                    guarantee_fine_region = np.concatenate((guarantee_fine_region, guarantee_fine_region, guarantee_fine_region), axis=-1)

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
                    if iteration==num_iteration-1:
                        save_normailzed_color_image(path_to_heat_slice,
                                                    prior_mean * mean_normailization_factor)
                   
                    kl_dist = self_inpaint.kullback_leible_distance_gaussian(posterior_mean, posterior_std, prior_mean, prior_std)
                    
                    # save_normailzed_grayscale_image(output_file.replace(suffix, "_vkl{}".format(iteration) + suffix),
                    #                             kl_dist / (12.0))
                    mask = create_mast_from_kldist(kl_dist, refind_mask_threshold)
                    visual_mask[...,1] = mask[...,0]

                    fixed_image = self_inpaint.inpaint_with_mask(deepcopy(main_img), mask, model=model)
                    attempt_mean, attempt_std = self_inpaint.create_uncertainty_map(deepcopy(fixed_image),
                                                                                    grid_size,
                                                                                    overlap_factor,
                                                                                    batch_size=batch_size,
                                                                                    model=model,
                                                                                    mix_method=mix_method,
                                                                                    input_hsv=image_org_hsv)


                    kl_dist_attempt = self_inpaint.kullback_leible_distance_gaussian(attempt_mean, attempt_std, prior_mean, prior_std)

                    entropy_gain = self_inpaint.entropy_gain_gaussian(posterior_std, attempt_std)
                   
                    # save_normailzed_grayscale_image(output_file.replace(suffix, "_vkl{}_atempt".format(iteration) + suffix),
                    #                                 (kl_dist_attempt)/ (12.0))

                    mask = create_mast_from_level_set_seg(kl_dist+kl_dist_attempt-entropy_gain*5, cur_mask,
                                                          threshold=refind_mask_threshold)
                    mask = np.maximum(mask, guarantee_fine_region)


                    # kl_dist_attempt = self_inpaint.kullback_leible_distance_gaussian(attempt_mean, attempt_std, posterior_mean, posterior_std)
                    # save_normailzed_color_image(output_file.replace(suffix, "_vkl{}_gain".format(iteration) + suffix),
                    #                             (kl_dist_attempt - kl_dist) / (12.0))

                    # save_normailzed_grayscale_image(output_file.replace(suffix, "_vkl{}_speed".format(iteration) + suffix),
                    #                             (kl_dist+kl_dist_attempt-entropy_gain*5) / 12.0)

                    fixed_image = self_inpaint.inpaint_with_mask(deepcopy(main_img), mask, model=model)
                    if iteration==num_iteration-1:
                        save_normailzed_color_image(os.path.join(subject_write_dir, slice_name), fixed_image)
                

            else:
                #tmp_test = np.concatenate((main_img, main_img, main_img), axis=1)
                #cv2.imwrite(os.path.join(subject_write_dir, slice_name), tmp_test*255)
                cv2.imwrite(os.path.join(subject_write_dir, slice_name), main_img*255)              
                empty_array = np.zeros_like(main_img)
                cv2.imwrite(path_to_heat_slice, empty_array*255)
        FinalTime=time.time()-initTime
        elapse_time.append(FinalTime)
        #print('This case took {} second'.format(FinalTime))
    
    
    
                
    params = {}



    params['data_path'] = img_path
    params['grid_size'] = grid_size
    params['mix_method'] = mix_method
    params['batch_size'] = batch_size
    params['image_size'] = image_size
    params['field_path'] = field_path
    params['image_org_hsv'] = image_org_hsv
    params['num_iteration'] = num_iteration
    params['overlap_factor'] = overlap_factor
    params['model_weight_path'] = checkpoint_dir
    params['elapsed_time_per_subject'] = elapse_time
    params['refind_mask_threshold'] = refind_mask_threshold
        
    return params




def detection(orig_subjects, inpaint_subjects, res_thr, summary_csv_path):
    summary_dict = {}
    n_orig_subject = len(orig_subjects)
    n_inpaint_subject = len(inpaint_subjects)
    
    if n_orig_subject!=n_inpaint_subject:
        raise Exception("Number of original subjects is not equal to the number of inpainted subjects!")
    else:
        for ix in range(n_orig_subject):
            
            main_subject = orig_subjects[ix]
            inpaint_subject = inpaint_subjects[ix]
            
            name_subject = main_subject.split('/')[-1]
            name_inpaint = inpaint_subject.split('/')[-1]
            
            if name_subject!=name_inpaint:
                raise Exception("Orig subject and inpainted subject are not matched!")
            else:
                
                main_slice_list = get_data_list(main_subject, '.png')
                inpaint_slice_list = get_data_list(inpaint_subject, '.png')
                
                main_img_array = load_img_array(main_slice_list)[:,:,:,0]
                inpaint_img_array = load_img_array(inpaint_slice_list)[:,:,:,0]
                
                residual_array = main_img_array - inpaint_img_array
                residual_array[residual_array<0] = 0
                slice_wise_diff_intensity = get_sum_int(residual_array)
                n_slice_thresholded = len(np.where(slice_wise_diff_intensity>res_thr)[0])
                
                summary_dict['subject_id'] = name_subject
                #
                if n_slice_thresholded>0:
                    prothesis_status = 1
                    print('subject {} contains prothesis!'.format(name_subject))
                else:
                    prothesis_status = 0
                    print('subject {} does not contain prothesis!'.format(name_subject))
                
                summary_dict['prothesis_label'] = prothesis_status
                df = pd.DataFrame(data=summary_dict,  index=[ix])
                if ix == 0:
                    df.to_csv(summary_csv_path,encoding='utf-8', mode='a')
                else:
                    df.to_csv(summary_csv_path,encoding='utf-8', header = None, mode='a')