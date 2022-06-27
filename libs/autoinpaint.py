import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from libs.tools import get_sum_int
from scipy.ndimage import morphology
from libs.img_utils import load_img_array
from libs.unet_model import InpaintingUnet
from libs.paths_dirs_stuff import get_data_list, get_sub_dirs, creat_dir
from libs.anomaly_tools import get_pred_candid, get_pred_random_masks, get_organ_mask
from libs.anomaly_tools import gen_random_masks_in_lung, get_candid_mask,get_circle_candids

# GPU memory setting
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)




def auto_inpaint(write_path_results, exp_name, img_path, field_path, checkpoint_dir):

 
    # Set the exp name
    write_path = os.path.join(write_path_results, exp_name)
        
    # Get data and load model
    img_subjects = get_sub_dirs(img_path)[1:]
    img_field_maps = get_sub_dirs(field_path)[1:]   
    
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    model = InpaintingUnet(conv_layer='gconv', load_weights=checkpoint,  train_bn=True)
    
    
    
    ######## Parameters setting 
    
    rad = 25      # moving circle size
    n_top_inpaint = 3          # top candidate regions 
    n_top_circle_select = 5
    image_size = 256
    interval_idx = 10  # pixel interval for moving windows
    res_thr = 0.25   # based on circle size normalization 
    n_subjects = len(img_subjects)
    elapse_time = []
    
    
    
    for ind in range(n_subjects):
        print('Anomaly detection of subject {} out of {}'.format(ind, n_subjects))
        subject_img_path = img_subjects[ind]
        subject_field_path = img_field_maps[ind]
      
        subject_name = subject_img_path.split('/')[-1]
        subject_write_dir = os.path.join(write_path, subject_name)
        creat_dir(subject_write_dir)
        
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
    
            n_top = n_top_inpaint   
            random_masks = gen_random_masks_in_lung(organ_mask, main_img, radius=rad, interval_idx = interval_idx)
            selects_ind = get_circle_candids(random_masks,main_img, n_top_circle_select)
            random_masks_selects = [random_masks[ii] for ii in selects_ind]
            n_random_mask_select = len(random_masks_selects)
            field_sum_int = np.sum(main_field_map)
            #print(field_sum_int)
            if field_sum_int>50: # instead of 0 avoid tiny colon ...
                if n_random_mask_select>0:
                    if n_top>=n_random_mask_select:
                        n_top = n_random_mask_select
                    else:
                        pass
        
                    pred_imgs, corrupt_imgs, resid_ints = get_pred_random_masks(random_masks_selects,main_img,model,organ_mask)
                    union_mask = get_candid_mask(resid_ints,random_masks_selects,radius=rad,n_top=n_top,res_thr=res_thr)
        
                    corrupt_img, my_pred = get_pred_candid(main_img,union_mask, model)
                    
                    #tmp_test = np.concatenate((main_img, corrupt_img, my_pred), axis=1)
                    #cv2.imwrite(os.path.join(subject_write_dir, slice_name), tmp_test*255)
                    #plt.imshow(tmp_test)  
                    cv2.imwrite(os.path.join(subject_write_dir, slice_name), my_pred*255)
                else:
                    #tmp_test = np.concatenate((main_img, main_img, main_img), axis=1)
                    #cv2.imwrite(os.path.join(subject_write_dir, slice_name), tmp_test*255)
                    cv2.imwrite(os.path.join(subject_write_dir, slice_name), main_img*255)
            else:
                #tmp_test = np.concatenate((main_img, main_img, main_img), axis=1)
                #cv2.imwrite(os.path.join(subject_write_dir, slice_name), tmp_test*255)
                cv2.imwrite(os.path.join(subject_write_dir, slice_name), main_img*255)
                
        FinalTime=time.time()-initTime
        elapse_time.append(FinalTime)
        #print('This case took {} second'.format(FinalTime))
    
    
    
                
    params = {}
    params['circle_rad'] = rad
    params['data_path'] = img_path
    params['image_size'] = image_size
    params['field_path'] = field_path
    params['circle_area_thr'] = res_thr
    params['interval_idx'] = interval_idx
    params['n_top_candidate'] = n_top_inpaint
    params['model_weight_path'] = checkpoint_dir
    params['elapsed_time_per_subject'] = elapse_time
    params['n_top_circle_select'] = n_top_circle_select
    
        
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