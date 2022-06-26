import os
import numpy as np
import pandas as pd
from tools import get_sum_int
from libs.img_utils import load_img_array
from libs.paths_dirs_stuff import get_data_list, get_sub_dirs


orig_img_path = './data/data_2d/image/'
inpaint_img_root = './data/results/'
exp_name = 'pelvis_anomaly_field_map'
inpaint_img_path = os.path.join(inpaint_img_root, exp_name)
summary_csv_name = 'summary_preds.csv'
summary_csv_path = os.path.join(inpaint_img_path, summary_csv_name)


orig_subjects = get_sub_dirs(orig_img_path)[1:]
inpaint_subjects = get_sub_dirs(inpaint_img_path)[1:]





res_thr = 400


def detection(orig_subjects, inpaint_subjects, res_thr):
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
                if n_slice_thresholded>15:
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


