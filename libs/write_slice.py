import os
import numpy as np
from libs.img_utils import normalize, slice_writer
from libs.sitk_stuff import read_nifti, reorient_itk
from libs.paths_dirs_stuff import creat_dir, get_data_list 




def execute_2d_writer(subject_paths, n_subject, write_path_img, min_bound, max_bound, n_slice, step_size, img_dim1, img_dim2):

    for item in enumerate(subject_paths):
        if item[0]%5==0:
            print('preparing 2D slices in progress ...')
        
        subject_name = item[1].split('/')[-1].split('.nii')[0]
        subject_dir = os.path.join(write_path_img, subject_name)
        creat_dir(subject_dir)
        
        img_itk, _, _, _, _, img_direction = read_nifti(item[1])
        img_array = reorient_itk(img_itk)
        img_norm = normalize(img_array, min_bound, max_bound)
        volume_size = img_norm.shape
        volume_depth = volume_size[0]
        
        if n_slice == 'all':
            depth = volume_depth 
        elif int(n_slice)<volume_depth:
            depth = int(n_slice)
        elif int(n_slice)>=volume_depth:
            depth = volume_depth 
            
        
        step_slice = np.arange(0, depth, step_size)
        
        for ix in step_slice:
            img_slice = img_norm[ix]
            slice_name = subject_name+'_slice_'+str(ix)+'.png'
            slice_path = os.path.join(subject_dir, slice_name)
            slice_writer(img_slice, img_dim1, img_dim2, slice_path)
    
    return None

        
        
        
    
    
    
    
    
    
    