import os
import argparse
from libs.paths_dirs_stuff import get_sub_dirs
from libs.write_slice import execute_2d_writer
from libs.json_stuff import load_json, save_json
from libs.autoinpaint import auto_inpaint, detection
from libs.paths_dirs_stuff import creat_dir, get_data_list 


# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config path")
args = parser.parse_args()
configs = load_json(args.config)


# set variables
nifti_dir = configs['nifti_dir']
write_path_img = configs['write_path_img']
write_path_map = configs['write_path_map']
write_path_results = configs['write_path_results']
exp_name = configs['exp_name']
checkpoint_dir = configs['checkpoint_dir']
n_slice = configs['n_slice']
step_size = configs['step_size']
res_thr = configs['res_thr']




# set fixed params
min_bound_img = -300
max_bound_img = 700
img_dim1 = 256
img_dim2 = 256
min_bound_map = 1800
max_bound_map = 3000
subject_paths = get_data_list(nifti_dir, '.nii.gz')
creat_dir(write_path_img)
creat_dir(write_path_map)
n_subject = len(subject_paths)
img_path = write_path_img
field_path = write_path_map



# executing 2D image preparation
execute_2d_writer(subject_paths, n_subject, write_path_img, min_bound_img, max_bound_img, n_slice, step_size, img_dim1, img_dim2)
execute_2d_writer(subject_paths, n_subject, write_path_map, min_bound_map, max_bound_map, n_slice, step_size, img_dim1, img_dim2)
print('\n'*3, '2D slice preparation is done!')
print('\n'*3, 'Anomaly detection is about to start ...')

# executing the auto-inpainting procedure
params_inpaint = auto_inpaint(write_path_results, exp_name, img_path, field_path, checkpoint_dir)

# set fixed params for detection
orig_subjects = get_sub_dirs(write_path_img)[1:]
inpaint_img_root = write_path_results
exp_name = exp_name
inpaint_img_path = os.path.join(inpaint_img_root, exp_name)
inpaint_subjects = get_sub_dirs(inpaint_img_path)[1:]
summary_csv_name = 'summary_preds.csv'
summary_csv_path = os.path.join(inpaint_img_path, summary_csv_name)

# execute prothesis detection
detection(orig_subjects, inpaint_subjects, res_thr, summary_csv_path)

# saving params in a .json file
exp_params = {}
exp_params['nifti_dir'] = nifti_dir
exp_params['write_path_img'] = write_path_img
exp_params['write_path_map'] = write_path_map
exp_params['write_path_results'] = write_path_results
exp_params['exp_name'] = exp_name
exp_params['checkpoint_dir'] = checkpoint_dir
exp_params['n_slice'] = n_slice
exp_params['step_size'] = step_size
exp_params['res_thr'] = res_thr
exp_params['params_inpaint'] = params_inpaint

json_name = exp_name+'.json'
save_json(inpaint_img_path, json_name, exp_params)
