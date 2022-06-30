import os
import argparse
from libs.paths_dirs_stuff import get_sub_dirs
from libs.write_slice import execute_2d_writer
from libs.json_stuff import load_json, save_json
from libs.autoinpaint import auto_inpaint, detection
from libs.paths_dirs_stuff import creat_dir, get_data_list 





parser = argparse.ArgumentParser(description='Pelvis Bone Anomaly with Autoinpainting')
#parser.add_argument('--input_dir', type=str, help='Input directory to the nifti volumes', required=True)
parser.add_argument('--checkpoint_dir', default="/model_weight", type=str, help='Directory to load model weights',required=False)
#parser.add_argument('--write_path_img', default="./data", type=str, help='Output directory to the 2D images', required=False)
#parser.add_argument('--write_path_map', default="./data", type=str, help='Output directory to the 2D images', required=False)
#parser.add_argument('--write_path_results', default="./data", type=str, help='Output directory to store the results', required=False)
parser.add_argument('--exp_name', default="Pelvis_Anomaly", type=str, help='Output directory to the 2D images',  required=False)
parser.add_argument('--n_slice', default="70", type=str, help="number of slices to be analized: either 'all' or an 'integer' ", required=False)
parser.add_argument('--step_size', default=1, type=int, help="interval between slices",  required=False)
parser.add_argument('--res_thr', default=400, type=int, help="threshold value",  required=False)
args = parser.parse_args()



# set variables
nifti_dir = "/input"
data_path = "/data/"

exp_name = args.exp_name
checkpoint_dir = args.checkpoint_dir
n_slice = args.n_slice
step_size = args.step_size
res_thr = args.res_thr

grid_size = 32
overlap_factor = 16
batch_size = 2
mix_method = "avg"
image_org_hsv = False



# set fixed params
write_path_img = os.path.join(data_path, 'data_2d/image')
write_path_map = os.path.join(data_path, 'data_2d/map')
write_path_results = os.path.join(data_path, 'results')
min_bound_img = -300
max_bound_img = 700
img_dim1 = 256
img_dim2 = 256
min_bound_map = 1800
max_bound_map = 3000
subject_paths = get_data_list(nifti_dir, '.nii.gz')
creat_dir(write_path_img)
creat_dir(write_path_map)
creat_dir(write_path_results)
n_subject = len(subject_paths)
img_path = write_path_img
field_path = write_path_map
num_iteration = 5
refind_mask_threshold = 5




# executing 2D image preparation
execute_2d_writer(subject_paths, n_subject, write_path_img, min_bound_img, max_bound_img, n_slice, step_size, img_dim1, img_dim2)
execute_2d_writer(subject_paths, n_subject, write_path_map, min_bound_map, max_bound_map, n_slice, step_size, img_dim1, img_dim2)
print('\n'*3, '2D slice preparation is done!')
print('\n'*3, 'Anomaly detection is about to start ...')

# executing the auto-inpainting procedure
params_inpaint = auto_inpaint(write_path_results, exp_name, img_path, field_path, \
                 checkpoint_dir, grid_size, overlap_factor, batch_size, \
                     mix_method, image_org_hsv, num_iteration, refind_mask_threshold)

# set fixed params for detection
orig_subjects = get_sub_dirs(write_path_img)[1:]
inpaint_img_root = write_path_results
exp_name = exp_name
inpaint_img_path = os.path.join(inpaint_img_root, exp_name)
inpaint_subjects = get_sub_dirs(inpaint_img_path)[1:]
summary_csv_name = 'summary_preds.csv'
summary_csv_path = os.path.join(write_path_results, summary_csv_name)

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
save_json(write_path_results, json_name, exp_params)
