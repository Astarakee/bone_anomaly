import os
import json



def save_json(write_dir, filename, dict_summary):
    
    json_dir = os.path.join(write_dir, filename)
    with open(json_dir, 'w') as fp:
        json.dump(dict_summary, fp, indent = 4)
 

def load_json(json_path):
    
    with open(json_path, 'r') as json_file:
        configs  = json.load(json_file)      
    return configs

