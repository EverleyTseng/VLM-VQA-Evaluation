import os
import cv2
import json
import numpy as np
from datetime import datetime

class Mask_Handler:
    def __init__(self, mask_path, job_type):
        self.mask_path = mask_path
        self.job_type = job_type
        self.dictionary = {}
    
    def save_mask(self, dataset_name, prompt_type, mask):
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d%H%M%S%f")[:17]
        mask_name = '{}_{}_{}.png'.format(dataset_name, prompt_type, timestamp_str)
        cv2.imwrite(os.path.join(self.mask_path, mask_name), mask)
        self.dictionary[mask_name] = {'dataset_name': dataset_name,
                                      'prompt_type': prompt_type,
                                      'job_type': self.job_type}
        return mask_name
        
    def export_dict(self):
        dict_path = os.path.join(self.mask_path, 'dictionary.json')
        if os.path.isfile(dict_path):
            with open(dict_path, 'r') as f:
                old_dictionary = json.load(f)
            self.dictionary.update(old_dictionary)
            
        with open(dict_path, 'w') as f:
            json.dump(self.dictionary, f)
        
        
        
        
        
        
        
        
        