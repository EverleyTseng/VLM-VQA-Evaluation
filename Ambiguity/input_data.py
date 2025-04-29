import os
import json
import numpy as np
import requests
from io import BytesIO

class Input_Data:
    def __init__(self, file_name, model_format='', root_path='default'):
        
        # ZH: here are the RC paths to images if you need them
        self.coco_train2014_path = '/pl/active/pl_ivc-group/datasets/coco/train2014'
        self.viz_train_path = '/pl/active/pl_ivc-group/datasets/vizwiz/VizWiz-TR/image'
        self.viz_val_path = '/pl/active/pl_ivc-group/datasets/vizwiz/VizWiz-VAL/image'
        self.msra_path = '/pl/active/pl_ivc-group/datasets/MSRA/MSRA_Image'
        self.paco_path = '/pl/active/pl_ivc-group/datasets/PACO_images'
        
        # ZH: root_path is to the GT 
        if root_path == 'default':
            root_path = '/pl/active/pl_ivc-group/everley/projects/ambiguity/datasets'
        dataset_info = self.register_dataset(file_name)
        with open(os.path.join(root_path, file_name), 'r') as f:
            data = json.load(f)
       
        self.file_name = file_name
        self.model_format = model_format
        self.dataset_name = dataset_info['dataset_name']
        self.data_format = dataset_info['data_format']
        self.data = data.copy()
        
    # ZH: here are messy datasets...some are duplicates or useless
    # ZH: but no need to change anything here, just stick to the files indicated in molmo script
    def register_dataset(self, file_name):
        registered_datasets = {'pipeline_paco.json':
                               {'dataset_name': 'pipeline_paco', 'data_format': 'paco'},
                               'AnswerTherapy_ambiguous.json':
                               {'dataset_name': 'AnswerTherapy_ambiguous', 'data_format': 'therapy'},
                               'AnswerTherapy_ambiguous_grd.json':
                               {'dataset_name': 'AnswerTherapy_ambiguous_grd', 'data_format': 'therapy'},
                               'AnswerTherapy_unambiguous_152_grd.json':
                               {'dataset_name': 'AnswerTherapy_unambiguous_152_grd', 'data_format': 'therapy'},
                               'AnswerTherapy_unambiguous_30_grd.json':
                               {'dataset_name': 'AnswerTherapy_unambiguous_30_grd', 'data_format': 'therapy'},
                               'MSRA_RLE_627.json':
                               {'dataset_name': 'MSRA_RLE_627', 'data_format': 'msra'},
                               
                               'MSRA_RLE_500.json':
                               {'dataset_name': 'MSRA_RLE_500', 'data_format': 'msra'},
                               'MSRA_RLE_126.json':
                               {'dataset_name': 'MSRA_RLE_126', 'data_format': 'msra'},
                               'AnswerTherapy_ambiguous_31.json':
                               {'dataset_name': 'AnswerTherapy_ambiguous_31', 'data_format': 'therapy'},
                               'AnswerTherapy_unambiguous.json':
                               {'dataset_name': 'AnswerTherapy_unambiguous', 'data_format': 'therapy'},
                               'AnswerTherapy_unambiguous_858.json':
                               {'dataset_name': 'AnswerTherapy_unambiguous_858', 'data_format': 'therapy'},
                               'AnswerTherapy_ambiguous_31_grd.json':
                               {'dataset_name': 'AnswerTherapy_ambiguous_31_grd', 'data_format': 'therapy'},
                               'AnswerTherapy_unambiguous_31_grd.json':
                               {'dataset_name': 'AnswerTherapy_unambiguous_31_grd', 'data_format': 'therapy'},
                               'LIVE_0_2520_ambiguous_results.json':
                               {'dataset_name': 'paco_live_ambiguous', 'data_format': 'paco'},
                               'LIVE_0_2520_unambiguous_results.json':
                               {'dataset_name': 'paco_live_unambiguous', 'data_format': 'paco'},
                               
                               'PACO_ambiguous_838.json':
                               {'dataset_name': 'PACO_ambiguous_838', 'data_format': 'paco'},
                               'PACO_unambiguous_838.json':
                               {'dataset_name': 'PACO_unambiguous_838', 'data_format': 'paco'},
                               'PACO_ambiguous_1345.json':
                               {'dataset_name': 'PACO_ambiguous_1345', 'data_format': 'paco'},
                               'PACO_unambiguous_1345.json':
                               {'dataset_name': 'PACO_unambiguous_1345', 'data_format': 'paco'},
                              }
        if file_name not in registered_datasets.keys():
            print('input file not registered', file_name, flush=True)
        return registered_datasets[file_name]
                
    # ZH: this is to read gt for querying from the model
    def get_queries(self):
        if self.data_format == 'paco':
            queries = self.read_paco(self.data)
        elif self.data_format == 'therapy':
            queries = self.read_answer_therapy(self.data)
        elif self.data_format == 'msra':
            queries = self.read_msra(self.data)
        else:
            print('wrong data format', self.data_format)
        return queries

    # ZH: I wrote some lame image readings here, you can just follow what I had in molmo
    # ZH: if it is easier for you, you can write your own image reading function
    def read_image(self, query):
        if self.model_format == 'internVL2' or self.model_format == 'glamm':
            return self.read_image_internVL2(query)
        elif self.model_format == 'gpt4o' or self.model_format == 'molmo':
            return self.read_image_gpt4o(query)
        else:
            print('Error: Add model_format in read_image', self.model_format)
    
    # ZH: I do this output periodically to prevent the program from breaking
    # ZH: when calling this function make sure "output_file_name" is pointed to your desired location
    def gen_output(self, results, output_file_name):
        output = {}
        output['dataset_name'] = self.dataset_name
        output['results'] = results.copy()
        if self.data_format == 'therapy':
            output['input_data'] = self.data[:len(results)]
        elif self.data_format == 'msra' or self.data_format == 'paco' :
            output['input_data'] = {key:value for ind, (key, value) in enumerate(self.data.items()) if ind < len(results)}
        else:
            print('Error: Add data type for output', self.data_format)
        with open(output_file_name, 'w') as f:
            json.dump(output, f)
    
    def read_paco(self, data):
        queries = []
        for ind, ann in data.items():
            query = {}
            query['image_url'] = ann['imageURL']
            query['image_path'] = os.path.join(self.paco_path, ann['imageURL'].split('/')[-1])
            query['question'] = ann['final_question']
            query['label'] = ann['ambiguity']
            if 'step2_prompts' in list(ann.keys()):
                query['step2_prompts'] = ann['step2_prompts'].copy()
            queries.append(query)
        return queries
    
    def read_answer_therapy(self, data):
        queries = []
        sub_strings = ['COCO_train2014', 'VizWiz_train', 'VizWiz_val']
        image_roots = [self.coco_train2014_path, self.viz_train_path, self.viz_val_path]
        for ann in data:
            query = {}
            for ind, (sub_str, image_root) in enumerate(zip(sub_strings, image_roots)):
                if sub_str in ann['image_filename']:
                    image_path = os.path.join(image_root, ann['image_filename'])
            query['image_path'] = image_path
            query['question'] = ann['question']
            query['label'] = 'ambiguous' if ann['ambiguous_question'] == 'Yes' else 'unambiguous'
            if 'step2_prompts' in list(ann.keys()):
                query['step2_prompts'] = ann['step2_prompts'].copy()
            queries.append(query)
        return queries
    
    def read_msra(self, data):
        queries = []
        for img_name, ann in data.items():
            query = {}
            query['image_path'] = os.path.join(self.msra_path, img_name)
            query['question'] = ann['question']
            query['label'] = 'unambiguous'
            if 'step2_prompts' in list(ann.keys()):
                query['step2_prompts'] = ann['step2_prompts'].copy()
            queries.append(query)
        return queries
    
    def read_image_internVL2(self, query):
        if self.data_format == 'paco':
            image_data = np.asarray(requests.get(query['image_url']).content)
            image_data = BytesIO(image_data)
# #             query['image_path'] = query['image_url'].split('/')[-1]
#             image_data = query['image_path']

        elif self.data_format == 'therapy' or self.data_format == 'msra':
            image_data = query['image_path']
        return image_data

    def read_image_gpt4o(self, query):
        if self.data_format == 'paco':
#             img_data = requests.get(query['image_url']).content
#             image_path = 'temp.png'
#             with open(image_path, 'wb') as handler:
#                 handler.write(img_data)
            image_path = query['image_path']
        elif self.data_format == 'therapy' or self.data_format == 'msra':
            image_path = query['image_path']
        return image_path
