from collections import defaultdict
import pickle
import pprint
from fuzzywuzzy import fuzz, process
import os
import torch.nn as nn
import shutil
import re

import numpy as np
def couple_roi_names(clinical_names, target):
    matches = {}
    if len(clinical_names) == 1 and len(target) == 1:
        matches[clinical_names[0]] = target[0]
    else:
        for c in clinical_names:
            best_match = process.extractOne(c, target, scorer=fuzz.WRatio)
            if best_match:
                matches[c] = best_match[0]
            
    return matches

def clear_directory_content(PATH):
    for filename in os.listdir(PATH):
        file_path = os.path.join(PATH, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def rm_pss(s, delim='_'):
    return delim.join(c for c in s if c.isalnum())

parts_to_remove_prim = ['in', 'of', 'the', 'left','rt','breast', 'er', 'pr', 'her', 'invasive', 'ovarian', 'urothelial', 'lung', 'endometrial', "esophageal", "neuroendocrine", "endometrioid"]
def process_prim(text):
    text = re.sub(r'[^a-zA-Z]+', ' ', text).lower().strip()
    text = text.replace('adenocarcinoma of the lung squamous cell carcinoma','')
    text = text.replace('unspecified breast cancer','')
    text = text.replace('renal cell', '')
    text = text.replace('renal cell', '')
    parts = [part for part in text.split(' ') if part not in parts_to_remove_prim]
    text = ' '.join(parts)
    text = ' '.join(text.split()).strip()
    return text

parts_to_remove_mets = ['urothelial', 'with', 'large', 'frontal', 'met', 'ca', 'cell', 'mets']

def process_mets(text):
    text = text.replace('Brain Mets - Urothelial','renal')
    text = re.sub(r'[^a-zA-Z]+', ' ', text).lower().strip()
    text = text.replace('gk brain mets lesions','')
    text = text.replace('large cell','')
    text = text.replace('kidney','renal')
    text = text.replace('post op cavity','')
    text = text.replace('brain mets','')
    text = text.replace('brain met','')
    text = text.replace('endometrial','uterus')
    text = text.replace('rcc','renal')
    parts = [part for part in text.split(' ') if part not in parts_to_remove_mets]
    text = ' '.join(parts)
    text = ' '.join(text.split()).strip()
    text = text.replace('melanoma','skin')
    text = text.replace('uterine','uterus')
    text = text.replace('esophageal','esophagus')
    text = text.replace('renal','urinary_system')
    return text
    
mapping_roi = {
    "lt": "left",
    "l": "left",
    "rt": "right",
    "r": "right",
    'crebellar':'cerebellar ',
    "cereb": "cerebellar",
    "cerebe": "cerebellar",
    "cerebelar": "cerebellar",
    "cerebllar": "cerebellar",
    "cerbellar": "cerebellar",
    "cerbella": "cerebellar",
    "cerebella": "cerebellar",
    "cerebellum": "cerebellar",
    "cerebellu": "cerebellar",
    "cerebel": "cerebellar",
    "ant": "anterior",
    "frontal": "frontal",
    "front": "frontal",
    "fro": "frontal",
    "mesial":"medial",
    "lat": "lateral",
    "late": "lateral",
    "inf": "inferior",
    "temp": "temporal",
    "tempora": "temporal",
    "parital": "parietal",
    "pariet": "parietal",
    "par": "parietal",
    "med": "medial",
    "mesial": "medial",
    "median": "medial",
    "occip": "occipital",
    "occ": "occipital",
    "occi": "occipital",
    "occipit": "occipital",
    "sup": "superior",
    "recur": "recurrence",
    "postop": "postoperation",
    "hemisph": "hemisphere",
    "hemisp": "hemisphere",
    "vetrtex": "vertex",
    "calv": "calvarium",
    "fourthventr": "fourth_ventricle",
    "motor": "frontal",
    "premotor": "frontal",
    "corr": "corrected",
    "post": "posterior",
    "cav": "cavity",
    "fla": "",
    "ven": "ventricle",
    "ventr": "ventricle",
    "cortex": "",
    "met": "",
    "su": "superior",
    "parame": "medial",
    "op": "occipital",
    "int": "internal",
    "deep": "",
    "lesion":"",
    "hippocampal" : "temporal",
    "postcentral": "parietal",
    "vertex": "parietal",
    "pontine": "pons",
    "atrium": "ventricle",
    "fourth": "",
    "thalmic": "thalamus",
    "vermis": "vermis",
    "insular": "insula",
    "paramedian": "medial",
}

roi_to_keep = [ 
    'left', 
    'right', 
    'cerebellar', 
    'frontal',
    'temporal', 
    'parietal', 
    'medial', 
    'occipital', 
    'capsule', 
    "tent", 
    "pons", 
    "ventricle", 
    "putamen", 
    "occipital_tent", 
    "insula",
    "thalamus",
    "vermis"
]
    
def process_roi(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    text = re.sub(r'[^a-zA-Z]+', ' ', text).lower().strip()
    
    if text.startswith('lt') and not text.startswith('lt '):
        text = text.replace('lt', 'left ')
        
    if text.startswith('cavity '):
        text = text.replace('cavity ', '')
    
    text = text.replace(' re tx', '')
        
    text = text.replace('frontal pole', 'frontal')
        
    text = text.replace('frontal pre', 'frontal')
    
    text = text.replace('para median', 'medial')
        
    parts = [ part if part not in mapping_roi.keys() else mapping_roi[part] for part in text.split(' ') ]
    
    result = ' '.join(parts).strip()
    
    if 'medial' in result and ('parietal' in result or 'cerebellar' in result or 'occipital' in result or 'frontal' in result or 'temporal' in result):
        result = result.replace('medial', '')
        
    if 'vermis' in result:
        result = "vermis"
    
    result = ' '.join(result.split()).strip()
    
    result = result.replace('posterior central', 'parietal')
    result = result.replace('corpus callo', 'medial')
    result = result.replace('occipital tent', 'occipital_tent')
    
    result = ' '.join([part for part in result.split() if part in roi_to_keep])
    
    return result

def print_statistics():
    PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
    for split in ['train', 'test', 'val']:
        with open(os.path.join(PATH, f'{split}_set.pkl'), 'rb') as f:
            data = pickle.load(f)
            labels, counts = np.unique([x for x in data['label']], return_counts=True)
            print(f'{split}: {dict(zip(labels, counts))}')
            
def print_statistics_subject_lesions():
    PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
    with open(os.path.join(PATH, 'global_data.pkl'), 'rb') as f:
        global_data = pickle.load(f)
        # subjects_id_train_set = sorted([ 463, 158, 247, 408, 234, 421, 431, 346, 487, 274, 338, 105, 293, 314, 227, 330, 391, 313, 270, 127, 324, 342, 121, 103, 114, 115, 151, 244, 245, 246, 467, 455, 152, 147])
        train_set = {}
        for subject in subjects_id_train_set:
            cardinality = [0, 0]
            cardinality[0] = sum([1 for i in range(len(global_data['subject_id'])) if global_data['label'][i] == 'stable' and global_data['subject_id'][i] == subject])
            cardinality[1] = sum([1 for i in range(len(global_data['subject_id'])) if global_data['label'][i] == 'recurrence' and global_data['subject_id'][i] == subject])
            train_set[subject] = cardinality
        
        ratio = 8
        validation_set = {}
        total_negative, total_positive = 0, 0
        
        while total_positive * ratio < total_negative or total_negative == 0 or total_positive == 0:
            subject_with_most_positive = max(train_set, key=lambda x:train_set[x][1] if train_set[x][0] > 0 else 0)
            validation_set.update({subject_with_most_positive: train_set[subject_with_most_positive]})
            total_negative += train_set[subject_with_most_positive][0]
            total_positive += train_set[subject_with_most_positive][1]
            train_set.pop(subject_with_most_positive)
        
        print('Train set')
        pprint.pprint(train_set)
        
        print('\nValidation set')
        pprint.pprint(validation_set)
        
        print(f'\ntrain set\n{sum([x[0] for x in train_set.values()])} : {sum([x[1] for x in train_set.values()])}')
        print(f'validation set\n{sum([x[0] for x in validation_set.values()])} : {sum([x[1] for x in validation_set.values()])}')
        
def print_statistics_globally():
    PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
    with open(os.path.join(PATH, 'global_data.pkl'), 'rb') as f:
        global_data = pickle.load(f)
        subjects_counter = defaultdict(lambda: [0, 0])
        
        for i, subject in enumerate(global_data['subject_id']):
            if global_data['label'][i] == 'stable':
                subjects_counter[subject][0] += 1 
            else:
                subjects_counter[subject][1] += 1 
        
    pprint.pprint(sorted(subjects_counter.items(), key=lambda x: x[1][1], reverse=True))
    
    sum_recurrence = sum([x[1] for x in subjects_counter.values()])
    sum_stable = sum([x[0] for x in subjects_counter.values()])
    
    print(f'\nsum of subjects_counter\n{sum([x[0] + x[1] for x in subjects_counter.values()])}')
    
    print(f'\nsum_recurrence: {sum_recurrence}\nsum_stable:{sum_stable}')
    
def weight_init(component):
    for m in component.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)