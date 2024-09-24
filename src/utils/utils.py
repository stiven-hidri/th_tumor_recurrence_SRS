import pickle
from fuzzywuzzy import fuzz, process
import os
import shutil
import re

import numpy as np
def couple_roi_names(clinical_names, target):
    matches = {}
    if len(clinical_names) == 1 and len(clinical_names) == 1:
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

def process_prim(text):
    text = re.sub(r'[^a-zA-Z]+', ' ', text).lower().strip()
    return text

mapping_mets = {
    "brain mets": '',
    "brain met": '',
}

def process_mets(text):
    text = re.sub(r'[^a-zA-Z]+', ' ', text).lower().strip()
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
    "ant": "anterior",
    "frontal": "frontal",
    "front": "frontal",
    "fro": "frontal",
    "mesial":"medial",
    "lat": "lateral",
    "inf": "inferior",
    "temp": "temporal",
    "tempora": "temporal",
    "parital": "parietal",
    "pariet": "parietal",
    "par": "parietal",
    "med": "medial",
    "mesial": "medial",
    "median": "medial",
    "medial": "medial",
    "occip": "occipital",
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
    "motor": "motor_cortex",
    "premotor": "premotor_cortex",
    "corr": "corrected",
    "post": "posterior",
    "cav": "cavity"
}
    
def process_roi(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    text = re.sub(r'[^a-zA-Z]+', ' ', text).lower().strip()
    
    if text.startswith('lt') and not text.startswith('lt '):
        text = text.replace('lt', 'left ')
    
    parts = [ part if part not in mapping_roi.keys() else mapping_roi[part] for part in text.split(' ') ]
    
    result = ' '.join(parts)
    
    return result

def print_statistics():
    PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
    for split in ['train', 'test', 'val']:
        with open(os.path.join(PATH, f'{split}_set.pkl'), 'rb') as f:
            data = pickle.load(f)
            labels, counts = np.unique([x for x in data['label']], return_counts=True)
            print(f'{split}: {dict(zip(labels, counts))}')
