from fuzzywuzzy import fuzz, process
import os
import shutil
import re
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

def clear_discrete(s):
    return ''.join(char for char in s if char.isalpha())
    
mapping_roi = {
    "lt": "left",
    "l": "left",
    "rt": "right",
    "r": "right",
    "cereb": "cerebellar",
    "cerebe": "cerebellar",
    "cerebelar": "cerebellar",
    "cerebllar": "cerebellar",
    "cerbellar": "cerebellar",
    "ant": "anterior",
    "front": "frontal",
    "lat": "lateral",
    "inf": "inferior",
    "temp": "temporal",
    "parital": "parietal",
    "pariet": "parietal",
    "med": "medial",
    "median": "medial",
    "medial": "medial",
    "occip": "occipital",
    "occi": "occipital",
    "occipit": "occipital",
    "sup": "superior",
    "recur": "recurrence",
    "postop": "postoperation",
    "post op": "post_operation",
    "mesial": "mesial",
    "hemisph": "hemisphere",
    "vetrtex": "vertex",
    "calv": "calvarium",
    "fourthventr": "fourth_ventricle",
    "motor": "motor_cortex",
    "premotor": "premotor_cortex",
    "corr": "corrected",
    "post": "posterior"
}
    
def process_roi(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    
    text = clear_discrete(text).lower()
    sorted_mapping = dict(sorted(mapping_roi.items(), key=lambda item: len(item[0]), reverse=True))
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in sorted_mapping.keys()) + r')\b')
    
    # Replace abbreviations with full terms
    def substitution(match):
        return sorted_mapping[match.group(0)]
    
    result = pattern.sub(substitution, text)
    
    return result