import os
import re
import pandas as pd
import pprint
from utils import process_roi, process_mets, process_prim

CLINICAL_DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'origin', 'Brain_TR_GammaKnife_Clinical_Information.xlsx')
clinical_data_ll = pd.read_excel(CLINICAL_DATA_FILE_PATH, sheet_name='lesion_level')
clinical_data_cl = pd.read_excel(CLINICAL_DATA_FILE_PATH, sheet_name='course_level')
clinical_data_ll_renaming = {'unique_pt_id': 'subject_id', 'Treatment Course':'course', 'Lesion Location':'roi', 'mri_type':'label', 'duration_tx_to_imag (months)': 'duration_tx_to_imag', 'Fractions':'fractions'}
clinical_data_cl_renaming = {'unique_pt_id': 'subject_id', 'Course #':'course', 'Diagnosis (Only want Mets)':'mets_diagnosis', 'Primary Diagnosis':'primary_diagnosis', 'Age at Diagnosis':'age', 'Gender':'gender'}
clinical_data_ll_types = {'subject_id':'int64','course':'Int8', 'roi':'string', 'label':'string', 'duration_tx_to_imag':'int8', 'fractions':'int8'}
clinical_data_cl_types = {'subject_id':'int64', 'course':'Int8', 'mets_diagnosis':'string', 'primary_diagnosis':'string', 'age':'int8', 'gender':'string'}
clinical_data_ll = clinical_data_ll.drop(clinical_data_ll.columns.difference(clinical_data_ll_renaming.keys()), axis=1).rename(columns=clinical_data_ll_renaming)
clinical_data_cl = clinical_data_cl.drop(clinical_data_cl.columns.difference(clinical_data_cl_renaming.keys()), axis=1).rename(columns=clinical_data_cl_renaming)
clinical_data_ll = clinical_data_ll.astype(clinical_data_ll_types)
clinical_data_cl = clinical_data_cl.astype(clinical_data_cl_types)
clinical_data = pd.merge_ordered(clinical_data_ll, clinical_data_cl, on=['subject_id', 'course'], how='inner')

list_roi = list()
list_mets = list()
list_prim = list()
set_roi = set()
set_mets = set()
set_prim = set()
word_set = set()

for i, row in clinical_data.iterrows():
    list_roi.append( 
        (row['roi'], process_roi(row['roi'])) 
    )
    
    list_mets.append(
        (row['mets_diagnosis'], process_mets(row['mets_diagnosis'])) 
    )
    
    list_prim.append( 
        (row['primary_diagnosis'], process_prim(row['primary_diagnosis'])) 
    )
    
    set_roi.add(process_roi(row['roi'])) 
    set_mets.add(process_mets(row['mets_diagnosis'])) 
    set_prim.add(process_prim(row['primary_diagnosis']))


for roi in set_roi:
    for word in roi.split(' '):
        word_set.add(word)

with open('clinical_data_features.txt', 'w') as f:
    f.write("\n\nLISTS\n\n")
    f.write("ROIs\n")
    f.write(pprint.pformat(list_roi, indent=4))
    f.write("\n\nMETSs\n")
    f.write(pprint.pformat(list_mets, indent=4))
    f.write("\n\nPRIMs\n")
    f.write(pprint.pformat(list_prim, indent=4))
    f.write("\n\nSETS\n\n")
    f.write("\n\nROIs\n")
    f.write(pprint.pformat(set_roi, indent=4))
    f.write("\n\nMETSs\n")
    f.write(pprint.pformat(set_mets, indent=4))
    f.write("\n\nPRIMs\n")
    f.write(pprint.pformat(set_prim, indent=4))
    f.write(f'\n\nROIS: {len(set_roi)}\tMETS: {len(set_mets)}\tPRIM: {len(set_prim)}\n')
    f.write(f'\n\nWORDS: {len(word_set)}\n')
    f.write(pprint.pformat(word_set, indent=4))
