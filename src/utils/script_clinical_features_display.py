import os
import re
import pandas as pd
import pprint
from utils import process_roi, process_mets, process_prim

CLINIC_DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'origin', 'Brain_TR_GammaKnife_Clinical_Information.xlsx')
clinic_data_ll = pd.read_excel(CLINIC_DATA_FILE_PATH, sheet_name='lesion_level')
clinic_data_cl = pd.read_excel(CLINIC_DATA_FILE_PATH, sheet_name='course_level')
clinic_data_ll_renaming = {'unique_pt_id': 'subject_id', 'Treatment Course':'course', 'Lesion Location':'roi', 'mri_type':'label', 'duration_tx_to_imag (months)': 'duration_tx_to_imag', 'Fractions':'fractions'}
clinic_data_cl_renaming = {'unique_pt_id': 'subject_id', 'Course #':'course', 'Diagnosis (Only want Mets)':'mets_diagnosis', 'Primary Diagnosis':'primary_diagnosis', 'Age at Diagnosis':'age', 'Gender':'gender'}
clinic_data_ll_types = {'subject_id':'int64','course':'Int8', 'roi':'string', 'label':'string', 'duration_tx_to_imag':'int8', 'fractions':'int8'}
clinic_data_cl_types = {'subject_id':'int64', 'course':'Int8', 'mets_diagnosis':'string', 'primary_diagnosis':'string', 'age':'int8', 'gender':'string'}
clinic_data_ll = clinic_data_ll.drop(clinic_data_ll.columns.difference(clinic_data_ll_renaming.keys()), axis=1).rename(columns=clinic_data_ll_renaming)
clinic_data_cl = clinic_data_cl.drop(clinic_data_cl.columns.difference(clinic_data_cl_renaming.keys()), axis=1).rename(columns=clinic_data_cl_renaming)
clinic_data_ll = clinic_data_ll.astype(clinic_data_ll_types)
clinic_data_cl = clinic_data_cl.astype(clinic_data_cl_types)
clinic_data = pd.merge_ordered(clinic_data_ll, clinic_data_cl, on=['subject_id', 'course'], how='inner')

set_roi = set()
set_mets = set()
set_prim = set()
word_set = set()

for i, row in clinic_data.iterrows():
    set_roi.add(process_roi(row['roi']))
    set_mets.add(process_mets(row['mets_diagnosis']))
    set_prim.add(process_prim(row['primary_diagnosis']))

for roi in set_roi:
    for word in roi.split(' '):
        word_set.add(word)

with open('clinic_data_features.txt', 'w') as f:
    f.write("ROIs\n")
    f.write(pprint.pformat(set_roi, indent=4))
    f.write("\nMETSs\n")
    f.write(pprint.pformat(set_mets, indent=4))
    f.write("\nPRIMs\n")
    f.write(pprint.pformat(set_prim, indent=4))
    f.write(f'\nROIS: {len(set_roi)}\tMETS: {len(set_mets)}\tPRIM: {len(set_prim)}\n')
    f.write(pprint.pformat(word_set, indent=4))
    f.write(f'\nWORDS: {len(word_set)}\n')
