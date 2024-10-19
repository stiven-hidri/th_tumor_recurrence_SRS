import argparse
import os
import pickle
import random
import re
import pandas as pd
import numpy as np
from rt_utils import RTStructBuilder
import shutil
from scipy.ndimage import zoom
from sklearn import preprocessing
import torch
from utils.utils import couple_roi_names, process_mets, process_prim, process_roi
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

class RawData_Reader():
    def __init__(self) -> None:
        #paths
        self.DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.ORIGIN_DATA_FOLDER_PATH = os.path.join(self.DATA_PATH, 'origin')
        self.ORIGIN_DATA_META_FILE_PATH = os.path.join(self.DATA_PATH, 'origin', 'metadata.csv')
        self.CLINIC_DATA_FILE_PATH = os.path.join(self.DATA_PATH, 'origin', 'Brain_TR_GammaKnife_Clinical_Information.xlsx')
        self.OUTPUT_PROCESSED_DATA_FOLDER_PATH = os.path.join(self.DATA_PATH, 'processed')
        self.OUTPUT_RAW_DATA_FOLDER_PATH = os.path.join(self.DATA_PATH, 'raw')

        #adjusting metadata and clinical data dataframes
        self.rawdata_meta = pd.read_csv(self.ORIGIN_DATA_META_FILE_PATH)
        self.clinic_data_ll = pd.read_excel(self.CLINIC_DATA_FILE_PATH, sheet_name='lesion_level')
        self.clinic_data_cl = pd.read_excel(self.CLINIC_DATA_FILE_PATH, sheet_name='course_level')
        
        # 0 = mets_diagnosis, 1 = primary_diagnosis, 2 = age, 3 = gender, 4 = roi, 5 = fractions, 6 = longest_diameter, 7 = number_of_lesions
        self.CLINIC_FEATURES_TO_DISCRETIZE = [0, 1, 3, 4]
        self.CLINIC_FEATURES_TO_NORMALIZE = [2, 5, 6, 7]
        self.CLINIC_FEATURES_TO_KEEP = [1, 4, 6, 7]
        
        self.split_info = None
        
        self.global_data = {'mr': [], 'rtd': [], 'clinic_data': [], 'label': [], 'subject_id': []}
        #final outputs 
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        self.ALL_SPLITS = None
        
    def only_process(self):
        self.__clean_output_directory__(dr.OUTPUT_PROCESSED_DATA_FOLDER_PATH)
        self.__load__()
        self.__preprocess_clinic_data__()
        self.__discretize_categorical_features__()
        
        self.__generate_split__()
        self.__normalize_splits__()
        self.__one_hot__()
        self.__augment_train_set__()
        self.__save__(raw=False)
        print('Done!', end='\r')
        
    def full_run(self, cleanOutDir = True):
        self.__adjust_dataframes__()
        
        if cleanOutDir:
            self.__clean_output_directory__(self.OUTPUT_RAW_DATA_FOLDER_PATH)
            self.__clean_output_directory__(self.OUTPUT_PROCESSED_DATA_FOLDER_PATH)
            
        self.__split_subjects__()
        
        # metadata_by_subjectid = self.rawdata_meta[(self.rawdata_meta['subject_id'] == 'GK_114')].groupby(['subject_id'])
        metadata_by_subjectid = self.rawdata_meta.groupby(['subject_id'])
        total_subjects = len(metadata_by_subjectid)
        
        print('Saving dcm to npy...')
        
        for cnt, (subject_id, values) in enumerate(metadata_by_subjectid):
            metadata_by_studyuid = [(k,v) for (k,v) in values.groupby(['study_uid'])]
            metadata_by_studyuid = sorted(metadata_by_studyuid, key = lambda x: x[1]['study_date'].iloc[0])
            
            subject_id = int(subject_id[0].split('_')[1]) # remove GK
            
            for course ,(_, values) in enumerate(metadata_by_studyuid, start=1):
                path_MR, path_RTD, path_RTS = self.__get_mr_rtd_rts_path__(values)
                
                mr, rtd = self.__get_mr_rtd_resampled__(path_MR, path_RTD)
                
                masks_with_longest_diameter = self.__get_rts__(path_RTS, path_MR, subject_id, course)
                
                masks = { roi: mask_with_longest_diameter[0] for roi, mask_with_longest_diameter in masks_with_longest_diameter.items() }
                longest_diameters = { roi: mask_with_longest_diameter[1] for roi, mask_with_longest_diameter in masks_with_longest_diameter.items() }
                
                rois = masks.keys()
                
                mrs, rtds = self.__mask_and_crop__(rois, mr, rtd, masks)
                
                labels = self.__get_labels__(rois, subject_id, course)
                clinic_data = self.__get_clinic_data__(rois, subject_id, course, longest_diameters)
                
                self.__append_data__(subject_id, mrs, rtds, clinic_data, labels)
                    
            print(f'\rStep: {cnt+1}/{total_subjects}', end='')
        
        self.__save__(raw=True)
        
        self.__preprocess_clinic_data__()
        self.__discretize_categorical_features__()
        
        self.__generate_split__()
        self.__normalize_splits__()
        self.__one_hot__()
        self.__augment_train_set__()
        self.__save__(raw=False)
        
        print('\nData has been read successfully!\n')
        
    def __load__(self):
        print('Loading data...', end='\r')
        with open(os.path.join(self.OUTPUT_RAW_DATA_FOLDER_PATH, 'global_data.pkl'), 'rb') as f:
            self.global_data = pickle.load(f)
    
    def __adjust_dataframes__(self):
        rawdata_meta_renaming = {'Study Date':'study_date','Study UID':'study_uid','Subject ID':'subject_id', 'Modality':'modality', 'File Location':'file_path'}
        clinic_data_ll_renaming = {'unique_pt_id': 'subject_id', 'Treatment Course':'course', 'Lesion Location':'roi', 'mri_type':'label', 'duration_tx_to_imag (months)': 'duration_tx_to_imag', 'Fractions':'fractions'}
        clinic_data_cl_renaming = {'unique_pt_id': 'subject_id', 'Course #':'course', 'Diagnosis (Only want Mets)':'mets_diagnosis', 'Primary Diagnosis':'primary_diagnosis', 'Age at Diagnosis':'age', 'Gender':'gender'}
        
        rawdata_meta_types = {'study_date':"datetime64[ns]" ,'study_uid':'string','subject_id':'string', 'modality':'string', 'file_path':'string'}
        clinic_data_ll_types = {'subject_id':'int64','course':'Int8', 'roi':'string', 'label':'string', 'duration_tx_to_imag':'int8', 'fractions':'int8'}
        clinic_data_cl_types = {'subject_id':'int64', 'course':'Int8', 'mets_diagnosis':'string', 'primary_diagnosis':'string', 'age':'int8', 'gender':'string'}
        
        self.rawdata_meta = self.rawdata_meta.drop(self.rawdata_meta.columns.difference(rawdata_meta_renaming.keys()), axis=1).rename(columns=rawdata_meta_renaming)
        self.clinic_data_ll = self.clinic_data_ll.drop(self.clinic_data_ll.columns.difference(clinic_data_ll_renaming.keys()), axis=1).rename(columns=clinic_data_ll_renaming)
        self.clinic_data_cl = self.clinic_data_cl.drop(self.clinic_data_cl.columns.difference(clinic_data_cl_renaming.keys()), axis=1).rename(columns=clinic_data_cl_renaming)
        
        self.rawdata_meta = self.rawdata_meta.astype(rawdata_meta_types)
        self.clinic_data_ll = self.clinic_data_ll.astype(clinic_data_ll_types)
        self.clinic_data_cl = self.clinic_data_cl.astype(clinic_data_cl_types)
        
        self.clinic_data = pd.merge_ordered(self.clinic_data_ll, self.clinic_data_cl, on=['subject_id', 'course'], how='inner')

    def __clean_output_directory__(self, path_dir):
        for filename in os.listdir(path_dir):
            file_path = os.path.join(path_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
        # for filename in os.listdir(os.path.join(self, 'lesions_cropped')):
        #     file_path = os.path.join('display', 'lesions_cropped', filename)
        #     if os.path.isfile(file_path) or os.path.islink(file_path):
        #         os.remove(file_path)
        #     elif os.path.isdir(file_path):
        #         shutil.rmtree(file_path)
                
        # for filename in os.listdir(os.path.join('display', 'data_whole_union')):
        #     file_path = os.path.join('display', 'lesions_cropped', filename)
        #     if os.path.isfile(file_path) or os.path.islink(file_path):
        #         os.remove(file_path)
        #     elif os.path.isdir(file_path):
        #         shutil.rmtree(file_path)

    def __get_mr_rtd_rts_path__(self, values):
        path_MR = os.path.join(self.ORIGIN_DATA_FOLDER_PATH, values.loc[values['modality'] == 'MR', 'file_path'].iloc[0])
        path_RTD = os.path.join(self.ORIGIN_DATA_FOLDER_PATH, values.loc[values['modality'] == 'RTDOSE', 'file_path'].iloc[0])
        path_RTS = os.path.join(self.ORIGIN_DATA_FOLDER_PATH, values.loc[values['modality'] == 'RTSTRUCT', 'file_path'].iloc[0])
        
        return path_MR, path_RTD, path_RTS

    def __get_mr_rtd_resampled__(self, path_MR, path_RTD):
        mr_dicom = self.__read_dicom__(path_MR)
        rtd_dicom = self.__read_dicom__(path_RTD)        
        
        rtd_resampled_dicom = sitk.Resample(rtd_dicom, mr_dicom)
        
        mr_np = sitk.GetArrayFromImage(mr_dicom)
        rtd_np = sitk.GetArrayFromImage(rtd_resampled_dicom)
        
        return mr_np, rtd_np

    def __read_dicom__(self, path):
        reader = sitk.ImageSeriesReader()
        reader_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(reader_names)
        image = reader.Execute()
        
        if image.GetDimension()==4 and image.GetSize()[3]==1:
            image = image[...,0]
        
        return image

    def __longest_diameter__(self, mask):
        indices = np.argwhere(mask > 0)
        
        if len(indices) < 2:
            return 0
        
        distances = pdist(indices, metric='euclidean')
        max_diameter = np.max(distances)
        
        return max_diameter

    def __get_rts__(self, path_RTS, series_path, subject_id, course):
        rt_struct_path = [os.path.join(path_RTS, f) for f in os.listdir(path_RTS) if f.endswith('.dcm')][0]
        rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path)
        rois = self.clinic_data.loc[(self.clinic_data['subject_id'] == subject_id) & (self.clinic_data['course'] == course), 'roi'].values
        
        couples = couple_roi_names(rois, rtstruct.get_roi_names())
        
        masks = {}
        for roi in rois:
            if 'Skull' not in roi: # exclude skull annotations
                mask_3d = rtstruct.get_roi_mask_by_name(couples[roi])
                
                mask_3d = mask_3d * 1
                mask_3d = np.swapaxes(mask_3d, 0, 2)
                mask_3d = np.swapaxes(mask_3d, 1, 2)
                masks[roi] = (mask_3d, self.__longest_diameter__(mask_3d))
            else:
                print('Skull!')
                
        return masks

    def __get_labels__(self, rois, subject_id, course):
        to_return = []
        
        for roi in rois:    
            label = self.clinic_data.loc[(self.clinic_data['subject_id']==subject_id)&(self.clinic_data['course']==course)&(self.clinic_data['roi']==roi), ['label']].values[0][0]
            to_return.append(label)
            
        return to_return

    def __get_clinic_data__(self, rois, subject_id, course, longest_diameters):
        to_return = []
        
        for roi in rois:
            clinic_data_row = self.clinic_data.loc[(self.clinic_data['subject_id']==subject_id)&(self.clinic_data['course']==course)&(self.clinic_data['roi']==roi), ['mets_diagnosis', 'primary_diagnosis', 'age', 'gender', 'roi', 'fractions']].values[0]
            clinic_data_row[0] = clinic_data_row[0]
            clinic_data_row[1] = clinic_data_row[1]
            clinic_data_row[3] = clinic_data_row[3]
            clinic_data_row[4] = clinic_data_row[4]
            clinic_data_row = np.append(clinic_data_row, longest_diameters[roi])
            clinic_data_row = np.append(clinic_data_row, self.clinic_data.groupby(['subject_id', 'course']).size().get((subject_id, course), 0))
            to_return.append(clinic_data_row)
            
        return to_return

    def __mask_and_crop__(self, rois, mr, rtd, masks):
        mr_return, rtd_return = [], []
        
        for roi in rois:
            mr_masked = masks[roi] * mr
            rtd_masked = masks[roi] * rtd
            
            mr_masked_cropped = self.__crop_les__(mr_masked)
            rtd_masked_cropped = self.__crop_les__(rtd_masked)
            
            mr_return.append(mr_masked_cropped)
            rtd_return.append(rtd_masked_cropped)
        
        return mr_return, rtd_return
    
    def __crop_les__(self, image, desired_shape = (40, 40, 40)):
        true_points = np.argwhere(image)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        cropped_arr = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]

        cropped_shape = np.array(cropped_arr.shape)
        
        factor = np.min(np.divide(desired_shape,cropped_shape))
        
        if factor < 1:
            cropped_arr = zoom(cropped_arr, (factor,factor,factor), order=1)
        
        cropped_shape = np.array(cropped_arr.shape)
        
        pad_before = np.maximum((desired_shape - cropped_shape) // 2, 0)
        pad_after = np.maximum(desired_shape - cropped_shape - pad_before, 0)
        
        padded_arr = np.pad(cropped_arr, (
            (pad_before[0], pad_after[0]), 
            (pad_before[1], pad_after[1]), 
            (pad_before[2], pad_after[2])
        ), mode='constant', constant_values=0)
        
        return padded_arr

    def __append_data__(self, subject_id, mrs, rtds, clinic_data, labels):
        for i in range(len(labels)):
            self.global_data['subject_id'].append(subject_id)
            self.global_data['mr'].append(np.float64(mrs[i]))
            self.global_data['rtd'].append(np.float64(rtds[i]))
            self.global_data['clinic_data'].append(clinic_data[i])
            self.global_data['label'].append(labels[i])

    def __augment_train_set__(self):
        print('Augmenting data...', end='\r')
        i = 0
        total_len = len(self.train_set['mr'])
        while i < total_len:
            if int(self.train_set['label'][i]) == 1:
                mr, rtd = self.train_set['mr'][i], self.train_set['rtd'][i]
                augmented_mr = self.__rotate_image__(mr)
                augmented_rtd = self.__rotate_image__(rtd)
                
                # flip
                augmented_mr.extend([torch.flip(mr, dims=[0]), torch.flip(mr, dims=[1])])
                augmented_rtd.extend([torch.flip(rtd, dims=[0]), torch.flip(rtd, dims=[1])])

                augmented_label = [self.train_set['label'][i]] * len(augmented_mr)
                augmented_clinic_data = [self.train_set['clinic_data'][i]] * len(augmented_mr)
                
                self.train_set['mr'].extend(augmented_mr)
                self.train_set['rtd'].extend(augmented_rtd)
                self.train_set['clinic_data'].extend(augmented_clinic_data)
                self.train_set['label'] = torch.cat((self.train_set['label'], torch.tensor(augmented_label).to(torch.float32).view(-1, 1)), dim=0)
                
            i+=1

    def __rotate_image__(self, image) -> dict:
        rotated_images = []
        axes = {'z':[0,1], 'x':[1, 2], 'y':[0,2]}
        angles = [90, 180, 270]
        
        for axes_key in axes.keys():
            for angle in angles:
                k = angle // 90  # Number of 90-degree rotations
                rotated_images.append(torch.rot90(image, k, axes[axes_key]))
        
        return rotated_images
    
    
    
    def __preprocess_clinic_data__(self):
        for i, value in enumerate(self.global_data['clinic_data']):
            self.global_data['clinic_data'][i][0] = process_mets(self.global_data['clinic_data'][i][0])
            self.global_data['clinic_data'][i][1] = process_prim(self.global_data['clinic_data'][i][1])
            self.global_data['clinic_data'][i][3] = self.global_data['clinic_data'][i][3].lower().strip()
            self.global_data['clinic_data'][i][4] = process_roi(self.global_data['clinic_data'][i][4])
    
    def __discretize_categorical_features__(self):
        self.global_data['label'] = np.array([1 if label == 'recurrence' else 0 for label in self.global_data['label']])
        
        tmp = np.array(self.global_data['clinic_data'])
        
        for j in self.CLINIC_FEATURES_TO_DISCRETIZE:
            tmp[:,j] = preprocessing.LabelEncoder().fit_transform(tmp[:,j])
        
        self.global_data['clinic_data'] = tmp.astype(np.int64)
    
    def __one_hot__(self):        
        max_values = {}
        for split in self.ALL_SPLITS:
            for tensor in split['clinic_data']:
                for j, feature in enumerate(tensor):
                    if j in self.CLINIC_FEATURES_TO_DISCRETIZE:
                        if j not in max_values:
                            max_values[j] = int(feature)
                        else:
                            max_values[j] = max(max_values[j], int(feature))
        
        for split in self.ALL_SPLITS:
            for i_tensor in range(len(split['clinic_data'])):
                new_tensor = [split['clinic_data'][i_tensor][j] for j in self.CLINIC_FEATURES_TO_NORMALIZE if j in self.CLINIC_FEATURES_TO_KEEP]                
                for j in self.CLINIC_FEATURES_TO_DISCRETIZE:
                    if j in self.CLINIC_FEATURES_TO_KEEP:
                        new_tensor.append(F.one_hot(split['clinic_data'][i_tensor][j].long(), num_classes=max_values[j]+1).float()[0])
                split['clinic_data'][i_tensor] = torch.cat(new_tensor)
                
        print(f"\n\nLength of clinical features: {len(self.train_set['clinic_data'][-1])}\n\n")
    
    def __compute_statistics__(self, data):
        statistics = {}
                
        for key in data.keys():
            if key != 'label':
                current_data = torch.cat([tensor.view(-1) for tensor in data[key]])
                
                if key == 'clinic_data':
                    statistics[key] = {}
                    for j in self.CLINIC_FEATURES_TO_NORMALIZE:
                        selected_data = []
                        
                        for tensor in data[key]:
                            selected_data.extend(tensor[self.CLINIC_FEATURES_TO_NORMALIZE].view(-1).tolist())  # Flatten and add to the list

                        all_data = torch.tensor(selected_data)

                        # Compute statistics
                        statistics[key][j] = {
                            'min': all_data.min().item(),
                            'max': all_data.max().item(),
                            'mean': all_data.mean().item(),
                            'std': all_data.std().item()
                        }
                else:
                    statistics[key] = {
                        'min': current_data.min().item(),
                        'max': current_data.max().item(),
                        'mean': current_data.mean().item(),
                        'std': current_data.std().item()
                    }
        
        return statistics        
    
    def __minmax_scaling__(sample, statistics):
        sample = (sample - statistics['min']) / (statistics['max'] - statistics['min'])
        return sample
        
    def __z_norm__(sample, statistics):
        sample = (sample - statistics['mean']) / statistics['std']
        return sample
    
    def __normalize__(self, split, statistics, f):
        for key in split.keys():
            if key != 'label':
                if key == 'clinic_data':
                    for j in statistics[key].keys():
                        for i_tensor in range(len(split[key])):
                            split[key][i_tensor][j] = f(split[key][i_tensor][j], statistics[key][j])
                else:
                    for i_tensor in range(len(split[key])):
                        split[key][i_tensor] = f(split[key][i_tensor], statistics[key])
    
    def __normalize_splits__(self, f = __minmax_scaling__):
        statistics = self.__compute_statistics__(self.train_set)
        
        with open(os.path.join(self.OUTPUT_PROCESSED_DATA_FOLDER_PATH, 'statistics.pkl'), 'wb') as f_writer:
            pickle.dump(statistics, f_writer)
        
        for split in self.ALL_SPLITS:
            self.__normalize__(split, statistics, f)
    
    def __generate_split__(self):
        self.__split_subjects__()
        #final outputs 
        self.train_set = {'mr': [], 'rtd': [], 'clinic_data': [], 'label': []}
        self.val_set = {'mr': [], 'rtd': [], 'clinic_data': [], 'label': []}
        self.test_set = {'mr': [], 'rtd': [], 'clinic_data': [], 'label': []}
        
        for i, subject_id in enumerate(self.global_data['subject_id']):
            target = None
            
            if subject_id in self.split_info['train']:
                target = self.train_set                
            if subject_id in self.split_info['test']:
                target = self.test_set
            elif subject_id in self.split_info['val']:
                target = self.val_set
            
            target['label'].append(self.global_data['label'][i])
            target['clinic_data'].append(torch.Tensor(self.global_data['clinic_data'][i]).to(torch.float32).view(-1, 1))
            target['mr'].append(torch.Tensor(self.global_data['mr'][i]).to(torch.float32))
            target['rtd'].append(torch.Tensor(self.global_data['rtd'][i]).to(torch.float32))
            
        self.ALL_SPLITS = [self.train_set, self.val_set, self.test_set]
            
        for split in self.ALL_SPLITS:
            split['label'] = torch.tensor(split['label']).to(torch.float32).view(-1, 1)
    
    def __save__(self, raw = True):
        print('Saving data...', end='\r')
        
        if raw:
            target = self.OUTPUT_RAW_DATA_FOLDER_PATH
            
            with open(os.path.join(target, 'global_data.pkl'), 'wb') as f:
                pickle.dump(self.global_data, f)
        else: 
            target = self.OUTPUT_PROCESSED_DATA_FOLDER_PATH
        
            with open(os.path.join(target, 'train_set.pkl'), 'wb') as f:
                pickle.dump(self.train_set, f)
            
            with open(os.path.join(target, 'test_set.pkl'), 'wb') as f:
                pickle.dump(self.test_set, f) 
                
            with open(os.path.join(target, 'val_set.pkl'), 'wb') as f:
                pickle.dump(self.val_set, f) 
            
    def __split_subjects__(self):
        subjects_test   =   [ 427, 243, 257, 224, 420, 312, 316, 199, 219, 492, 332, 364, 132 ]
        # old
        # subjects_train  =   [ 463, 158, 247, 408, 234, 421, 431, 346, 487, 274, 338, 105, 293, 314, 227, 330, 391, 313, 270, 127, 324, 342, 121, 103, 114, 115, 151, 244, 245, 246, 467 ]
        # subjects_val    =   [ 455, 152, 147]
        
        # new
        subjects_train =      [ 103, 105, 114, 115, 121, 127, 147, 151, 158, 227, 234, 244, 245, 246, 247, 293, 313, 314, 324, 330, 342, 346, 391, 408, 421, 431, 455, 463, 467, 487 ]
        subjects_val =        [ 152, 270, 274, 338 ]
        
        self.split_info = {
            'train': subjects_train, 
            'test': subjects_test,
            'val': subjects_val
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_process', action='store_true')
    args = parser.parse_args()
    
    dr = RawData_Reader()
    
    if args.only_process:
        dr.only_process()
    else:
        dr.full_run(cleanOutDir=False)
