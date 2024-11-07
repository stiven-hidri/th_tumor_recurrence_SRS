import os
import pickle
import re
import pandas as pd
import numpy as np
import pydicom as dicom
import pydicom
from rt_utils import RTStructBuilder
import shutil
from scipy.ndimage import zoom
from sklearn import preprocessing
from utils import rm_pss, couple_roi_names
import random
import SimpleITK as sitk

class RawData_Reader():
    def __init__(self) -> None:
        #paths
        self.RAWDATA_FOLDER_PATH = os.path.join('..','..', 'descriptive')
        self.RAWDATA_META_FILE_PATH = os.path.join(self.RAWDATA_FOLDER_PATH, 'metadata.csv')
        self.CLINICAL_DATA_FILE_PATH = os.path.join('..','..', 'Brain-TR-GammaKnife-Clinical-Information.xlsx')
        self.OUTPUT_DATA_FOLDER_PATH = os.path.join('..','..', 'data')

        #adjusting metadata and clinical data dataframes
        self.rawdata_meta = pd.read_csv(self.RAWDATA_META_FILE_PATH)
        self.clinical_data_ll = pd.read_excel(self.CLINICAL_DATA_FILE_PATH, sheet_name='lesion_level')
        self.clinical_data_cl = pd.read_excel(self.CLINICAL_DATA_FILE_PATH, sheet_name='course_level')
        
        self.clinical_data = None

        self.split_info = None
        
        self.global_data = {'mr': [], 'rtd': [], 'clinical_data': [], 'label': [], 'subject_id': []}
        #final outputs 
        self.train_set = {'mr': [], 'rtd': [], 'clinical_data': [], 'label': []}
        self.test_set = {'mr': [], 'rtd': [], 'clinical_data': [], 'label': []}
        
    def main(self, cleanOutDir = True):
        self.__adjust_dataframes__()
        
        if cleanOutDir:
            self.__clean_output_directory__()
        
        # metadata_by_subjectid = self.rawdata_meta[(self.rawdata_meta['subject_id'] == 'GK_487') | (self.rawdata_meta['subject_id'] == 'GK_132') | (self.rawdata_meta['subject_id'] == 'GK_152')].groupby(['subject_id'])
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
                masks = self.__get_rts__(path_RTS, path_MR, subject_id, course)
                
                rois = masks.keys()
                
                labels = self.__get_labels__(rois, subject_id, course)                  
                mr, rtd = self.__mask_and_crop__(rois, mr, rtd, masks)
                self.__save_lesions__(rois, mr, rtd, subject_id, course, labels)
                #mr, rtd = self.__normalize__(mr, rtd)
            
                # clinical_data = self.__get_clinical_data__(rois, subject_id, course)
                    
            print(f'\rStep: {cnt+1}/{total_subjects}', end='')
        
        #self.__encode_discrete__()
        
        print('\nData has been read successfully!\n')
        
    def __adjust_dataframes__(self):
        rawdata_meta_renaming = {'Study Date':'study_date','Study UID':'study_uid','Subject ID':'subject_id', 'Modality':'modality', 'File Location':'file_path'}
        clinical_data_ll_renaming = {'unique_pt_id': 'subject_id', 'Treatment Course':'course', 'Lesion Location':'roi', 'mri_type':'label', 'duration_tx_to_imag (months)': 'duration_tx_to_imag', 'Fractions':'fractions'}
        clinical_data_cl_renaming = {'unique_pt_id': 'subject_id', 'Course #':'course', 'Diagnosis (Only want Mets)':'mets_diagnosis', 'Primary Diagnosis':'primary_diagnosis', 'Age at Diagnosis':'age', 'Gender':'gender'}
        
        rawdata_meta_types = {'study_date':"datetime64[ns]" ,'study_uid':'string','subject_id':'string', 'modality':'string', 'file_path':'string'}
        clinical_data_ll_types = {'subject_id':'int64','course':'Int8', 'roi':'string', 'label':'string', 'duration_tx_to_imag':'int8', 'fractions':'int8'}
        clinical_data_cl_types = {'subject_id':'int64', 'course':'Int8', 'mets_diagnosis':'string', 'primary_diagnosis':'string', 'age':'int8', 'gender':'string'}
        
        self.rawdata_meta = self.rawdata_meta.drop(self.rawdata_meta.columns.difference(rawdata_meta_renaming.keys()), axis=1).rename(columns=rawdata_meta_renaming)
        self.clinical_data_ll = self.clinical_data_ll.drop(self.clinical_data_ll.columns.difference(clinical_data_ll_renaming.keys()), axis=1).rename(columns=clinical_data_ll_renaming)
        self.clinical_data_cl = self.clinical_data_cl.drop(self.clinical_data_cl.columns.difference(clinical_data_cl_renaming.keys()), axis=1).rename(columns=clinical_data_cl_renaming)
        
        self.rawdata_meta = self.rawdata_meta.astype(rawdata_meta_types)
        self.clinical_data_ll = self.clinical_data_ll.astype(clinical_data_ll_types)
        self.clinical_data_cl = self.clinical_data_cl.astype(clinical_data_cl_types)
        
        self.clinical_data = pd.merge_ordered(self.clinical_data_ll, self.clinical_data_cl, on=['subject_id', 'course'], how='inner')

    def __clean_output_directory__(self):
        for filename in os.listdir(os.path.join('.', 'lesions_cropped')):
            file_path = os.path.join('.', 'lesions_cropped', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                
        # for filename in os.listdir(os.path.join('.', 'data_whole_union')):
        #     file_path = os.path.join('.', 'lesions_cropped', filename)
        #     if os.path.isfile(file_path) or os.path.islink(file_path):
        #         os.remove(file_path)
        #     elif os.path.isdir(file_path):
        #         shutil.rmtree(file_path)

    def __get_mr_rtd_rts_path__(self, values):
        path_MR = os.path.join(self.RAWDATA_FOLDER_PATH, values.loc[values['modality'] == 'MR', 'file_path'].iloc[0])
        path_RTD = os.path.join(self.RAWDATA_FOLDER_PATH, values.loc[values['modality'] == 'RTDOSE', 'file_path'].iloc[0])
        path_RTS = os.path.join(self.RAWDATA_FOLDER_PATH, values.loc[values['modality'] == 'RTSTRUCT', 'file_path'].iloc[0])
        
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

    def __get_rts__(self, path_RTS, series_path, subject_id, course):
        rt_struct_path = [os.path.join(path_RTS, f) for f in os.listdir(path_RTS) if f.endswith('.dcm')][0]
        rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path)
        rois = self.clinical_data.loc[(self.clinical_data['subject_id'] == subject_id) & (self.clinical_data['course'] == course), 'roi'].values
        
        couples = couple_roi_names(rois, rtstruct.get_roi_names())
        
        masks = {}
        for roi in rois:
            if 'Skull' not in roi: # exclude skull annotations
                mask_3d = rtstruct.get_roi_mask_by_name(couples[roi])
                
                mask_3d = mask_3d * 1
                mask_3d = np.swapaxes(mask_3d, 0, 2)
                mask_3d = np.swapaxes(mask_3d, 1, 2)
                masks[roi] = mask_3d
                
        return masks

    def __get_labels__(self, rois, subject_id, course):
        to_return = []
        
        for roi in rois:    
            label = self.clinical_data.loc[(self.clinical_data['subject_id']==subject_id)&(self.clinical_data['course']==course)&(self.clinical_data['roi']==roi), ['label']].values[0][0]
            to_return.append(label)
            
        return to_return

    def __get_clinical_data__(self, rois, subject_id, course):
        to_return = []
        
        for roi in rois:
            clinical_data_row = self.clinical_data.loc[(self.clinical_data['subject_id']==subject_id)&(self.clinical_data['course']==course)&(self.clinical_data['roi']==roi), ['mets_diagnosis', 'primary_diagnosis', 'age', 'gender', 'duration_tx_to_imag', 'fractions']].values[0]
            clinical_data_row[0] = re.sub('[^0-9a-zA-Z]+', '_', clinical_data_row[0]).lower()
            to_return.append(clinical_data_row)
            
        return to_return
    
    def __save_lesions__(self, rois, mr, rtd, subject_id, course, labels):
        lesion_folder_path = os.path.join('.','lesions_cropped')
        
        for i, roi in enumerate(rois):
            processed_roi = rm_pss(roi, '')
            lesion_file_name = f'{subject_id}_{course}_{processed_roi}.pkl'
            
            to_save = {
                'mr': mr[i],
                'rtd': rtd[i],
                'label':labels[i]
            }
            
            with open(os.path.join(lesion_folder_path, lesion_file_name), 'wb') as f:
                pickle.dump(to_save, f)
    
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
    
    # def __crop_les__(self, image):
    #     true_points = np.argwhere(image)
    #     top_left = true_points.min(axis=0)
    #     bottom_right = true_points.max(axis=0)
    #     cropped_arr = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]
    #     return cropped_arr
    
    # def __crop_les__(self, image, desired_shape = (40, 40, 40)):
        
    #     true_points = np.argwhere(image)
    #     top_left = true_points.min(axis=0)
    #     bottom_right = true_points.max(axis=0)
    #     cropped_arr = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]

    #     cropped_shape = np.array(cropped_arr.shape)
        
    #     if( cropped_shape[0] > desired_shape[0] or cropped_shape[1] > desired_shape[1] or cropped_shape[2] > desired_shape[2]):
    #         cropped_arr = zoom(cropped_arr, (.4,.4,.4))
        
    #     cropped_shape = np.array(cropped_arr.shape)
        
    #     # Calculate the amount of padding needed in each dimension
    #     pad_before = np.maximum((desired_shape - cropped_shape) // 2, 0)
    #     pad_after = np.maximum(desired_shape - cropped_shape - pad_before, 0)
        
    #     # Pad the cropped array with zeros if it is smaller than the desired shape
    #     padded_arr = np.pad(cropped_arr, (
    #         (pad_before[0], pad_after[0]), 
    #         (pad_before[1], pad_after[1]), 
    #         (pad_before[2], pad_after[2])
    #     ), mode='constant', constant_values=0)
        
    #     try:
    #         assert padded_arr.shape == desired_shape
    #     except AssertionError:
    #         print(padded_arr.shape, cropped_arr.shape)
        
    #     return padded_arr
    
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
        
        try:
            assert padded_arr.shape == desired_shape
        except AssertionError:
            print(padded_arr.shape, cropped_arr.shape)
        
        return padded_arr


    def __normalize__(Self, mr, rtd):
        for i in range(len(mr)):
            cur_mr, cur_rtd = mr[i], rtd[i]
            
            cur_mr = np.float64(cur_mr) 
            cur_rtd = np.float64(cur_rtd)
            
            cur_mr /= 65535.
            cur_rtd /= 65535.
            
            mr[i], rtd[i] = cur_mr, cur_rtd
        
        return mr, rtd

    
    def __encode_discrete__(self):
        for i in range(len(self.global_data['label'])):
            self.global_data['label'][i] = [1 if x == 'recurrence' else 0 for x in self.global_data['label'][i]]
        
        tmp = np.array([l for sl in self.global_data['clinical_data'] for l in sl])        
        tmp_enc = [preprocessing.LabelEncoder().fit(tmp[:,j]) for j in[0,1,3] ]
        
        for i in range(len(self.global_data['clinical_data'])):
            tmp_cur = np.array(self.global_data['clinical_data'][i])
            for n, j in enumerate([0, 1, 3]):
                tmp_cur[:,j] = tmp_enc[n].transform(tmp_cur[:,j])

            self.global_data['clinical_data'][i] = tmp_cur.tolist()
    
     



if __name__ == '__main__':
    dr = RawData_Reader()
    dr.main(cleanOutDir=True)