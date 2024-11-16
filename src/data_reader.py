import argparse
import os
import pickle
import pprint
import re
import pandas as pd
import numpy as np
from rt_utils import RTStructBuilder
import shutil
from utils.utils import couple_roi_names, process_mets, process_prim, process_roi
import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import pdist
import sys
import matplotlib.pyplot    as plt

class RawData_Reader():
    def __init__(self) -> None:
        
        self.DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        
        if os.path.isdir(os.path.join(self.DATA_PATH, 'origin')):
            self.ORIGIN_DATA_FOLDER_PATH = os.path.join(self.DATA_PATH, 'origin')
            self.ORIGIN_DATA_META_FILE_PATH = os.path.join(self.DATA_PATH, 'origin', 'metadata.csv')
            self.CLINICAL_DATA_FILE_PATH = os.path.join(self.DATA_PATH, 'origin', 'Brain_TR_GammaKnife_Clinical_Information.xlsx')
            #adjusting metadata and clinical data dataframes
            self.rawdata_meta = pd.read_csv(self.ORIGIN_DATA_META_FILE_PATH)
            self.clinical_data_ll = pd.read_excel(self.CLINICAL_DATA_FILE_PATH, sheet_name='lesion_level')
            self.clinical_data_cl = pd.read_excel(self.CLINICAL_DATA_FILE_PATH, sheet_name='course_level')
        
        self.OUTPUT_PROCESSED_DATA_FOLDER_PATH = os.path.join(self.DATA_PATH, 'processed')
        self.OUTPUT_RAW_DATA_FOLDER_PATH = os.path.join(self.DATA_PATH, 'raw')
        
        # 0 = mets_diagnosis, 1 = primary_diagnosis, 2 = age, 3 = gender, 4 = roi, 5 = fractions, 6 = longest_diameter, 7 = number_of_lesions
        self.CLINIC_FEATURES_TO_DISCRETIZE = [0, 1, 3, 4]
        self.CLINIC_FEATURES_TO_NORMALIZE = [2, 5, 6, 7]
        self.CLINIC_FEATURES_TO_KEEP = [0, 1, 2, 3, 4, 6, 7]
        
        self.split_info = None
        
        self.global_data = {'mr': [], 'rtd': [], 'clinical_data': [], 'label': [], 'subject_id': []}
        #final outputs 
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
        self.ALL_SPLITS = None
    
    def plot(self, stuff, keys, progressive=0, c_init=6):
        
        r = len(keys)  # Number of rows (types of images)
        indexes = np.where(np.sum(stuff['mr'][progressive], axis=(1, 2)) > 0)[0]  # Find non-empty slices more efficiently
        
        c = c_init if len(indexes) >= c_init else len(indexes)  # Number of columns (slices)
        indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
        
        fig = plt.figure(figsize=(13, 8))  # Optional: Smaller figure size
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)  # Adjust values as needed
        
        fig.suptitle(f'Label: {stuff["label"][progressive]}', fontsize=12)

        for i, key in enumerate(keys):
            image = stuff[key][progressive]
            for j in range(c):
                ax = fig.add_subplot(r, c, i * c + j + 1)  # Create subplot
                ax.axis('off')
                ax.imshow(image[indexes[j]])
                if j == 0:
                    ax.set_title(f'{key}', fontsize=10)
                    
        plt.show()
        
    def full_run(self, cleanOutDir = True, debug = False):
        self.__adjust_dataframes__()
        
        if cleanOutDir:
            self.__clean_output_directory__(self.OUTPUT_PROCESSED_DATA_FOLDER_PATH)
        
        if debug:
            metadata_by_subjectid = self.rawdata_meta[(self.rawdata_meta['subject_id'] == 'GK_243')].groupby(['subject_id'])
        else:
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
                
                masks_with_longest_diameter = self. __get_rts__(path_RTS, path_MR,subject_id, course)
                
                masks = { roi: mask_with_longest_diameter[0] for roi, mask_with_longest_diameter in masks_with_longest_diameter.items() }
                
                assert ( masks[list(masks.keys())[0]].shape == mr.shape)
                
                longest_diameters = { roi: mask_with_longest_diameter[1] for roi, mask_with_longest_diameter in masks_with_longest_diameter.items() }
                
                mrs, rtds = self.__mask_and_crop__(mr, rtd, masks)
                
                rois = masks.keys()
                
                labels = self.__get_labels__(rois, subject_id, course)
                clinical_data = self.__get_clinical_data__(rois, subject_id, course, longest_diameters)
                
                self.__append_data__(subject_id, mrs, rtds, clinical_data, labels)
                
                # if debug:
                #     self.plot(self.global_data, ['mr', 'rtd'], -1)
                    
            print(f'\rStep: {cnt+1}/{total_subjects}', end='')
        
        self.__clean_clinical_data__()
        
        self.__save__()
        
        print('\nData has been read successfully!')
    
    def __adjust_dataframes__(self):
        rawdata_meta_renaming = {'Study Date':'study_date','Study UID':'study_uid','Subject ID':'subject_id', 'Modality':'modality', 'File Location':'file_path'}
        clinical_data_ll_renaming = {'unique_pt_id': 'subject_id', 'Treatment Course':'course', 'Lesion Location':'roi', 'mri_type':'label', 'duration_tx_to_imag (months)': 'duration_tx_to_imag', 'Fractions':'fractions', "Lesion #": "lesion_number", "Lesion Name in NRRD files":"nrrd_filename"}
        clinical_data_cl_renaming = {'unique_pt_id': 'subject_id', 'Course #':'course', 'Diagnosis (Only want Mets)':'mets_diagnosis', 'Primary Diagnosis':'primary_diagnosis', 'Age at Diagnosis':'age', 'Gender':'gender'}
        
        rawdata_meta_types = {'study_date':"datetime64[ns]" ,'study_uid':'string','subject_id':'string', 'modality':'string', 'file_path':'string'}
        clinical_data_ll_types = {'subject_id':'int64','course':'Int8', 'roi':'string', 'label':'string', 'duration_tx_to_imag':'int8', 'fractions':'int8', "lesion_number": "int8", "nrrd_filename": "string"}
        clinical_data_cl_types = {'subject_id':'int64', 'course':'Int8', 'mets_diagnosis':'string', 'primary_diagnosis':'string', 'age':'int8', 'gender':'string'}
        
        self.rawdata_meta = self.rawdata_meta.drop(self.rawdata_meta.columns.difference(rawdata_meta_renaming.keys()), axis=1).rename(columns=rawdata_meta_renaming)
        self.clinical_data_ll = self.clinical_data_ll.drop(self.clinical_data_ll.columns.difference(clinical_data_ll_renaming.keys()), axis=1).rename(columns=clinical_data_ll_renaming)
        self.clinical_data_cl = self.clinical_data_cl.drop(self.clinical_data_cl.columns.difference(clinical_data_cl_renaming.keys()), axis=1).rename(columns=clinical_data_cl_renaming)
        
        self.rawdata_meta = self.rawdata_meta.astype(rawdata_meta_types)
        self.clinical_data_ll = self.clinical_data_ll.astype(clinical_data_ll_types)
        self.clinical_data_cl = self.clinical_data_cl.astype(clinical_data_cl_types)
        
        self.clinical_data = pd.merge_ordered(self.clinical_data_ll, self.clinical_data_cl, on=['subject_id', 'course'], how='inner')

    def __clean_output_directory__(self, path_dir):
        for filename in os.listdir(path_dir):
            file_path = os.path.join(path_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def __get_mr_rtd_rts_path__(self, values):
        path_MR = os.path.join(self.ORIGIN_DATA_FOLDER_PATH, values.loc[values['modality'] == 'MR', 'file_path'].iloc[0])
        path_RTD = os.path.join(self.ORIGIN_DATA_FOLDER_PATH, values.loc[values['modality'] == 'RTDOSE', 'file_path'].iloc[0])
        path_RTS = os.path.join(self.ORIGIN_DATA_FOLDER_PATH, values.loc[values['modality'] == 'RTSTRUCT', 'file_path'].iloc[0])
        
        return path_MR, path_RTD, path_RTS

    def resample_image_to_reference(self, image, reference_image):
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(reference_image.GetSize())
        resampler.SetOutputSpacing(reference_image.GetSpacing())
        resampler.SetOutputOrigin(reference_image.GetOrigin())
        resampler.SetOutputDirection(reference_image.GetDirection())
        resampler.SetTransform(sitk.Transform())  # Identity transform
        resampler.SetInterpolator(sitk.sitkLinear)  # Use linear interpolation for continuous data

        # Execute resampling
        resampled_image = resampler.Execute(image)
        return resampled_image

    def __get_mr_rtd_resampled__(self, path_MR, path_RTD):
        mr_dicom = self.__read_dicom__(path_MR)
        rtd_dicom = self.__read_dicom__(path_RTD)
        
        rtd_dicom_resampled = self.resample_image_to_reference(rtd_dicom, mr_dicom)
        
        mr_np = sitk.GetArrayFromImage(mr_dicom)
        rtd_np = sitk.GetArrayFromImage(rtd_dicom_resampled)
        
        assert(rtd_np.shape == mr_np.shape)
        
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

    def __clean_roi_name__(self, roi:str):
        roi = roi.lower()
                
        roi = roi.replace("l ", "lt")
        key = roi.replace("left ", "lt")
        roi = roi.replace("r ", "rt")
        roi = roi.replace("right ", "rt")
        
        roi = roi.replace(" ", "")
        
        return roi

    def __get_rts__(self, path_RTS, series_path, subject_id, course):
        rt_struct_path = [os.path.join(path_RTS, f) for f in os.listdir(path_RTS) if f.endswith('.dcm')][0]
        rtstruct = RTStructBuilder.create_from(dicom_series_path=series_path, rt_struct_path=rt_struct_path, )
    
        # lesion_location_with_lesion_id = self.clinical_data.loc[(self.clinical_data['subject_id'] == subject_id) & (self.clinical_data['course'] == course), ['roi', 'lesion_number']].values
        
        # id_rois_cd = { i+1:r[0] for i, r in enumerate(lesion_location_with_lesion_id) }
        # id_roi_rts = { int(structure.ROINumber):structure.ROIName for structure in rtstruct.ds.StructureSetROISequence}
        
        # map_log = { id_rois_cd[k]: id_roi_rts[k] for k, v in id_rois_cd.items() }
        
        lesion_location_with_nrrdfilename = self.clinical_data.loc[(self.clinical_data['subject_id'] == subject_id) & (self.clinical_data['course'] == course), ['roi', 'nrrd_filename']].values    

        pattern = r'(GK\.\d{3}_\d_L)'

        roinrrd_roicd = { re.split(pattern, e[1])[-1] :e[0] for e in lesion_location_with_nrrdfilename }
        roi_rts = [ roi_rts for roi_rts in rtstruct.get_roi_names() if "*" not in roi_rts]
        roinrrd = list(roinrrd_roicd.keys())
        
        mapping = couple_roi_names(roinrrd, roi_rts)
        
        print()
        print(f"subject_id: {subject_id}, course: {course}")
        pprint.pprint(mapping)
        
        masks = {}
        for nrrd_roi, roi_rts in mapping.items():
            
            mask_3d = rtstruct.get_roi_mask_by_name(roi_rts)
                
            mask_3d = mask_3d * 1
            mask_3d = np.swapaxes(mask_3d, 0, 2)
            mask_3d = np.swapaxes(mask_3d, 1, 2)
            masks[roinrrd_roicd[nrrd_roi]] = (mask_3d, self.__longest_diameter__(mask_3d))
                
        return masks

    def __get_labels__(self, rois, subject_id, course):
        to_return = {}
        
        for roi in rois:    
            label = self.clinical_data.loc[(self.clinical_data['subject_id']==subject_id)&(self.clinical_data['course']==course)&(self.clinical_data['roi']==roi), ['label']].values[0][0]
            to_return[roi] = label
            
        return to_return

    def __get_clinical_data__(self, rois, subject_id, course, longest_diameters):
        to_return = {}
        
        for roi in rois:
            clinical_data_row = self.clinical_data.loc[(self.clinical_data['subject_id']==subject_id)&(self.clinical_data['course']==course)&(self.clinical_data['roi']==roi), ['mets_diagnosis', 'primary_diagnosis', 'age', 'gender', 'roi', 'fractions']].values[0]
            clinical_data_row = np.append(clinical_data_row, longest_diameters[roi])
            clinical_data_row = np.append(clinical_data_row, self.clinical_data.groupby(['subject_id', 'course']).size().get((subject_id, course), 0))
            to_return[roi] = clinical_data_row
            
        return to_return

    def __mask_and_crop__(self, mr, rtd, masks):
        mr_return, rtd_return = {}, {}
        
        for roi in masks.keys():            
            mr_masked_cropped = self.__crop_les__(mr, masks[roi])
            rtd_masked_cropped = self.__crop_les__(rtd, masks[roi])
            
            mr_return[roi] = mr_masked_cropped
            rtd_return[roi] = rtd_masked_cropped
        
        return mr_return, rtd_return
    
    def __crop_les__(self, image, mask):
        
        true_points = np.argwhere(mask)
        
        top_left = true_points.min(axis=0)
        
        bottom_right = true_points.max(axis=0)
        
        cropped_arr = image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]
        
        return cropped_arr

    def __append_data__(self, subject_id, mrs, rtds, clinical_data, labels):
        for roi in mrs.keys():
            self.global_data['subject_id'].append(subject_id)
            self.global_data['mr'].append(np.float64(mrs[roi]))
            self.global_data['rtd'].append(np.float64(rtds[roi]))
            self.global_data['clinical_data'].append(clinical_data[roi])
            self.global_data['label'].append(labels[roi]) 
    
    def __clean_clinical_data__(self):
        for i, value in enumerate(self.global_data['clinical_data']):
            self.global_data['clinical_data'][i][0] = process_mets(self.global_data['clinical_data'][i][0])
            self.global_data['clinical_data'][i][1] = process_prim(self.global_data['clinical_data'][i][1])
            self.global_data['clinical_data'][i][3] = self.global_data['clinical_data'][i][3].lower().strip()
            self.global_data['clinical_data'][i][4] = process_roi(self.global_data['clinical_data'][i][4])
        
    def __save__(self):
        print('\nSaving data...')
        
        target = self.OUTPUT_PROCESSED_DATA_FOLDER_PATH
    
        with open(os.path.join(target, 'global_data.pkl'), 'wb') as f:
            pickle.dump(self.global_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', dest='DEBUG')
    args = parser.parse_args()
    
    dr = RawData_Reader()
    dr.full_run(debug=args.DEBUG)
