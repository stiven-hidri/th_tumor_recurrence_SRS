from copy import copy, deepcopy
import os
import pickle
import pywt
from torch.utils.data import Dataset
from utils.augment import combine_aug
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
from sklearn import preprocessing
import torch.nn.functional as F
from scipy.ndimage import zoom

class ClassifierDatasetSplit(Dataset):
    def __init__(self, model_name:str, data:dict, split_name:str):
        self.data = data
        self.model_name = model_name
        self.split_name = split_name
        self.p_augmentation = None
        self.augmentation_techniques = None

    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        clinical_data = self.data['clinical_data'][idx]
        label = self.data['label'][idx]
        
        if 'wdt' in self.model_name:
            mr_rtd_fusion = self.data['mr_rtd_fusion'][idx]
            
            if self.split_name == 'train':
                mr_rtd_fusion, _ = combine_aug(mr_rtd_fusion, None, p_augmentation=self.p_augmentation, augmentations_techinques=self.augmentation_techniques)
                
            return mr_rtd_fusion, clinical_data, label
        else:
            mr = self.data['mr'][idx]
            rtd = self.data['rtd'][idx]
            
            if self.split_name == 'train':
                mr, rtd = combine_aug(mr, rtd, p_augmentation=self.p_augmentation, augmentations_techinques=self.augmentation_techniques)
                
            return mr, rtd, clinical_data, label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class ClassifierDataset(Dataset):
    def __init__(self, model_name:str):
        super().__init__()
        self.model_name = model_name
        
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
        
        self.CLINIC_FEATURES_TO_DISCRETIZE = [0, 1, 3, 4]
        self.CLINIC_FEATURES_TO_NORMALIZE = [2, 5, 6, 7]
        # 0 = mets_diagnosis, 1 = primary_diagnosis, 2 = age, 3 = gender, 4 = roi, 5 = fractions, 6 = longest_diameter, 7 = number_of_lesions
        self.CLINIC_FEATURES_TO_KEEP = [0, 1, 2, 3, 4, 6, 7]
        
        self.ORIGINAL_TEST_SET = [ 427, 243, 257, 224, 420, 312, 316, 199, 219, 492, 332, 364, 132 ]
        
        self.global_data = None
        self.max_values = None
        
        self.__load__()

    def __len__(self):
        return len(self.train['label'])

    def __wdt_fusion__(self, mr, rtd):
        coeffs_mr = pywt.dwtn(mr, 'db1', axes=(0, 1, 2))
        coeffs_rtd = pywt.dwtn(rtd, 'db1', axes=(0, 1, 2))
            
        # fused_details_avg = {key: (coeffs_mr[key]*.6 + coeffs_rtd[key]*.4) for key in coeffs_mr.keys()}

        fused_details_e = {}
        for key in coeffs_mr.keys():
            if key == 'aaa':  # Skip approximation coefficients for energy fusion
                fused_details_e[key] = (coeffs_mr[key]*.55 + coeffs_rtd[key]*.45)
            else:
                energy1 = np.abs(coeffs_mr[key]) ** 2
                energy2 = np.abs(coeffs_rtd[key]) ** 2
                fused_details_e[key] = np.where(energy1 > energy2, coeffs_mr[key], coeffs_rtd[key])

        fused_image_e = pywt.idwtn(fused_details_e, 'db1', axes=(0, 1, 2))
        
        fused_image_e = torch.Tensor(fused_image_e).to(torch.float32) * ( mr > 0 )
        return fused_image_e

    def __augment_by_flipping__(self, split):
        print('Augmenting data...', end='\r')
        i = 0
        total_len = len(split['mr'])
        while i < total_len:
            if int(split['label'][i]) == 1:
                mr, rtd, mr_rtd_fusion = split['mr'][i], split['rtd'][i], split['mr_rtd_fusion'][i]
                
                augmented_mr = [ torch.flip(mr, dims=[0]), torch.flip(mr, dims=[1]), torch.flip(mr, dims=[2]) ] 
                augmented_rtd = [ torch.flip(rtd, dims=[0]), torch.flip(rtd, dims=[1]), torch.flip(rtd, dims=[2]) ] 
                augmented_mr_rtd_fusion = [torch.flip(mr_rtd_fusion, dims=[0]), torch.flip(mr_rtd_fusion, dims=[1]), torch.flip(mr_rtd_fusion, dims=[2])]
                
                augmemted_subject_id = [split['subject_id'][i]] * len(augmented_mr)
                augmented_label = [split['label'][i]] * len(augmented_mr)
                augmented_clinical_data = [split['clinical_data'][i]] * len(augmented_mr)
                
                split['mr'].extend(augmented_mr)
                split['rtd'].extend(augmented_rtd)
                split['mr_rtd_fusion'].extend(augmented_mr_rtd_fusion)
                
                split['subject_id'].extend(augmemted_subject_id)
                split['clinical_data'].extend(augmented_clinical_data)
                split['label'].extend(augmented_label)
                

            
            i+=1
            
        return split
        
    def __one_hot__(self, split, max_values):        
        for i_tensor in range(len(split['clinical_data'])):
            new_tensor = [split['clinical_data'][i_tensor][j] for j in self.CLINIC_FEATURES_TO_NORMALIZE if j in self.CLINIC_FEATURES_TO_KEEP]                
            for j in self.CLINIC_FEATURES_TO_DISCRETIZE:
                if j in self.CLINIC_FEATURES_TO_KEEP:
                    new_tensor.append(F.one_hot(split['clinical_data'][i_tensor][j].long(), num_classes=max_values[j]+1).float()[0])
            split['clinical_data'][i_tensor] = torch.cat(new_tensor)
            
        return split
    
    def __discretize_categorical_features__(self, global_data):
        
        global_data['label'] = np.array([1 if label == 'recurrence' else 0 for label in global_data['label']])
        
        tmp = np.array(global_data['clinical_data'])
        
        for j in self.CLINIC_FEATURES_TO_DISCRETIZE:
            tmp[:,j] = preprocessing.LabelEncoder().fit_transform(tmp[:,j])
        
        global_data['clinical_data'] = tmp.astype(np.float32)
        
        return global_data
        
    def __minmax_scaling__(self, sample, statistics):
        sample = (sample - statistics['min']) / (statistics['max'] - statistics['min'])
        return sample
        
    def __z_norm__(self, sample, statistics):
        sample = (sample - statistics['mean']) / statistics['std']
        return sample
    
    def __normalize__(self, split, statistics, f):
        for key in split.keys():
            if key != 'label' and key != 'mr_rtd_fusion' and key != 'subject_id':
                if key == 'clinical_data':
                    for j in statistics[key].keys():
                        for i_tensor in range(len(split[key])):
                            split[key][i_tensor][j] = f(split[key][i_tensor][j], statistics[key][j])
                else:
                    for i_tensor in range(len(split[key])):
                        split[key][i_tensor] = f(split[key][i_tensor], statistics[key])
                        
        return split
    
    def __compute_statistics__(self, data):
        statistics = {}
                
        for key in data.keys():
            if key != 'label' and key != 'mr_rtd_fusion' and key != 'subject_id':
                current_data = torch.cat([tensor.view(-1) for tensor in data[key]])
                
                if key == 'clinical_data':
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

    def __average_statistics__(self, statistics_list):
        averaged_statistics = {}

        # Iterate over the statistics of the first entry to initialize the averaged_statistics
        for key in statistics_list[0].keys():
            if key not in averaged_statistics:
                averaged_statistics[key] = {}
                
                if key == 'clinical_data':
                    for j in statistics_list[0][key].keys():
                        averaged_statistics[key][j] = {}
                        averaged_statistics[key][j] = {
                            'min': 0,
                            'max': 0,
                            'mean': 0,
                            'std': 0
                        }
                else:
                    averaged_statistics[key] = {
                        'min': 0,
                        'max': 0,
                        'mean': 0,
                        'std': 0
                    }
        
        # Sum up the statistics across all entries in the statistics_list
        for stats in statistics_list:
            for key in stats.keys():
                if key == 'clinical_data':
                    for j in stats[key].keys():
                        averaged_statistics[key][j]['min'] += stats[key][j]['min']
                        averaged_statistics[key][j]['max'] += stats[key][j]['max']
                        averaged_statistics[key][j]['mean'] += stats[key][j]['mean']
                        averaged_statistics[key][j]['std'] += stats[key][j]['std']
                else:
                    averaged_statistics[key]['min'] += stats[key]['min']
                    averaged_statistics[key]['max'] += stats[key]['max']
                    averaged_statistics[key]['mean'] += stats[key]['mean']
                    averaged_statistics[key]['std'] += stats[key]['std']
        
        num_statistics = len(statistics_list)
        
        for key in averaged_statistics.keys():
            if key == 'clinical_data':
                for j in averaged_statistics[key].keys():    
                    averaged_statistics[key][j]['min'] /= num_statistics
                    averaged_statistics[key][j]['max'] /= num_statistics
                    averaged_statistics[key][j]['mean'] /= num_statistics
                    averaged_statistics[key][j]['std'] /= num_statistics
            else:
                averaged_statistics[key]['min'] /= num_statistics
                averaged_statistics[key]['max'] /= num_statistics
                averaged_statistics[key]['mean'] /= num_statistics
                averaged_statistics[key]['std'] /= num_statistics

        return averaged_statistics

    def __get_max_values__(self, clinical_data):
        max_values = {}
        for tensor in clinical_data:
            for j, feature in enumerate(tensor):
                if j in self.CLINIC_FEATURES_TO_DISCRETIZE:
                    if j not in max_values:
                        max_values[j] = int(feature)
                    else:
                        max_values[j] = max(max_values[j], int(feature))
                        
        return max_values

    def __pad_resize_images__(self, global_data, desired_shape=(42, 42, 42)):        
        keys = ['mr', 'rtd']
        for i in range(len(global_data['mr'])):
            for k in keys:
                image = global_data[k][i]
                
                cropped_shape = np.array(image.shape)

                factor = np.min(np.divide(desired_shape,cropped_shape))
                
                if factor < 1:
                    image = zoom(image, (factor,factor,factor), order=1)
                
                cropped_shape = np.array(image.shape)
                
                pad_before = np.maximum((desired_shape - cropped_shape) // 2, 0)
                pad_after = np.maximum(desired_shape - cropped_shape - pad_before, 0)
                
                image = np.pad(image, (
                    (pad_before[0], pad_after[0]), 
                    (pad_before[1], pad_after[1]), 
                    (pad_before[2], pad_after[2])
                ), mode='constant', constant_values=0)
                
                global_data[k][i] = image
                
        return global_data

    def __load__(self) -> None:
        with open(os.path.join(self.DATA_PATH, 'global_data.pkl'), 'rb') as f:
            self.global_data = pickle.load(f)
            
        self.global_data = self.__pad_resize_images__(self.global_data)
        self.global_data = self.__discretize_categorical_features__(self.global_data)
        self.max_values = self.__get_max_values__(self.global_data['clinical_data'])
        
        self.global_data['mr'] = [torch.Tensor(e).to(torch.float32) for e in self.global_data['mr']]
        self.global_data['rtd'] = [torch.Tensor(e).to(torch.float32) for e in self.global_data['rtd']]
        self.global_data['clinical_data'] = [torch.Tensor(e).to(torch.float32).view(-1, 1) for e in self.global_data['clinical_data']]
        
        print('Dataset is loaded')
    
    def return_data_dictionary(self, mr, rtd, clinical_data, labels, subjects, idx, mr_rtd_fusion=None, tensor_conversion = True):
        if tensor_conversion:
            data_dictionary = {
                "mr": [torch.Tensor(mr[i]).to(torch.float32) for i in idx],
                "rtd": [torch.Tensor(rtd[i]).to(torch.float32) for i in idx],
                "clinical_data": [torch.Tensor(clinical_data[i]).to(torch.float32).view(-1, 1) for i in idx],
                'label': torch.tensor([labels[i] for i in idx]).to(torch.float32).view(-1, 1),
                'subject_id': [subjects[i] for i in idx]
            }
        else:
            data_dictionary = {
                "mr": [mr[i] for i in idx],
                "rtd": [rtd[i] for i in idx],
                "clinical_data": [clinical_data[i] for i in idx],
                'label': [labels[i] for i in idx ],
                'subject_id': [subjects[i] for i in idx]
            }
        
        return data_dictionary
    
    def detach_test_set(self):
        test_idx = [i for i in range(len(self.global_data['mr'])) if self.global_data['subject_id'][i] in self.ORIGINAL_TEST_SET]
        
        self.test_set_tmp = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], test_idx)
        
        self.global_data['mr'] = [ self.global_data['mr'][idx] for idx in range(len(self.global_data['mr'])) if idx not in test_idx]
        self.global_data['rtd'] = [ self.global_data['rtd'][idx] for idx in range(len(self.global_data['rtd'])) if idx not in test_idx]
        self.global_data['label'] = [ self.global_data['label'][idx] for idx in range(len(self.global_data['label'])) if idx not in test_idx]
        self.global_data['subject_id'] = [ self.global_data['subject_id'][idx] for idx in range(len(self.global_data['subject_id'])) if idx not in test_idx]
        self.global_data['clinical_data'] = [ self.global_data['clinical_data'][idx] for idx in range(len(self.global_data['clinical_data'])) if idx not in test_idx]
    
    def __generate_mrrtdfusion__(self, split):
        split['mr_rtd_fusion'] =  [self.__wdt_fusion__(split['mr'][i], split['rtd'][i]) for i in split['mr']],
        return split
        
    def create_split_keep_test(self, train_idx, val_idx):
        
        train_set = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], train_idx)
        val_set = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], val_idx)
        test_set = deepcopy(self.test_set_tmp)
        
        statistics = self.__compute_statistics__(train_set)
        
        train_set = self.__normalize__(train_set, statistics, f=self.__minmax_scaling__)
        val_set = self.__normalize__(val_set, statistics, f=self.__minmax_scaling__)
        test_set = self.__normalize__(test_set, statistics, f=self.__minmax_scaling__)
        
        train_set = self.__generate_mrrtdfusion__(train_set)
        val_set = self.__generate_mrrtdfusion__(val_set)
        test_set = self.__generate_mrrtdfusion__(test_set)
        
        train_set = self.__one_hot__(train_set, self.max_values)
        val_set = self.__one_hot__(val_set, self.max_values)
        test_set = self.__one_hot__(test_set, self.max_values)
        
        # train_set = self.__augment_by_flipping__(train_set)
        
        return ClassifierDatasetSplit(model_name=self.model_name, data=train_set, split_name="train"), ClassifierDatasetSplit(model_name=self.model_name, data=val_set, split_name="val"), ClassifierDatasetSplit(model_name=self.model_name, data=test_set, split_name="test")
    
    def create_split_whole_dataset(self, train_idx, test_idx) -> list[list[ClassifierDatasetSplit]]:    
/*************  ✨ Codeium Command ⭐  *************/
/******  7fc196d6-c4fc-4cf6-b711-9f69d4fbc46e  *******/
        inner_cv = StratifiedGroupKFold(n_splits=6)
            
        train_outer = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], train_idx)
        test_set = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], test_idx)

        # Use the first split as the train/validation division
        inner_train_idx, val_idx = next(inner_cv.split(train_outer['mr'], train_outer['label'], train_outer['subject_id']))
        
        val_set = self.return_data_dictionary(train_outer['mr'], train_outer['rtd'], train_outer['clinical_data'], train_outer['label'], train_outer['subject_id'], val_idx, mr_rtd_fusion=train_outer['mr_rtd_fusion'],tensor_conversion=False)
        train_set = self.return_data_dictionary(train_outer['mr'], train_outer['rtd'], train_outer['clinical_data'], train_outer['label'], train_outer['subject_id'], inner_train_idx, mr_rtd_fusion=train_outer['mr_rtd_fusion'], tensor_conversion=False)
        
        statistics = self.__compute_statistics__(train_set)
        
        train_set = self.__normalize__(train_set, statistics, f=self.__minmax_scaling__)
        val_set = self.__normalize__(val_set, statistics, f=self.__minmax_scaling__)
        test_set = self.__normalize__(test_set, statistics, f=self.__minmax_scaling__)
        
        train_set = self.__generate_mrrtdfusion__(train_set)
        val_set = self.__generate_mrrtdfusion__(val_set)
        test_set = self.__generate_mrrtdfusion__(test_set)
        
        train_set = self.__one_hot__(train_set, self.max_values)
        val_set = self.__one_hot__(val_set, self.max_values)
        test_set = self.__one_hot__(test_set, self.max_values)
        
        train_set = self.__augment_by_flipping__(train_set)
        
        return ClassifierDatasetSplit(model_name=self.model_name, data=train_set, split_name="train"), ClassifierDatasetSplit(model_name=self.model_name, data=val_set, split_name="val"), ClassifierDatasetSplit(model_name=self.model_name, data=test_set, split_name="test")

    def create_split_static(self):
        
        print('Creating splits...')
        
        train_size = 0.7
        val_size = 0.15
        test_size = 0.15
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_val_idx, test_idx = next(gss.split(self.global_data['mr'], self.global_data['label'], self.global_data['subject_id']))
        
        train_val = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], train_val_idx)
        test_set = self.return_data_dictionary(self.global_data['mr'], self.global_data['rtd'], self.global_data['clinical_data'], self.global_data['label'], self.global_data['subject_id'], test_idx)
        
        val_relative_size = val_size / (train_size + val_size)
        
        gss_inner = GroupShuffleSplit(n_splits=1, test_size=val_relative_size, random_state=42)
        train_idx, val_idx = next(gss_inner.split(train_val['mr'], train_val['label'], train_val['subject_id']))
        
        train_set = self.return_data_dictionary(train_val['mr'], train_val['rtd'], train_val['clinical_data'], train_val['label'], train_val['subject_id'], train_idx, mr_rtd_fusion=train_val['mr_rtd_fusion'], tensor_conversion=False)
        val_set = self.return_data_dictionary(train_val['mr'], train_val['rtd'], train_val['clinical_data'], train_val['label'], train_val['subject_id'], val_idx, mr_rtd_fusion=train_val['mr_rtd_fusion'], tensor_conversion=False)
        
        statistics = self.__compute_statistics__(train_set)
        
        train_set = self.__normalize__(train_set, statistics, f=self.__minmax_scaling__)
        val_set = self.__normalize__(val_set, statistics, f=self.__minmax_scaling__)
        test_set = self.__normalize__(test_set, statistics, f=self.__minmax_scaling__)
        
        train_set = self.__one_hot__(train_set, self.max_values)
        val_set = self.__one_hot__(val_set, self.max_values)
        test_set = self.__one_hot__(test_set, self.max_values)
        
        train_set = self.__augment_by_flipping__(train_set)
        
        return ClassifierDatasetSplit(model_name=self.model_name, data=train_set, split_name="train"), ClassifierDatasetSplit(model_name=self.model_name, data=val_set, split_name="val"), ClassifierDatasetSplit(model_name=self.model_name, data=test_set, split_name="test")