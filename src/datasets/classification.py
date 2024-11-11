import os
import pickle
import pywt
from torch.utils.data import Dataset
from utils.augment import combine_aug
from torchvision import transforms
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import torch
from sklearn import preprocessing
import torch.nn.functional as F
from scipy.ndimage import zoom

class ClassifierDatasetSplit(Dataset):
    def __init__(self, model_name:str, data:dict, split_name:str, p_augmentation:float=None, augmentation_techniques:list=None):
        self.data = data
        self.model_name = model_name
        self.split_name = split_name
        self.p_augmentation = p_augmentation
        self.augmentation_techniques = augmentation_techniques

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
    def __init__(self, model_name:str, keep_test:bool, k:int):
        super().__init__()
        self.model_name = model_name
        self.k = k
        self.keep_test = keep_test
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
        self.CLINIC_FEATURES_TO_DISCRETIZE = [0, 1, 3, 4]
        self.CLINIC_FEATURES_TO_NORMALIZE = [2, 5, 6, 7]
        self.CLINIC_FEATURES_TO_KEEP = [0, 1, 2, 3, 4, 6, 7]
        self.list_train, self.list_val, self.test_set = self.__load__()

    def __len__(self):
        return len(self.train['label'])

    def __wdt_fusion__(self, mr, rtd):
        coeffs_mr = pywt.dwtn(mr, 'db1', axes=(0, 1, 2))
        coeffs_rtd = pywt.dwtn(rtd, 'db1', axes=(0, 1, 2))
            
        # fused_details_avg = {key: (coeffs_mr[key]*.6 + coeffs_rtd[key]*.4) for key in coeffs_mr.keys()}

        fused_details_e = {}
        for key in coeffs_mr.keys():
            if key == 'aaa':  # Skip approximation coefficients for energy fusion
                fused_details_e[key] = (coeffs_mr[key] + coeffs_rtd[key]) / 2
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

                augmented_label = [split['label'][i]] * len(augmented_mr)
                augmented_clinical_data = [split['clinical_data'][i]] * len(augmented_mr)
                
                split['mr'].extend(augmented_mr)
                split['rtd'].extend(augmented_rtd)
                split['mr_rtd_fusion'].extend(augmented_mr_rtd_fusion)
                
                split['clinical_data'].extend(augmented_clinical_data)
                split['label'] = torch.cat((split['label'], torch.tensor(augmented_label).to(torch.float32).view(-1, 1)), dim=0)
                

            
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
            if key != 'label' and key != 'mr_rtd_fusion':
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
            if key != 'label' and key != 'mr_rtd_fusion':
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

    def __pad_resize_images__(self, global_data, desired_shape=(90, 90, 90)):        
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
            global_data = pickle.load(f)
            
        if self.keep_test:
            subjects_test = [ 427, 243, 257, 224, 420, 312, 316, 199, 219, 492, 332, 364, 132 ]
        else: 
            subjects_test = []
            
        global_data = self.__pad_resize_images__(global_data)
            
        global_data = self.__discretize_categorical_features__(global_data)
        
        max_values = self.__get_max_values__(global_data['clinical_data'])
            
        subjects = global_data['subject_id']
        mr = [torch.Tensor(e).to(torch.float32) for e in global_data['mr']]
        rtd = [torch.Tensor(e).to(torch.float32) for e in global_data['rtd']]
        clinical_data = [torch.Tensor(e).to(torch.float32).view(-1, 1) for e in global_data['clinical_data']]
        labels = global_data['label']
            
        list_train, list_val = [], []
        test = None

        n_splits = self.k
        skf = StratifiedGroupKFold(n_splits=n_splits)
        
        if self.keep_test:
            test_idx = [i for i in range(len(mr)) if subjects[i] in subjects_test]
            test_set = {
                "mr": [torch.Tensor(mr[i]).to(torch.float32) for i in test_idx],
                "rtd": [torch.Tensor(rtd[i]).to(torch.float32) for i in test_idx],
                "mr_rtd_fusion": [self.__wdt_fusion__(mr[i], rtd[i]) for i in test_idx],
                "clinical_data": [torch.Tensor(clinical_data[i]).to(torch.float32).view(-1, 1) for i in test_idx],
                'label': torch.tensor([labels[i] for i in test_idx]).to(torch.float32).view(-1, 1),
            }
            
            mr = [ mr[idx] for idx in range(len(mr)) if idx not in test_idx]
            rtd = [ rtd[idx] for idx in range(len(rtd)) if idx not in test_idx]
            labels = [ labels[idx] for idx in range(len(labels)) if idx not in test_idx]
            subjects = [ subjects[idx] for idx in range(len(subjects)) if idx not in test_idx]
            clinical_data = [ clinical_data[idx] for idx in range(len(clinical_data)) if idx not in test_idx]
        
        all_statistics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(mr, labels, subjects)):
            
            train_set = {
                "mr": [torch.Tensor(mr[i]).to(torch.float32) for i in train_idx],
                "rtd": [torch.Tensor(rtd[i]).to(torch.float32) for i in train_idx],
                "mr_rtd_fusion": [self.__wdt_fusion__(mr[i], rtd[i]) for i in train_idx],
                "clinical_data": [torch.Tensor(clinical_data[i]).to(torch.float32).view(-1, 1) for i in train_idx],
                'label': torch.tensor([e for i, e in enumerate(labels) if i in train_idx]).to(torch.float32).view(-1, 1)
            }

            val_set = {
                "mr": [torch.Tensor(mr[i]).to(torch.float32) for i in val_idx],
                "rtd": [torch.Tensor(rtd[i]).to(torch.float32) for i in val_idx],
                "mr_rtd_fusion": [self.__wdt_fusion__(mr[i], rtd[i]) for i in val_idx],
                "clinical_data": [torch.Tensor(clinical_data[i]).to(torch.float32).view(-1, 1) for i in val_idx],
                'label': torch.tensor([labels[i] for i in val_idx]).to(torch.float32).view(-1, 1)
            }
            
            statistics = self.__compute_statistics__(train_set)
            all_statistics.append(statistics)
            
            train_set = self.__normalize__(train_set, statistics, f=self.__minmax_scaling__)
            val_set = self.__normalize__(val_set, statistics, f=self.__minmax_scaling__)
            
            train_set = self.__one_hot__(train_set, max_values)
            val_set = self.__one_hot__(val_set, max_values)
            
            # train_set = self.__augment_by_flipping__(train_set)
            
            list_train.append(train_set)
            list_val.append(val_set)
            
        if self.keep_test:
            averaged_statistics = self.__average_statistics__(all_statistics)
            test_set = self.__normalize__(test_set, averaged_statistics, f=self.__minmax_scaling__)
            test_set = self.__one_hot__(test_set, max_values)
            
        return list_train, list_val, test_set
    
    def create_splits(self, p_augmentation:float, augmentation_techniques:list):
        split = [ [], [], ClassifierDatasetSplit(model_name=self.model_name, data=self.test_set, split_name="test") ]
        
        for i in range (len(self.list_train)):
            split[0].append(ClassifierDatasetSplit(model_name=self.model_name, data=self.list_train[i], split_name='train', p_augmentation=p_augmentation, augmentation_techniques=augmentation_techniques))
            split[1].append(ClassifierDatasetSplit(model_name=self.model_name, data=self.list_val[i], split_name="val" if self.keep_test else "test"))
        
        return split
    
