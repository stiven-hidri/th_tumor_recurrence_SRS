import os
import pickle
from torch.utils.data import Dataset
from utils.augment import combine_aug
from sklearn.preprocessing import MinMaxScaler


class ClassifierDatasetSplit(Dataset):
    def __init__(self, data: dict, split_name: str, p_augmentation=.0, augmentation_techniques=[]):
        self.data = data
        self.split_name = split_name
        self.p_augmentation = p_augmentation
        self.augmentation_techniques = augmentation_techniques
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
        with open(os.path.join(self.DATA_PATH, f'statistics.pkl'), 'rb') as f:
            self.statistics = pickle.load(f)

    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        mr = self.data['mr'][idx]
        rtd = self.data['rtd'][idx]
        clinic_data = self.data['clinic_data'][idx]
        label = self.data['label'][idx]
        
        if self.split_name == 'train':
            # mr, rtd = combine_aug(mr, rtd, p_augmentation=self.p_augmentation if int(label) == 0 else 1 - self.p_augmentation, augmentations_techinques=self.augmentation_techniques)
            mr, rtd = combine_aug(mr, rtd, p_augmentation=self.p_augmentation, augmentations_techinques=self.augmentation_techniques)
            # mr_min = mr.min()            
            # mr_max = mr.max()
            # rtd_min = rtd.min()
            # rtd_max = rtd.max()
            
            # mr = (mr - mr_min) / (mr_max - mr_min)
            # rtd = (rtd - rtd_min) / (rtd_max - rtd_min)
        
        return mr, rtd, clinic_data, label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class ClassifierDataset(Dataset):
    def __init__(self, p_augmentation=.3, augmentation_techniques=['shear', 'gaussian_noise', 'flip', 'rotate', 'brightness']):
        super().__init__()
        self.p_augmentation = p_augmentation
        self.augmentation_techniques = augmentation_techniques
        self.DATA_PATH = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'processed')
        self.train, self.test, self.val = self.__load__()

    def __len__(self):
        return len(self.train['label'])

    def __load__(self) -> None:
        with open(os.path.join(self.DATA_PATH, 'train_set.pkl'), 'rb') as f:
            train = pickle.load(f)
            
        with open(os.path.join(self.DATA_PATH, 'test_set.pkl'), 'rb') as f:
            test = pickle.load(f)
            
        with open(os.path.join(self.DATA_PATH, 'val_set.pkl'), 'rb') as f:
            val = pickle.load(f)

        return train, test, val
    
    def create_splits(self):
        return ClassifierDatasetSplit(self.train, 'train', p_augmentation=self.p_augmentation, augmentation_techniques=self.augmentation_techniques), ClassifierDatasetSplit(self.test, 'test'), ClassifierDatasetSplit(self.val, 'val')
