import os
import pickle
from torch.utils.data import Dataset
from utils.augment import combine_aug
from sklearn.preprocessing import MinMaxScaler


class ClassifierDatasetSplit(Dataset):
    def __init__(self, data: dict, split_name: str):
        self.data = data
        self.split_name = split_name

    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        mr = self.data['mr'][idx]
        rtd = self.data['rtd'][idx]
        clinic_data = self.data['clinic_data'][idx]
        label = self.data['label'][idx]
        
        if int(label) == 1 and self.split_name == 'train':
            mr, rtd = combine_aug(mr, rtd)
            
            mr = ( mr - mr.min() ) / (mr.max() - mr.min())
            rtd = ( rtd - rtd.min() ) / (rtd.max() - rtd.min())
        
        return mr, rtd, clinic_data, label

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class ClassifierDataset(Dataset):
    def __init__(self):
        super().__init__()
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
        return ClassifierDatasetSplit(self.train, 'train'), ClassifierDatasetSplit(self.test, 'test'), ClassifierDatasetSplit(self.val, 'val')
