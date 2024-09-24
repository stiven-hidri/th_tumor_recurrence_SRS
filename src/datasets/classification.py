import os
import pickle
from torch.utils.data import Dataset

class ClassifierDatasetSplit(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        mr = self.data['mr'][idx]
        rtd = self.data['rtd'][idx]
        clinic_data = self.data['clinic_data'][idx]
        label = self.data['label'][idx]
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
        return ClassifierDatasetSplit(self.train), ClassifierDatasetSplit(self.test), ClassifierDatasetSplit(self.val)
