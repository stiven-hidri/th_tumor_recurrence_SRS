import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from enum import Enum

class ClassifierDatasetSplit(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Define transformations 
        #TODO:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        sample = self.data.iloc[idx, :]
        image = Image.open(sample.image_path).convert('L')
        image = transform(image)
        label = sample.label
        return (image, label)

    def __iter__(self):
        for sample in self.data:
            yield sample

class ClassifierDataset(Dataset):
    def __init__(self, source):
        super().__init__()
        self.data = self.__load__(source)

    def __len__(self):
        return len(self.data)

    def __load__(self, source) -> None:
        data_defect = list()
        data_no_defect = list()
        synthetized_defect_images = list()
        
        #TODO: todo

        return pd.DataFrame(data)

    def create_splits(self, splits_proportion: List[float]) -> Tuple[ClassifierDatasetSplit]:
        # TODO: todo
        # assert sum(splits_proportion) == 1, 'The proportions of the splits must be sum up to 1'

        # # Calculate the number of rows for each split based on proportions
        # split_sizes = [int(prop * len(self)) for prop in splits_proportion]
        
        # # Shuffle the DataFrame
        # df = self.data.sample(frac=1).reset_index(drop=True)
        
        # # Split the DataFrame into chunks based on proportions
        # splits = list()
        # start_idx = 0
        # for size in split_sizes:
        #     splits.append(ClassifierDatasetSplit(df.iloc[start_idx:start_idx + size]))
        #     start_idx += size
    
        return tuple(splits)    
