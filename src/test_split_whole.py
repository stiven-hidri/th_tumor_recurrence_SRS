from itertools import product
import pprint
import re
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader
from utils import Parser
from datasets import ClassifierDataset
from modules import ClassificationModule
import os
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import StratifiedGroupKFold
from params_grid import *
import numpy as np
from collections import defaultdict, Counter
from argparse import ArgumentParser
from sklearn.metrics import f1_score
    
if __name__ == '__main__':
    # Get configuration arguments
    parser = Parser()

    config, device = parser.parse_args()
    
    cnt = 0
    version = 0
    results_list = []
    
    # Choose parameter grid based on model name
    if config.model.name == "base_model" or 'enhanced' in config.model.name:
        param_grid = param_grid_basemodel
    elif config.model.name == "conv_lstm":
        param_grid = param_grid_convlstm
    elif config.model.name == "mlp_cd":
        param_grid = param_grid_mlpcd
    elif config.model.name == "wdt_conv":
        param_grid = param_grid_wdt
    elif config.model.name == "trans_med":
        param_grid = param_grid_transmed
    else:
        raise NotImplementedError("Model not implemented")
    
    k = config.logger.k
    majority_vote = config.logger.majority_vote
    
    # Load sources and create datasets
    classifier_dataset = ClassifierDataset(model_name=config.model.name)
    p_augmentation=0.
    batch_size=0
    
    torch.manual_seed(42)

    test_labels = None
    
    test_predictions = {
        'label_predicted': [],
        'true_labels': [],
        'predictions': []
    }
    
    n_splits = k
    outer_cv = StratifiedGroupKFold(n_splits=n_splits)
    
    print('Creating splits...')
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(classifier_dataset.global_data['mr'], classifier_dataset.global_data['label'], classifier_dataset.global_data['subject_id'])):
        
        train_set, val_set, test_set = classifier_dataset.create_split_whole_dataset(train_idx, test_idx)
        
        statisticss = {
            "train":[0, 0], 
            "val":  [0, 0],
            "test": [0, 0]
            }
        
        for key in statisticss.keys():
            
            if key == "train":
                dataset = train_set.data
            elif key == "val":
                dataset = val_set.data
            elif key == "test":
                dataset = test_set.data
            
            for j in range(len(dataset['label'])):
                statisticss[key][int(dataset['label'][j])]+=1
        
        pprint.pprint(f"fold: {fold}\t{statisticss}")