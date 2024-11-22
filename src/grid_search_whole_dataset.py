from itertools import product
import pprint
import re
import shutil
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

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high') 
# torch.cuda.set_per_process_memory_fraction(fraction=.5, device=None)

model_parameters = [ 
    "name",
    "epochs",
    "batch_size",
    "optimizer",
    "scheduler",
    "rnn_type",
    "hidden_size",
    "num_layers",
    "lr",
    "weight_decay",
    "dropout",
    "lf",
    "use_clinical_data",
    "alpha_fl",
    "gamma_fl",
    "pos_weight",
    "augmentation_techniques",
    "p_augmentation",
    "depth_attention"
]

model_parameters_toshow = [ 
    "batch_size",
    "rnn_type",
    "hidden_size",
    "num_layers",
    "lr",
    "weight_decay",
    "dropout",
    "use_clinical_data",
    "alpha_fl",
    "gamma_fl",
    "p_augmentation",
    "depth_attention"
]

def delete_checkpoints(experiment_name, log_dir):
    experiment_path = os.path.join(log_dir, experiment_name)
    for dir_name in os.listdir(experiment_path):
        dir_path = os.path.join(experiment_path, dir_name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

def load_checkpoint(config, checkpoint_cb, fold, version):
    if checkpoint_cb.best_model_path:
        module = ClassificationModule.load_from_checkpoint(name=config.model.name, checkpoint_path=checkpoint_cb.best_model_path)
    else:
        folder_path = checkpoint_cb.dirpath
        
        pattern = re.compile(r"epoch=(\d+)_val_loss=([\d\.]+)\.ckpt")

        # Variables to keep track of the best checkpoint
        best_val_loss = float('inf')
        best_checkpoint = None

        # Iterate over the files in the folder
        for filename in os.listdir(folder_path):
            match = pattern.match(filename)
            if match:
                epoch = int(match.group(1))
                val_loss = float(match.group(2))
                
                # Check if current file has a lower validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint = filename
        
        path_to_best_checkpoint = os.path.join('log', config.logger.experiment_name, f'version_{version}_fold_{fold}', best_checkpoint)
        
        module = ClassificationModule.load_from_checkpoint(name=config.model.name, checkpoint_path=path_to_best_checkpoint)
        
    return module

def calculate_statistics(predicted_labels, true_labels, predictions):
    C = confusion_matrix(true_labels, predicted_labels)
    TP, TN, FP, FN = C[1,1], C[0,0], C[0,1], C[1,0]
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
    
    statistics = {
        "accuracy": (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
        "recall": recall,
        "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
        "precision": precision,
        "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        "pr_auc": auc(recall_curve, precision_curve),
        "roc_auc": roc_auc
    }
    
    return statistics
    
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
    
    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    all_param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Load sources and create datasets
    classifier_dataset = ClassifierDataset(model_name=config.model.name)
    
    for i, param_set in enumerate(all_param_combinations):
        print(f"**********\n{i+1}/{len(all_param_combinations)}:\n**********")
        
        # contains trian and val folds and eventually test set
        if "p_augmentation" in param_set.keys():
            p_augmentation = param_set["p_augmentation"]
        else:
            p_augmentation=0.
        
        batch_size = param_set['batch_size']
    
        del param_set['batch_size']
        
        # torch.manual_seed(40)
        
        # Initialize list to collect test predictions for majority voting
        test_labels = None
        
        test_predictions = {
            'label_predicted': [],
            'true_labels': [],
            'predictions': []
        }
        
        for attr, value in list(config.model.__dict__.items()):
            if attr not in param_grid.keys() and attr in model_parameters:
                param_set[attr] = value
        
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
            
            
            train_set.p_augmentation = p_augmentation
            train_set.augmentation_techniques = []
            
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
            val_dataloader = DataLoader(val_set, batch_size=batch_size, num_workers=4, persistent_workers=True)
            test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=4, persistent_workers=True)

            # Logger for each experiment
            logger = TensorBoardLogger(save_dir=config.logger.log_dir, version=f"version_{version}_fold_{fold}", name=config.logger.experiment_name)
            
            param_set['experiment_name'] = config.logger.experiment_name
            param_set['version'] = version
            
            # param_set['alpha_fl'] = get_alpha_fl(train_dataloader.dataset.data['label'])

            # Create the model with the current hyperparameters
            module = ClassificationModule(**param_set)

            # Checkpoint callback
            checkpoint_cb = ModelCheckpoint(monitor=config.checkpoint.monitor, dirpath=os.path.join(config.logger.log_dir, config.logger.experiment_name, f'version_{version}_fold_{fold}'), filename='{epoch:03d}_{' + config.checkpoint.monitor + ':.6f}', save_weights_only=True, save_top_k=config.checkpoint.save_top_k, mode=config.checkpoint.mode)
            
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=6, verbose=True, mode="min")

            # Trainer
            trainer = Trainer(logger=logger, accelerator=device, devices=[0] if device == "gpu" else "auto", default_root_dir=config.logger.log_dir, max_epochs=config.model.epochs, check_val_every_n_epoch=1, callbacks=[checkpoint_cb, early_stop_callback], log_every_n_steps=1, num_sanity_val_steps=0,reload_dataloaders_every_n_epochs=1)

            # Train
            if not config.model.only_test:
                trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
                threhsold = (module.training_best_threshold + module.validation_best_threshold) / 2

            # Collect test predictions per fold if test set is available
            module = load_checkpoint(config, checkpoint_cb, fold, version)

            # Get test predictions
            current_test_result = trainer.predict(model=module, dataloaders=test_dataloader)
            
            for result in current_test_result:
                predicted_labels = [ 1 if p > threhsold else 0 for p in result["predictions"] ]
                test_predictions['label_predicted'].extend(predicted_labels)
                test_predictions['true_labels'].extend(result["labels"])
                test_predictions['predictions'].extend(result["predictions"])
            
        
        delete_checkpoints(config.logger.experiment_name, log_dir=config.logger.log_dir)
            
        performance = calculate_statistics(test_predictions['label_predicted'], test_predictions['true_labels'], test_predictions['predictions'])
        
        # Append this result to results_list
        result_dict = {
            'version': version,
            'batch_size': batch_size,
            **{k:val for k, val in param_set.items() if k in model_parameters_toshow},
            **performance
        }
            
        results_list.append(result_dict)
        version += 1

    os.rmdir(os.path.join(config.logger.log_dir, config.logger.experiment_name))

    # Convert the results list to a pandas DataFrame
    df_results = pd.DataFrame(results_list)
    
    df_results.to_excel(os.path.join(os.path.dirname(__file__), 'results_csv', f"{config.logger.experiment_name}.xlsx"), index=False)
