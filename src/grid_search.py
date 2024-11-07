from itertools import product
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
from params_grid import param_grid_basemodel, param_grid_convlstm, param_grid_mlpcd, param_grid_wdt
import numpy as np
from collections import defaultdict, Counter
from argparse import ArgumentParser

torch.set_num_threads(8)
torch.cuda.set_per_process_memory_fraction(fraction=.33, device=None)

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
    "p_augmentation_per_technique"
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
    "gamma_fl",
    "pos_weight",
    "p_augmentation",
]

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

def calculate_statistics(pred_labels, true_labels):
    C = confusion_matrix(true_labels, pred_labels)
    TP, TN, FP, FN = C[1,1], C[0,0], C[0,1], C[1,0]
    
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_labels)
        
    statistics = {
        "accuracy": (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
        "recall": recall,
        "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
        "precision": precision,
        "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        "pr_auc": auc(recall_curve, precision_curve)
    }
    
    return statistics

def choose_best_threshold_validation(self, predictions, labels):
    pred_probs = [p.cpu() for p in predictions]
    true_labels = [l.cpu() for l in labels]
            
    fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)

    youden_j = tpr - fpr
    best_t_j = thresholds[np.argmax(youden_j)]
    
    return ['j'], [best_t_j], [self.calculate_statistics(pred_probs, true_labels, best_t_j)]
    
def get_alpha_fl(all_labels_dataloader):
    all_labels = [int(l) for l in all_labels_dataloader]  # Adjust 'label' to your actual label field

    positive_count = sum(all_labels)
    negative_count = len(all_labels) - positive_count
    
    alpha_fl = negative_count / (positive_count + negative_count)
    
    return alpha_fl
    
def calculate_mean_statistics(statistics):
    mean_statistics = {}
    n = float(len(statistics))
    
    for key in statistics[0].keys():
        s = 0.0
        for element in statistics:
            s+=float(element[key])
            
        mean_statistics["val_"+key] = float(s / n)
        
    return mean_statistics
    
def cross_validate_only_training_set(list_train, list_val, test_set, config, param_set, version):
    # Define test dataloader if external test set is available
    
    batch_size = param_set['batch_size']
    
    del param_set['batch_size']
    
    test_dataloader = DataLoader(test_set, batch_size=batch_size, num_workers=4, persistent_workers=True) if test_set is not None else None
    
    # Initialize list to collect test predictions for majority voting
    test_predictions = defaultdict(list)
    test_labels = None
    val_thresholds = np.array([])
    val_statistics = np.array([])
    
    for attr, value in list(config.model.__dict__.items()):
        if attr not in param_grid.keys() and attr in model_parameters:
            param_set[attr] = value
    
    # K-fold cross-validation
    for fold, (train_set, val_set) in enumerate(zip(list_train, list_val)):     
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, num_workers=4, persistent_workers=True)

        # Logger for each experiment
        logger = TensorBoardLogger(save_dir=config.logger.log_dir, version=version, name=config.logger.experiment_name)
        
        param_set['experiment_name'] = config.logger.experiment_name
        param_set['version'] = version
        
        param_set['alpha_fl'] = get_alpha_fl(train_dataloader.dataset.data['label'])

        # Create the model with the current hyperparameters
        module = ClassificationModule(**param_set)

        # Checkpoint callback
        checkpoint_cb = ModelCheckpoint(monitor=config.checkpoint.monitor, dirpath=os.path.join(config.logger.log_dir, config.logger.experiment_name, f'version_{version}_fold_{fold}'), filename='{epoch:03d}_{' + config.checkpoint.monitor + ':.6f}', save_weights_only=True, save_top_k=config.checkpoint.save_top_k, mode=config.checkpoint.mode)
        
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=3, verbose=False, mode="min")

        # Trainer
        trainer = Trainer(logger=logger, accelerator=device, devices=[0] if device == "gpu" else "auto", default_root_dir=config.logger.log_dir, max_epochs=config.model.epochs, check_val_every_n_epoch=1, callbacks=[checkpoint_cb, early_stop_callback], log_every_n_steps=1, num_sanity_val_steps=0,reload_dataloaders_every_n_epochs=1)

        # Train
        if not config.model.only_test:
            trainer.fit(model=module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            val_thresholds = np.append(val_thresholds, module.validation_best_threshold)
            val_statistics = np.append(val_statistics, module.best_validation_statistics)

        # Collect test predictions per fold if test set is available
        module = load_checkpoint(config, checkpoint_cb, fold, version)

        # Get test predictions
        test_results = trainer.predict(model=module, dataloaders=test_dataloader)
        
        labels = []
        cnt_prediction = 0
        
        for result in test_results:
            for sample_i, (p, l) in enumerate(zip(result["predictions"], result["labels"])):
                test_predictions[cnt_prediction].append(p.item())  # Collect predictions for each test sample
                cnt_prediction += 1
                if test_labels is None:
                    labels.append(l.item())
        
        if test_labels is None:
            test_labels = labels

    # Perform majority voting
    final_predictions = []
    
    final_t = np.mean(val_thresholds)
    
    final_val_statistics = calculate_mean_statistics(val_statistics)
    
    for sample_idx, preds in test_predictions.items():
        test_predictions[sample_idx] = [1 if p > final_t else 0 for p in preds]
    
    for sample_idx, preds in test_predictions.items():
        majority_vote = Counter(preds).most_common(1)[0][0]  # Majority vote
        final_predictions.append(majority_vote)
        
    statistics = calculate_statistics(final_predictions, test_labels)

    # Append this result to results_list
    return  {
        'version': version,
        'batch_size': batch_size,
        **{k:val for k, val in param_set.items() if k in model_parameters_toshow},
        'threshold':final_t,
        **final_val_statistics,
        **statistics
    }

def cross_validate_whole_dataset():
    pass

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
    elif config.model.name == "model_wdt":
        param_grid = param_grid_wdt
    else:
        raise NotImplementedError("Model not implemented")
    
    keep_test = config.logger.keep_test
    
    # Generate all parameter combinations
    keys, values = zip(*param_grid.items())
    all_param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Load sources and create datasets
    classifier_dataset = ClassifierDataset(model_name=config.model.name, keep_test=keep_test)
    
    for i, param_set in enumerate(all_param_combinations):
        print(f"**********\n{i+1}/{len(all_param_combinations)}:\n**********")
        
        # contains trian and val folds and eventually test set
        if "p_augmentation" in param_set.keys():
            p_augmentation = param_set["p_augmentation"]
        else:
            p_augmentation=0.
        list_train, list_val, test_set = classifier_dataset.create_splits(p_augmentation=p_augmentation, augmentation_techniques=[])
        
        if keep_test:
            result_dict = cross_validate_only_training_set(list_train, list_val, test_set, config, param_set, version)
        else:
            result_dict = cross_validate_whole_dataset(list_train, list_val, config, param_set, version)
            
        results_list.append(result_dict)
        version += 1

    # Convert the results list to a pandas DataFrame
    df_results = pd.DataFrame(results_list)
    
    df_results.to_excel(os.path.join(os.path.dirname(__file__), 'results_csv', f"{config.logger.experiment_name}.xlsx"), index=False)
