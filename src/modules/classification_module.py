import os
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve, precision_recall_curve
import torch
import numpy as np
from lightning.pytorch import LightningModule
from models.base_model import BaseModel
from models.conv_rnn import ConvRNN
from models.wdt_conv import WDTConv
from models.conv_lstm import ConvLSTM
from models.trans_med import TransMedModel
from models.mlp_cd import MlpCD
from models.base_model_enhancedV2 import BaseModel_EnhancedV2
from utils.loss_functions import BCELoss, WeightedBCELoss, FocalLoss
from sklearn.metrics import roc_auc_score

class ClassificationModule(LightningModule):
    def __init__(self, name: str, epochs: int, lr: float, optimizer: str, scheduler: str, weight_decay: float, lf:str, pos_weight:float, dropout: float, use_clinical_data:bool, alpha_fl:float, gamma_fl:float, rnn_type: str, hidden_size: int, num_layers: int, experiment_name: str, version: int, augmentation_techniques: list, p_augmentation: float, depth_attention: int):
        super().__init__()
        self.save_hyperparameters()
        self.name = name
        
        print(f'Using {name} lr: {lr} weight_decay: {weight_decay} lf: {lf} dropout: {dropout} alpha_fl: {alpha_fl} gamma_fl: {gamma_fl} p_augmentation: {p_augmentation} ' )
        
        # Config    
        # Network
        if name == 'base_model_enhancedV2':
            self.model = BaseModel_EnhancedV2(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'trans_med':
            self.model = TransMedModel(use_clinical_data=use_clinical_data, dropout=dropout, depth_attention=depth_attention)
        elif name == 'wdt_conv':
            self.model = WDTConv(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'base_model':
            self.model = BaseModel(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'conv_rnn':
            self.model = ConvRNN(dropout=dropout, rnn_type=rnn_type, hidden_size=hidden_size, num_layers=num_layers, use_clinical_data=use_clinical_data)
        elif name == 'conv_lstm':
            self.model = ConvLSTM(dropout=dropout, hidden_size=hidden_size, num_layers=num_layers, use_clinical_data=use_clinical_data, rnn_type=rnn_type)
        elif name == 'mlp_cd':
            self.model = MlpCD(dropout=dropout, pretrained=False)
        else:
            raise ValueError(f'Network {name} not available.')

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_clinical_data = use_clinical_data
        self.alpha_fl = alpha_fl
        self.gamma_fl = gamma_fl
        self.dropout = dropout
        self.pos_weight = pos_weight
        self.experiment_name = experiment_name
        self.version = version
        self.augmentation_techniques = augmentation_techniques
        self.p_augmentation = p_augmentation
        self.depth_attention = depth_attention
        
        if lf == "bce":
            self.lf = BCELoss()
        elif lf == "wbce":
            self.lf = WeightedBCELoss(self.pos_weight, 1-self.pos_weight)
        else:
            self.lf = FocalLoss(alpha=alpha_fl, gamma=gamma_fl)
            
        self.validation_labels = []
        self.validation_losses = []
        self.validation_outputs = []
        self.validation_best_threshold = .5
        
        self.training_labels = []
        self.training_losses = []
        self.training_outputs = []
        self.training_best_threshold = .5
        
        self.test_stuff = {
            'predictions':  [],
            'labels': []
        }

    def forward(self, clinical_data, mr=None, rtd=None, mr_rtd_fusion=None):
        if 'mlp_cd' in self.name:
            y = self.model(clinical_data)
        elif 'wdt_conv' in self.name:
            y = self.model(mr_rtd_fusion, clinical_data)
        else:
            y = self.model(mr, rtd, clinical_data)
            
        return y

    def loss_function(self, prediction, label):
        return self.lf.forward(prediction, label)

    def on_train_epoch_start(self):
        self.training_labels.append([])
        self.training_losses.append([])
        self.training_outputs.append([])
    
    def training_step(self, batch):
        if self.name == 'wdt_conv':
            mr_rtd_fusion, clinical_data, label = batch
            prediction = self(mr_rtd_fusion=mr_rtd_fusion, clinical_data=clinical_data)
        else:
            mr, rtd, clinical_data, label = batch
            prediction = self(mr=mr, rtd=rtd, clinical_data=clinical_data)
        
        loss = self.loss_function(prediction, label)
            
        self.training_outputs[-1].extend(torch.sigmoid(prediction))
        self.training_labels[-1].extend(label)
        self.training_losses[-1].append(loss.item())
        
        self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        
        if len(self.training_labels) > 1:
            if np.mean(self.training_losses[-1]) < np.mean(self.training_losses[-2]):
                to_be_removed = -2
            else:
                to_be_removed = -1
                
            self.training_labels.pop(to_be_removed)
            self.training_losses.pop(to_be_removed)
            self.training_outputs.pop(to_be_removed)
        
        name, threshold, statistics = self.choose_best_threshold(self.training_outputs[-1], self.training_labels[-1])
        
        self.training_best_threshold = threshold
        self.best_training_statistics = statistics
        
        self.log('threshold_train', threshold, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('f1_train', statistics['f1_score'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('precision_train', statistics['precision'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_train', statistics['recall'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('specificity_train', statistics['specificity'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('accuracy_train', statistics['accuracy'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('pr_auc_train', statistics['pr_auc'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('roc_auc_train', statistics['roc_auc'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('lr', lr, prog_bar=True, on_epoch=True)   

    def calculate_statistics(self, pred_probs, true_labels, t):
        pred_labels = np.array([1 if p >= t else 0 for p in pred_probs])
        C = confusion_matrix(true_labels, pred_labels)
        TP, TN, FP, FN = C[1,1], C[0,0], C[0,1], C[1,0]
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        precision_curve, recall_curve, thresholds = precision_recall_curve(true_labels, pred_probs)
        roc_auc = roc_auc_score(true_labels, pred_probs)
        
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

    def choose_best_threshold(self, outputs, labels):
        pred_probs = [p.item() for p in outputs]
        true_labels = [l.item() for l in labels]
                
        #maximize_j
        fpr, tpr, thresholds_j = roc_curve(true_labels, pred_probs)
        youden_j = tpr - fpr
        best_t_j = thresholds_j[np.argmax(youden_j)]
        
        #maximize_f1
        precision, recall, thresholds_f1 = precision_recall_curve(true_labels, pred_probs)
        fscore = (2 * precision * recall) / (precision + recall)
        fscore = np.nan_to_num(fscore, nan=-np.inf)
        best_t_f1 = thresholds_f1[np.argmax(fscore)]
        
        names, thresholds_techinques, statistics = ['j', 'f1'], [best_t_j, best_t_f1], [self.calculate_statistics(pred_probs, true_labels, best_t_j), self.calculate_statistics(pred_probs, true_labels, best_t_f1)]
        
        final_i = int(np.argmax([x['f1_score'] for x in statistics]))
        return names[final_i], thresholds_techinques[final_i], statistics[final_i]

    def on_validation_epoch_start(self):
        self.validation_labels.append([])
        self.validation_losses.append([])
        self.validation_outputs.append([])

    def validation_step(self, batch):
        if self.name == 'wdt_conv':
            mr_rtd_fusion, clinical_data, label = batch
            prediction = self(mr_rtd_fusion=mr_rtd_fusion, clinical_data=clinical_data)
        else:
            mr, rtd, clinical_data, label = batch
            prediction = self(mr=mr, rtd=rtd, clinical_data=clinical_data)
            
        loss = self.loss_function(prediction, label)
        
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        self.validation_outputs[-1].extend(torch.sigmoid(prediction))
        self.validation_labels[-1].extend(label)
        
        self.validation_losses[-1].append(loss.item())
    
        return {"loss": loss}
        
    def on_validation_epoch_end(self):
        if len(self.validation_labels) > 1:
            if np.mean(self.validation_losses[-1]) < np.mean(self.validation_losses[-2]):
                to_be_removed = -2
            else:
                to_be_removed = -1
                
            self.validation_labels.pop(to_be_removed)
            self.validation_losses.pop(to_be_removed)
            self.validation_outputs.pop(to_be_removed)
        
        name, threshold, statistics = self.choose_best_threshold(self.validation_outputs[-1], self.validation_labels[-1])
        
        self.validation_best_threshold = threshold
        self.best_validation_statistics = statistics
        
        self.log('threshold_val', threshold, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('f1_val', statistics['f1_score'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('precision_val', statistics['precision'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('recall_val', statistics['recall'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('specificity_val', statistics['specificity'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('accuracy_val', statistics['accuracy'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('pr_auc_val', statistics['pr_auc'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        if self.name == 'wdt_conv':
            mr_rtd_fusion, clinical_data, label = batch
            prediction = self(mr_rtd_fusion=mr_rtd_fusion, clinical_data=clinical_data)
        else:
            mr, rtd, clinical_data, label = batch
            prediction = self(mr=mr, rtd=rtd, clinical_data=clinical_data)
        
        
        pred_probs = [p.item() for p in torch.sigmoid(prediction)]
        true_labels = [l.item() for l in label]
        
        self.test_stuff['predictions'].extend(pred_probs)
        self.test_stuff['labels'].extend(true_labels)

    def on_test_epoch_end(self):
        # Aggregate and log metrics across all batches
        performance = self.calculate_statistics(self.test_stuff['predictions'], self.test_stuff['labels'], self.validation_best_threshold)
        
        self.log('test_f1', performance['f1_score'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_precision', performance['precision'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_recall', performance['recall'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_specificity', performance['specificity'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_accuracy', performance['accuracy'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_pr_auc', performance['pr_auc'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_roc_auc', performance['roc_auc'], logger=True, prog_bar=True, on_step=False, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        if self.name == 'wdt_conv':
            mr_rtd_fusion, clinical_data, label = batch
            logits = self(mr_rtd_fusion=mr_rtd_fusion, clinical_data=clinical_data)
        else:
            mr, rtd, clinical_data, label = batch
            logits = self(mr=mr, rtd=rtd, clinical_data=clinical_data)

        predictions = [l.cpu().item() for l in torch.sigmoid(logits)]
        labels =  [l.cpu().item() for l in label]

        return {"predictions": predictions, "labels": labels}

    def configure_optimizers(self):
        scheduler = None
        
        if self.optimizer == 'adam':
            print("Using Adam optimizer")
            optimizers = torch.optim.Adam(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        
        elif self.optimizer == 'adamw':
            print("Using AdamW optimizer")
            optimizers = torch.optim.AdamW(self.parameters(), lr = self.lr)
        
        elif self.optimizer == 'sgd':
            print("Using SGD optimizer")
            optimizers = torch.optim.SGD(self.parameters(), lr = self.lr, momentum=.7, weight_decay=self.weight_decay)
            
        ##Schedulers
        
        if self.scheduler == 'cosine':
            print("Using CosineAnnealingLR scheduler")
            scheduler = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizers, T_0=10, T_mult=1, eta_min=self.lr*1e-1)]
        
        elif self.scheduler == 'step':
            print("Using StepLR scheduler")
            scheduler = [torch.optim.lr_scheduler.StepLR(optimizers, step_size=25, gamma=0.5)]
            
        elif self.scheduler == 'exp':
            print("Using exp scheduler")
            scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma=.9)]
        
        elif self.scheduler == 'plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.1, patience=1, min_lr=1e-12)
            return  {
                        'optimizer': optimizers,
                        'lr_scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
            
        elif self.scheduler == 'constant':
            print("Using constant scheduler")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers, lr_lambda=lambda epoch: 1)
            return  {
                        'optimizer': optimizers,
                        'lr_scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizers], scheduler
        return [optimizers]