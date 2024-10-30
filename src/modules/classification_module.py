import os
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import torch
import numpy as np
from lightning.pytorch import LightningModule
from models.base_model import BaseModel
from models.conv_rnn import ConvRNN
from models.base_model_enhanced import BaseModel_Enhanced
from models.model_wdt import ModelWDT
from models.conv_long_lstm import ConvLongLSTM
from models.mlp_cd import MlpCD
from models.base_model_enhancedV2 import BaseModel_EnhancedV2
from utils.loss_functions import BCELoss, WeightedBCELoss, FocalLoss
from sklearn.metrics import roc_auc_score

class ClassificationModule(LightningModule):
    def __init__(self, name: str, epochs: int, lr: float, optimizer: str, scheduler: str, weight_decay: float, lf:str, pos_weight:float, dropout: float, use_clinical_data:bool, alpha_fl:float, gamma_fl:float, rnn_type: str, hidden_size: int, num_layers: int, experiment_name: str, version: int, augmentation_techniques: list, p_augmentation: float, p_augmentation_per_technique: float):
        super().__init__()
        self.save_hyperparameters()
        self.name = name
        
        # Config    
        # Network
        if name == 'base_model_enhancedV2':
            self.model = BaseModel_EnhancedV2(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'base_model_enhanced':
            self.model = BaseModel_Enhanced(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'model_wdt':
            self.model = ModelWDT(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'base_model':
            self.model = BaseModel(dropout=dropout, use_clinical_data=use_clinical_data)
        elif name == 'conv_rnn':
            self.model = ConvRNN(dropout=dropout, rnn_type=rnn_type, hidden_size=hidden_size, num_layers=num_layers, use_clinical_data=use_clinical_data)
        elif name == 'conv_long_lstm':
            self.model = ConvLongLSTM(dropout=dropout, hidden_size=hidden_size, num_layers=num_layers, use_clinical_data=use_clinical_data)
        elif name == 'mlp_cd':
            self.model = MlpCD(dropout=dropout)
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
        self.p_augmentation_per_technique = p_augmentation_per_technique
        
        if lf == "bce":
            self.lf = BCELoss()
        elif lf == "wbce":
            self.lf = WeightedBCELoss(self.pos_weight, 1-self.pos_weight)
        else:
            self.lf = FocalLoss(alpha=alpha_fl, gamma=gamma_fl)
    
        self.test_outputs = []
        self.validation_outputs = []
        self.validation_losses = []

    def forward(self, clinic_data, mr=None, rtd=None, mr_rtd_fusion=None):
        if 'mlp_cd' in self.name:
            y = self.model(clinic_data)
        elif 'model_wdt' in self.name:
            y = self.model(mr_rtd_fusion, clinic_data)
        else:
            y = self.model(mr, rtd, clinic_data)
            
        return y

    def loss_function(self, prediction, label):
        return self.lf.forward(prediction, label)

    def training_step(self, batch):
        
        if self.name == 'model_wdt':
            mr_rtd_fusion, clinic_data, label = batch
            prediction = self(clinic_data, mr_rtd_fusion=mr_rtd_fusion)
        else:
            mr, rtd, clinic_data, label = batch
            prediction = self(clinic_data, mr = mr, rtd = rtd )
        
        loss = self.loss_function(prediction, label)
        self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_epoch=True)   

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        self.validation_losses = []

    def validation_step(self, batch):
        if self.name == 'model_wdt':
            mr_rtd_fusion, clinic_data, label = batch
            prediction = self(clinic_data, mr_rtd_fusion=mr_rtd_fusion)
        else:
            mr, rtd, clinic_data, label = batch
            prediction = self(clinic_data, mr = mr, rtd = rtd )
            
        loss = self.loss_function(prediction, label)
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        for i in range(label.shape[0]):
            self.validation_outputs.append((
                torch.sigmoid(prediction[i]),
                label[i]
            )
        )
            
        self.validation_losses.append(loss.item())
    
        return {"loss": loss}
    
    def on_validation_epoch_end(self):
        validation_outputs_path = os.path.join(os.path.dirname(__file__), '..', 'log', self.experiment_name, f'version_{self.version}', "validation_outputs.pkl")
        
        if os.path.exists(validation_outputs_path):
            with open(validation_outputs_path, 'rb') as f:
                previous_validation_outputs, min_loss = pickle.load(f)
                
        if not os.path.exists(validation_outputs_path) or np.mean(self.validation_losses) < min_loss:
            with open(validation_outputs_path, 'wb') as f:
                pickle.dump((self.validation_outputs, np.mean(self.validation_losses)), f)
    
    def test_step(self, batch):
        if self.name == 'model_wdt':
            mr_rtd_fusion, clinic_data, label = batch
            prediction = self(clinic_data, mr_rtd_fusion=mr_rtd_fusion)
        else:
            mr, rtd, clinic_data, label = batch
            prediction = self(clinic_data, mr = mr, rtd = rtd )

        for i in range(label.shape[0]):
            self.test_outputs.append((
                torch.sigmoid(prediction[i]),
                label[i],
                self.loss_function(prediction[i], label[i])    
            ))  

        return None        

    def calculate_statistics(self, pred_probs, true_labels, t):
        pred_labels = np.array([1 if p >= t else 0 for p in pred_probs])
        C = confusion_matrix(true_labels, pred_labels)
        TP, TN, FP, FN = C[1,1], C[0,0], C[0,1], C[1,0]
        
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        
        statistics = {
            "accuracy": (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0,
            "recall": recall,
            "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
            "precision": precision,
            "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
            "auc_roc": roc_auc_score(true_labels, pred_labels)
        }
        
        return statistics
    def choose_best_threshold_validation(self, validation_outputs):
        
        pred_probs = [vo[0].cpu() for vo in validation_outputs]
        true_labels = [vo[1].cpu() for vo in validation_outputs]
        
        # thresholds = np.arange(0.0, 1.0, 0.01)
        # best_t_f1 = 0.5
        # best_f1 = 0

        # for t in thresholds:
        #     f1_score = self.calculate_statistics(pred_probs, true_labels, t)['f1_score']

        #     if f1_score > best_f1:
        #         best_f1 = f1_score
        #         best_t_f1 = t
                
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)

        # 1. Maximizing Youden's J statistic (J = TPR - FPR)
        youden_j = tpr - fpr
        best_threshold_index_j = np.argmax(youden_j)
        best_t_j = thresholds[best_threshold_index_j]
        
        # distances = np.sqrt((1 - tpr)**2 + fpr**2)
        # best_threshold_index_dist = np.argmin(distances)
        # best_t_roc = thresholds[best_threshold_index_dist]
        
        #  return ['f1', 'j', 'roc'], [best_t_f1, best_t_j, best_t_roc], [self.calculate_statistics(pred_probs, true_labels, best_f1), self.calculate_statistics(pred_probs, true_labels, best_t_j), self.calculate_statistics(pred_probs, true_labels, best_t_j), self.calculate_statistics(pred_probs, true_labels, best_t_roc)]
        return ['j'], [best_t_j], [self.calculate_statistics(pred_probs, true_labels, best_t_j), self.calculate_statistics(pred_probs, true_labels, best_t_j)]

    def on_test_epoch_end(self):        
            
        validation_outputs_path = os.path.join(os.path.dirname(__file__), '..', 'log', self.experiment_name, f'version_{self.version}', "validation_outputs.pkl")
        
        if os.path.exists(validation_outputs_path):
            with open(validation_outputs_path, 'rb') as f:
                validation_outputs, min_loss = pickle.load(f)
        
        names, thresholds, val_statistics = self.choose_best_threshold_validation(validation_outputs)
        
        for name, t, val_stat in zip(names, thresholds, val_statistics):
            self.log(f'VAL_{name}_t\n', t, logger=True, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'VAL_{name}_Accuracy', val_stat["accuracy"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'VAL_{name}_AUC', val_stat["auc_roc"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'VAL_{name}_Precision', val_stat["precision"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'VAL_{name}_Recall', val_stat["recall"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'VAL_{name}_Specificity', val_stat["specificity"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'VAL_{name}_F1', val_stat["f1_score"], logger=True, prog_bar=False, on_epoch=True)
            
            print()
        
        pred_probs = np.array([prediction.cpu().item() for prediction, _, _ in self.test_outputs])
        true_labels = np.array([int(label.cpu().item()) for _, label, _ in self.test_outputs])
        
        for name, t in zip(names, thresholds):
        
            test_statistics = self.calculate_statistics(pred_probs, true_labels, t)
            
            self.log(f'TEST_{name}_t\n', t, logger=True, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'TEST_{name}_Accuracy', test_statistics["accuracy"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'TEST_{name}_AUC', test_statistics["auc_roc"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'TEST_{name}_Precision', test_statistics["precision"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'TEST_{name}_Recall', test_statistics["recall"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'TEST_{name}_Specificity', test_statistics["specificity"], logger=True, prog_bar=False, on_epoch=True)
            self.log(f'TEST_{name}_F1', test_statistics["f1_score"], logger=True, prog_bar=False, on_epoch=True)
            
            print()

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
            optimizers = torch.optim.SGD(self.parameters(), lr = self.lr)
            
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.5, patience=1, min_lr=1e-12)
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