import os
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import torch
import numpy as np
from lightning.pytorch import LightningModule
from models.base_model import BaseModel
from models.conv_rnn import ConvRNN
from utils.loss_functions import BCELoss, WeightedBCELoss, FocalLoss

class ClassificationModule(LightningModule):
    def __init__(self, name: str, epochs: int, lr: float, optimizer: str, scheduler: str, weight_decay: float, lf:str, pos_weight:float, dropout: float, alpha_fl:float, gamma_fl:float, rnn_type: str, experiment_name: str, version: int):
        super().__init__()
        self.save_hyperparameters()
        self.name = name
        
        # Config    
        # Network
        if 'base_model' in name:
            self.model = BaseModel(dropout=dropout)
        elif 'conv_rnn' in name:
            self.model = ConvRNN(dropout=dropout, rnn_type=rnn_type)
        else:
            raise ValueError(f'Network {name} not available.')

        # Training params
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.weight_decay = weight_decay
        self.rnn_type = rnn_type
        self.alpha_fl = alpha_fl
        self.gamma_fl = gamma_fl
        self.dropout = dropout
        self.pos_weight = pos_weight
        self.experiment_name = experiment_name
        self.version = version
        
        if lf == "bce":
            self.lf = BCELoss()
        elif lf == "wbce":
            self.lf = WeightedBCELoss(self.pos_weight, 1-self.pos_weight)
        else:
            self.lf = FocalLoss(alpha=alpha_fl, gamma=gamma_fl)
    
        self.test_outputs = []
        self.validation_outputs = []
        self.validation_losses = []

    def forward(self, mr, rtd, clinic_data):
        if self.name == 'base_model':
            y = self.model(mr, rtd)
        else:
            y = self.model(mr, rtd)
            
        return y

    def loss_function(self, prediction, label):
        return self.lf.forward(prediction, label)

    def training_step(self, batch):
        mr, rd, clinic_data, label = batch
        
        prediction = self(mr, rd, clinic_data)
        loss = self.loss_function(prediction, label)
        self.log('train_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, on_epoch=True)   

    def on_validation_epoch_end(self):
        validation_outputs_path = os.path.join(os.path.dirname(__file__), '..', 'log', self.experiment_name, f'version_{self.version}', "validation_outputs.pkl")
        
        if os.path.exists(validation_outputs_path):
            with open(validation_outputs_path, 'rb') as f:
                previous_validation_outputs, min_loss = pickle.load(f)
                
        if not os.path.exists(validation_outputs_path) or np.mean(self.validation_losses) < min_loss:
            with open(validation_outputs_path, 'wb') as f:
                pickle.dump((self.validation_outputs, np.mean(self.validation_losses)), f)

    def on_validation_epoch_start(self):
        self.validation_outputs = []
        self.validation_losses = []

    def validation_step(self, batch):
        mr, rd, clinic_data, label = batch
        prediction = self(mr, rd, clinic_data)
        loss = self.loss_function(prediction, label)
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        for i in range(mr.shape[0]):
            self.validation_outputs.append((
                prediction[i],
                label[i]
            )
        )
            
        self.validation_losses.append(loss.item())
    
        return {"loss": loss}
    
    def test_step(self, batch):
        mr, rd, clinic_data, label = batch
        #prediction = self(mr, rd, clinic_data)

        prediction = torch.sigmoid(self(mr, rd, clinic_data))

        for i in range(mr.shape[0]):
            self.test_outputs.append((
                prediction[i],
                label[i],
                self.loss_function(prediction[i], label[i])    
            ))  

        return None

    def choose_best_threshold(self, validation_outputs):
        
        pred_probs = [vo[0] for vo in validation_outputs]
        true_labels = [vo[1] for vo in validation_outputs]
        
        thresholds = np.arange(0.0, 1.0, 0.05)
        best_threshold = 0.5
        best_f1 = 0

        for t in thresholds:
            pred_labels = np.array([1 if p >= t else 0 for p in pred_probs])
            C = confusion_matrix(true_labels, pred_labels)
            TP, TN, FP, FN = C[1,1], C[0,0], C[0,1], C[1,0]
            
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = t
        
        return best_threshold

    def on_test_epoch_end(self):        
            
        validation_outputs_path = os.path.join(os.path.dirname(__file__), '..', 'log', self.experiment_name, f'version_{self.version}', "validation_outputs.pkl")
        
        if os.path.exists(validation_outputs_path):
            with open(validation_outputs_path, 'rb') as f:
                validation_outputs, min_loss = pickle.load(f)
        
        t = self.choose_best_threshold(validation_outputs)
        
        pred_labels = np.array([1 if (prediction).cpu().item() >= t else 0 for prediction, _, _ in self.test_outputs])
        true_labels = np.array([int(label.cpu().item()) for _, label, _ in self.test_outputs])
        
        C = confusion_matrix(true_labels, pred_labels)

        TP, TN, FP, FN = C[1,1], C[0,0], C[0,1], C[1,0]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        
        self.log('test_accuracy', accuracy, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('threshold', t, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        self.log('test_precision', precision, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_sensitivity', sensitivity, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_specificity', specificity, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_f1', f1_score, logger=True, prog_bar=True, on_step=False, on_epoch=True)

        pred_probs = np.array([prediction.cpu().item() for prediction, _, _ in self.test_outputs])
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.plot(fpr, tpr, label='ROC Curve', color='blue')
        ax.plot([0, 1], [0, 1], 'k--', label='Chance Level')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC')

        for i, threshold in enumerate(thresholds):
            ax.annotate(f'{threshold:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center')

        ax.legend()
        fig.canvas.draw()
        
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.copy() 
        plot_image = torch.from_numpy(plot_image)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.logger.experiment.add_image(f'ROC_Curve.png', plot_image, self.current_epoch, dataformats='HWC')

        plt.close(fig)

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
        
        if self.scheduler == 'cosine':
            print("Using CosineAnnealingLR scheduler")
            scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=self.epochs, eta_min=1e-8,)]
        
        elif self.scheduler == 'step':
            print("Using StepLR scheduler")
            scheduler = [torch.optim.lr_scheduler.StepLR(optimizers, step_size=10, gamma=0.1)]
            
        elif self.scheduler == 'exp':
            print("Using exp scheduler")
            scheduler = [torch.optim.lr_scheduler.ExponentialLR(optimizers, gamma=.9)]
        
        elif self.scheduler == 'plateau':
            print("Using ReduceLROnPlateau scheduler")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers, mode='min', factor=0.1, patience=2, min_lr=1e-12)
            return  {
                        'optimizer': optimizers,
                        'lr_scheduler': scheduler,
                        'monitor': 'val_loss'
                    }
        
        if scheduler is not None:
            return [optimizers], scheduler
        return [optimizers]
