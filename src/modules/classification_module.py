from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import torch
import numpy as np
from lightning.pytorch import LightningModule
from models.base_model import BaseModel
from models.conv_rnn import ConvRNN
from utils.loss_functions import WeightedBinaryCrossEntropy, FocalLoss, BinaryCrossEntropy

class ClassificationModule(LightningModule):
    def __init__(self, name: str, epochs: int, lr: float, optimizer: str, scheduler: str, weight_decay: float, lf:str, pos_weight:float, dropout: float, alpha_fl:float, gamma_fl:float, rnn_type: str):
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
        
        self.lf = WeightedBinaryCrossEntropy(pos_weight=torch.Tensor([self.pos_weight])) if lf == "wbce" else FocalLoss(alpha=alpha_fl, gamma=gamma_fl) if lf == "fl" else BinaryCrossEntropy() 

        # Test outputs
        self.test_outputs = []
        self.val_outputs = []

    def forward(self, mr, rtd, clinic_data):
        if self.name == 'base_model':
            y = self.model(mr, rtd, clinic_data)
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

    def validation_step(self, batch):
        mr, rd, clinic_data, label = batch
        prediction = self(mr, rd, clinic_data)
        loss = self.loss_function(prediction, label)
        self.log('val_loss', loss.item(), logger=True, prog_bar=True, on_step=False, on_epoch=True)
        
        for i in range(mr.shape[0]):
            self.val_outputs.append((
                prediction[i],
                label[i],
                self.loss_function(prediction[i], label[i])    
            ))   
        
        return {"loss": loss}
    
    def test_step(self, batch):
        mr, rd, clinic_data, label = batch
        prediction = self(mr, rd, clinic_data)

        for i in range(mr.shape[0]):
            self.test_outputs.append((
                prediction[i],
                label[i],
                self.loss_function(prediction[i], label[i])    
            ))  

        return None

    def on_test_epoch_end(self):
        # for i, (data, prediction, label, loss) in enumerate(self.test_outputs):
        #     # Determine if prediction is correct
        #     predicted_class = torch.argmax(prediction).item()
        #     correct_class = label.item()
        #     correct_prediction = (predicted_class == correct_class)

        #     # Display prediction and label with color based on correctness
        #     if correct_prediction:
        #         continue
        #     else:
        #         fig, ax = plt.subplots(figsize=(6, 6))

        #         # Display the image
        #         ax.imshow(image.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        #         ax.set_title('Input Image')
        #         prediction_text_color = 'red'

        #         prediction_text = Defect(predicted_class).name
        #         label_text = Defect(correct_class).name

        #         ax.text(0.5, -0.1, f'Prediction: {prediction_text}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color=prediction_text_color)
        #         ax.text(0.5, -0.2, f'Label: {label_text}', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='black')

        #         plt.suptitle(f'Result {i+1}')

        #         fig.canvas.draw()

        #         # Copy the buffer to make it writable
        #         plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #         plot_image = plot_image.copy()  # Make the buffer writable
        #         plot_image = torch.from_numpy(plot_image)
        #         plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        #         # Log the image to TensorBoard
        #         self.logger.experiment.add_image(f'Comparison_{i+1}.png', plot_image, self.current_epoch, dataformats='HWC')

        #         plt.close(fig)

        # Calculate the mean test loss
        
        t = .5
        
        pred_labels = np.array([1 if (prediction).cpu().item() >= t else 0 for prediction, _, _ in self.test_outputs])
        true_labels = np.array([int(label.cpu().item()) for _, label, _ in self.test_outputs])
        
        C = confusion_matrix(true_labels, pred_labels)

        TP = C[1,1] 
        TN = C[0,0]
        FP = C[0,1]
        FN = C[1,0]
        
        loss = torch.tensor([sample[-1] for sample in self.test_outputs]).mean()

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        
        self.log('test_loss', loss, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_accuracy', accuracy, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_precision', precision, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_sensitivity', sensitivity, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_specificity', specificity, logger=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_f1', f1_score, logger=True, prog_bar=True, on_step=False, on_epoch=True)

        # Plot the ROC curve

        pred_probs = np.array([prediction.cpu().item() for prediction, _, _ in self.test_outputs])
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.plot(fpr, tpr)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.plot([0, 1], [0, 1], 'k--')

        for i, threshold in enumerate(thresholds):
            ax.plot([0, fpr[i]], [tpr[i], tpr[i]], 'b-')
            ax.plot([fpr[i], fpr[i]], [0, tpr[i]], 'b-')
            ax.text(fpr[i], tpr[i], f'{threshold:.2f}', ha='center', va='bottom')

        fig.canvas.draw()

        # Copy the buffer to make it writable
        plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_image = plot_image.copy()  # Make the buffer writable
        plot_image = torch.from_numpy(plot_image)
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Log the image to TensorBoard
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
