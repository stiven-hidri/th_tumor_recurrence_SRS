import torch
import torch.nn as nn
from .resnet3d import generate_model as generate_model
from .rnn import RNNModel, LSTMModel, GRUModel
from torchvision.models import resnet34      
from models.mlp_cd import MlpCD 
import os            
            
class ConvRNN(nn.Module):
    def __init__(self, rnn_type='rnn', hidden_size=128, num_layers = 1, dropout = .1, use_clinical_data=True, out_dim_backbone=512):
        super(ConvRNN, self).__init__()

        self.use_clinical_data = use_clinical_data
        
        input_dim_rnn = out_dim_backbone + 10 if self.use_clinical_data else out_dim_backbone

        self.backbone = generate_model(34)
        self.backbone.fc = nn.Identity()
        
        self.hidden_size = hidden_size
        
        if self.use_clinical_data:
            self.cd_backbone = MlpCD()
            path_to_mlpcd_weights = os.path.join(os.path.dirname(__file__), 'saved_models', 'mlp_cd.ckpt')
            checkpoint = torch.load(path_to_mlpcd_weights)
            checkpoint['state_dict'] = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
            self.cd_backbone.load_state_dict(checkpoint['state_dict'])
            self.cd_backbone.final_fc = nn.Identity()

        if rnn_type == 'rnn':
            self.rnn = RNNModel(input_dim_rnn, hidden_dim=hidden_size, layer_dim=num_layers, dropout_prob=dropout)
        elif rnn_type == 'gru':
            self.rnn = GRUModel(input_dim_rnn, hidden_dim=hidden_size, layer_dim=num_layers, dropout_prob=dropout)
            
        self.final_fc = nn.Linear(hidden_size, 1)
    
    def forward(self, mr, rtd, clinical_data = None):
        
        
        mr = mr.unsqueeze(1)
        rtd = rtd.unsqueeze(1)
        
        batch_size, _, depth, height, width = mr.shape
        
        if self.use_clinical_data:
            features_clinical_data = self.cd_backbone(clinical_data)
        
        feat_mr = self.backbone(mr)
        feat_rtd = self.backbone(rtd)
        
        if self.use_clinical_data:
            feat_mr = torch.cat((feat_mr, features_clinical_data), dim=-1)
            feat_rtd = torch.cat((feat_rtd, features_clinical_data), dim=-1)
        
        feats = [feat_mr, feat_rtd]

        feats = torch.stack(feats, dim=1)
        
        out = self.rnn(feats)[:, -1, :]
        
        out = self.final_fc(out)

        return out