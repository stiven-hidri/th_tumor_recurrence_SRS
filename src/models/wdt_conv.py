import torch
import torch.nn as nn
from .rnn import RNNModel, GRUModel   
from models.mlp_cd import MlpCD 
from models.resnet34_3d import ResNet34_3d

import os            

device = None

def generate_resnet34_3d(cuda=True):
    
    model = ResNet34_3d(shortcut_type='A', no_cuda=not cuda)
    
    model.fc = nn.Identity()
    
    return model
            
class WDTConv(nn.Module):
    def __init__(self, dropout = .1, use_clinical_data=True, out_dim_backbone=512, hidden_size_cd=64, hidden_size_fc1 = 256, hidden_size_fc = 256):
        super(WDTConv, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_clinical_data = use_clinical_data
        
        input_dim = out_dim_backbone + hidden_size_cd if self.use_clinical_data else out_dim_backbone

        self.backbone = generate_resnet34_3d()
        
        if self.use_clinical_data:
            self.cd_backbone = MlpCD(pretrained=False)
            
            self.cd_backbone.final_fc = nn.Identity()

        self.dropout = nn.Dropout(p=dropout)  # Dropout with 50% probability
        self.relu = nn.ReLU()
        
        self.final_fc1 = nn.Linear(input_dim, hidden_size_fc1)
        self.bn1 = nn.BatchNorm1d(hidden_size_fc)       
        
        self.final_fc_final = nn.Linear(hidden_size_fc, 1)
        
    
    def forward(self, mr_rtd_fusion, clinical_data):
        mr_rtd_fusion = mr_rtd_fusion.unsqueeze(1)
        
        feat = self.backbone(mr_rtd_fusion)
        
        if self.use_clinical_data:
            feat = torch.cat([feat, self.cd_backbone(clinical_data)], dim=1)
        
        out = self.dropout(self.relu(self.bn1(self.final_fc1(feat))))
        # out = self.relu(self.bn2(self.final_fc2(out)))
        out = self.final_fc_final(out)

        return out