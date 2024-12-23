import torch
import torch.nn as nn
from models.mlp_cd import MlpCD 
from models.resnet34_3d import ResNet34_3d
from models.convolutional_backbone import ConvBackbone3D, MobileNet3D
import os            

device = None

def generate_resnet34_3d(cuda=True):
    model = ResNet34_3d(shortcut_type='A', no_cuda=not cuda, pretrained=False)
    model.fc = nn.Identity()
    
    return model
            
class WDTConv(nn.Module):
    def __init__(self, dropout = .1, use_clinical_data=True, out_dim_backbone=256, out_dim_clincal_features=64):
        super(WDTConv, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_clinical_data = use_clinical_data
        
        input_dim = out_dim_backbone + out_dim_clincal_features if self.use_clinical_data else out_dim_backbone
        
        self.backbone = ConvBackbone3D(out_dim_backbone=out_dim_backbone, dropout=dropout)
        
        if self.use_clinical_data:
            self.cd_backbone = MlpCD(pretrained=False)
            self.cd_backbone.final_fc = nn.Identity()  
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.final_fc = nn.Linear(input_dim, 1)
    
    def forward(self, mr_rtd_fusion, clinical_data):
        mr_rtd_fusion = mr_rtd_fusion.unsqueeze(1)
        
        feat = self.backbone(mr_rtd_fusion)
        
        if self.use_clinical_data:
            feat = torch.cat([feat, self.cd_backbone(clinical_data)], dim=1)
        
        out = self.final_fc(feat)

        return out