import torch
import torch.nn as nn
import torchvision.models as models
from models.rnn import *
from models.mlp_cd import MlpCD
import os
import timm

class BackboneCNN(nn.Module):
    def __init__(self, out_features = 256):
        
        super(BackboneCNN, self).__init__()
    
        self.backbone = models.resnet34(pretrained=True)
        
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride, 
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
        )
        
        with torch.no_grad():
            original_weights = self.backbone.conv1.weight
            self.backbone.conv1.weight[:, :2, :, :] = original_weights[:, :2, :, :]
        
        in_fatures = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Linear(in_fatures, out_features)

    def forward(self, x):
        return self.backbone(x)

# LSTM Model
class ConvLSTM(nn.Module):
    def __init__(self, dropout:.3, hidden_size=64, num_layers = 2, backbone_output_feat = 256, clinical_data_output_dim = 10, use_clinical_data=True):
        super(ConvLSTM, self).__init__()
        self.use_clinical_data=use_clinical_data
        self.backbone = BackboneCNN(out_features=backbone_output_feat)
        
        self.input_dim_rnn = backbone_output_feat + clinical_data_output_dim if self.use_clinical_data else backbone_output_feat
            
        self.lstm = LSTMModel(self.input_dim_rnn, hidden_dim=hidden_size, layer_dim=num_layers, dropout_prob=dropout)
        
        self.hidden_size = hidden_size
        
        if self.use_clinical_data:
            self.cd_backbone = MlpCD(pretrained=True)
            self.cd_backbone.final_fc = nn.Identity()

        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.fc = nn.Linear(hidden_size, 1)

    def extract_slices_2channel(self, mr, rtd):
        slices = []
        
        slices_x = torch.stack((mr.permute(2, 0, 1, 3), rtd.permute(2, 0, 1, 3)), dim=2)
        # slices_y = torch.stack((mr.permute(3, 0, 2, 1), rtd.permute(3, 0, 2, 1)), dim=2)
        # slices_z = torch.stack((mr.permute(1, 0, 3, 2), rtd.permute(1, 0, 3, 2)), dim=2)

        # Concatenate all slices along a new dimension (the slice dimension)
        # slices = torch.cat((slices_x, slices_y, slices_z), dim=0)

        slices = slices_x
        
        # Reshape to have channels in the correct order
        slices = slices.permute(1, 0, 2, 3, 4)  # Shape: [batch_size, num_slices, 2, H, W]
        
        return slices
    
    def extract_slices_1channel(self, mr, rtd):
        # Stack slices along the x-axis, y-axis, and z-axis
        slices_x = torch.stack((mr.permute(2, 0, 1, 3), rtd.permute(2, 0, 1, 3)), dim=2)  # Shape: [depth_x, batch_size, 2, H, W]
        slices_y = torch.stack((mr.permute(3, 0, 2, 1), rtd.permute(3, 0, 2, 1)), dim=2)  # Shape: [depth_y, batch_size, 2, H, W]
        slices_z = torch.stack((mr.permute(1, 0, 3, 2), rtd.permute(1, 0, 3, 2)), dim=2)  # Shape: [depth_z, batch_size, 2, H, W]

        # Concatenate all slices along a new dimension (the slice dimension)
        slices = torch.cat((slices_x, slices_y, slices_z), dim=0)  # Shape: [num_slices, batch_size, 2, H, W]
        
        # Reshape to have channels in the correct order
        slices = slices.permute(1, 0, 2, 3, 4)  # Shape: [batch_size, num_slices, 2, H, W]
        slices = slices.reshape(slices.size(0), -1, 1, slices.size(3), slices.size(4))
        
        return slices

    def forward(self, mr, rtd, clinical_data):
        slices = self.extract_slices_2channel(mr, rtd)
        batch_size, num_slices, _, H, W = slices.shape
        
        features = self.backbone(slices.reshape(-1, 2, H, W)).reshape(batch_size, num_slices, -1)
        
        if self.use_clinical_data:
            features_clinical_data = self.cd_backbone(clinical_data).unsqueeze(1)
            features = torch.cat((features, features_clinical_data.expand(-1, num_slices, -1)), dim=2)
        
        lstm_output = self.layer_norm(self.lstm(features))
        output = self.fc(lstm_output[:, -1, :])

        return output


