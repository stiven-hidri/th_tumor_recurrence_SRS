import torch
import torch.nn as nn
import torchvision.models as models
from models.mlp_cd import MlpCD
from einops import rearrange
from models.convolutional_backbone import ConvBackbone, MobileNet

class BackboneCNN(nn.Module):
    def __init__(self, out_features=512, pretrained=True):
        
        super(BackboneCNN, self).__init__()
    
        self.backbone = models.resnet18(pretrained=pretrained)
        
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride, 
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
        )
        
        if pretrained:
            with torch.no_grad():
                original_weights = self.backbone.conv1.weight
                self.backbone.conv1.weight[:, :2, :, :] = original_weights[:, :2, :, :]
        
        in_fatures = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Linear(in_fatures, out_features)

    def forward(self, x):
        return self.backbone(x)

# LSTM Model
class ConvLSTM(nn.Module):
    def __init__(self, dropout:.3, hidden_size=64, num_layers = 2, out_dim_backbone = 512, out_dim_clincal_features = 64, use_clinical_data=True):
        super(ConvLSTM, self).__init__()
        self.use_clinical_data=use_clinical_data
        
        self.backbone = BackboneCNN(out_dim_backbone, pretrained=False)
        
        # self.backbone = MobileNet(in_channels=2, dropout=dropout, out_dim_backbone=out_dim_backbone)
        
        self.input_dim_rnn = out_dim_backbone + out_dim_clincal_features if self.use_clinical_data else out_dim_backbone
            
        self.lstm = nn.LSTM(input_size=self.input_dim_rnn, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.hidden_size = hidden_size
        
        if self.use_clinical_data:
            self.cd_backbone = MlpCD(pretrained=False)
            self.cd_backbone.final_fc = nn.Identity()

        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.fina_fc = nn.Linear(hidden_size, 1)

    def forward(self, mr, rtd, clinical_data):
        
        batch_size, d, w, h = mr.shape
        
        slices_mr = rearrange(mr, 'b d w h -> (b d) w h').unsqueeze(1)
        slices_rtd = rearrange(rtd, 'b d w h -> (b d) w h').unsqueeze(1)

        combined_slices = torch.cat([slices_mr, slices_rtd], dim=1)        
        features = self.backbone(combined_slices)
        
        features = rearrange(features, "(b l) f -> b l f", b=batch_size)
        
        if self.use_clinical_data:
            features_clinical_data = self.cd_backbone(clinical_data).unsqueeze(1)
            features = torch.cat((features, features_clinical_data.expand(-1, features.shape[1], -1)), dim=2)

        lstm_output, _ = self.lstm(features)
        
        output = self.fina_fc(lstm_output)[:, -1, :]

        return output


