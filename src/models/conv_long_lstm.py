import torch
import torch.nn as nn
import torchvision.models as models
from models.rnn import LSTMModel
from models.mlp_cd import MlpCD
import os

class BackboneCNN(nn.Module):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        
        self.backbone = models.resnet34(pretrained=True)
        
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        
        self.backbone.conv1 = nn.Conv2d(in_channels=2,
                                        out_channels=self.backbone.conv1.out_channels,
                                        kernel_size=self.backbone.conv1.kernel_size,
                                        stride=self.backbone.conv1.stride, 
                                        padding=self.backbone.conv1.padding,
                                        bias=self.backbone.conv1.bias)
        
        with torch.no_grad():
            original_weights = self.backbone.conv1.weight
            self.backbone.conv1.weight[:, :2, :, :] = original_weights[:, :2, :, :]
            
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
            
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

# LSTM Model
class ConvLongLSTM(nn.Module):
    def __init__(self, dropout:.3, hidden_size=64, num_layers = 2, output_dim=1, clinical_data_output_dim = 10, use_clinical_data=True):
        super(ConvLongLSTM, self).__init__()
        self.use_clinical_data=use_clinical_data
        self.backbone = BackboneCNN()
        
        self.input_dim_rnn = 512 + clinical_data_output_dim if self.use_clinical_data else 512
            
        self.lstm = LSTMModel(self.input_dim_rnn, hidden_dim=hidden_size, layer_dim=num_layers, dropout_prob=dropout)
        
        self.hidden_size = hidden_size
        
        if self.use_clinical_data:
            self.cd_backbone = MlpCD()
            path_to_mlpcd_weights = os.path.join(os.path.dirname(__file__), 'saved_models', 'mlp_cd.ckpt')
            checkpoint = torch.load(path_to_mlpcd_weights)
            checkpoint['state_dict'] = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items()}
            self.cd_backbone.load_state_dict(checkpoint['state_dict'])
            self.cd_backbone.final_fc = nn.Identity()
            
            for param in self.cd_backbone.parameters():
                param.requires_grad = False

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, mr, rtd, clinical_data):
        
        batch_size, depth, height, width = mr.shape
        channels = 2
        
        combined = torch.stack([mr, rtd], dim=2).view(-1, channels, height, width)

        features = self.backbone(combined).view(batch_size, depth, -1)
        
        if self.use_clinical_data:
            features_clinical_data = torch.relu(self.cd_backbone(clinical_data)).unsqueeze(1).repeat(1, depth, 1)
            final_features = torch.cat((features, features_clinical_data), dim=-1)
        else:
            final_features = features

        lstm_out = self.layer_norm(self.lstm(final_features))

        output = self.fc(lstm_out[:, -1, :])

        return output


