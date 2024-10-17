import torch
import torch.nn as nn
from .resnet3d import generate_model as generate_model
from .rnn import RNNModel, LSTMModel, GRUModel
from torchvision.models import resnet34      
from models.shufflenetv2 import ShuffleNetV2      
            
class ConvRNN(nn.Module):
    def __init__(self, rnn_type='rnn', hidden_size=128, num_layers = 1, dropout = .1, clinical_data_input_dim=48, clinical_data_output_dim=10):
        super(ConvRNN, self).__init__()
        
        #self.backbone = generate_model(34)

        self.backbone = ShuffleNetV2(out_dim=512, sample_size=40, width_mult=2.)
        self.hidden_size = hidden_size

        # self.backbone.fc = nn.Identity()
        
        self.fc_clinical_data = nn.Linear(clinical_data_input_dim, clinical_data_output_dim)

        if rnn_type == 'rnn':
            self.rnn = RNNModel(512 + clinical_data_output_dim, hidden_dim=hidden_size, layer_dim=num_layers, output_dim=hidden_size, dropout_prob=dropout)
        elif rnn_type == 'lstm':
            self.rnn = LSTMModel(512 + clinical_data_output_dim, hidden_dim=hidden_size, layer_dim=num_layers, output_dim=hidden_size, dropout_prob=dropout)
        elif rnn_type == 'gru':
            self.rnn = GRUModel(512 + clinical_data_output_dim, hidden_dim=hidden_size, layer_dim=num_layers, output_dim=hidden_size, dropout_prob=dropout)
            
        self.final_fc = nn.Linear(hidden_size, 1)
    
    def forward(self, mr, rtd, clinical_data):
        
        mr = mr.unsqueeze(1)
        rtd = rtd.unsqueeze(1)
        
        batch_size, _, depth, height, width = mr.shape
        
        features_clinical_data = torch.relu(self.fc_clinical_data(clinical_data))
        
        feat_mr = self.backbone(mr)
        feat_rtd = self.backbone(rtd)
        
        feat_mr = torch.cat((feat_mr, features_clinical_data), dim=-1)
        feat_rtd = torch.cat((feat_rtd, features_clinical_data), dim=-1)
        
        feats = [feat_mr, feat_rtd]

        feats = torch.stack(feats, dim=1)
        
        out = self.rnn(feats)
        
        out = self.final_fc(out)

        return out