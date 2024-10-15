import torch
import torch.nn as nn
from .resnet3d import generate_model as generate_model
from .rnn import RNNModel, LSTMModel, GRUModel, GRUModel_atn
from torchvision.models import resnet34      
from models.shufflenet import ShuffleNet      
            
class ConvRNN(nn.Module):
    def __init__(self, in_dim=3, rnn_type='rnn', hidden_size=256, dropout = .1, clinical_data_input_dim=48, clinical_data_output_dim=10):
        super(ConvRNN, self).__init__()
        
        #self.backbone = generate_model(34)

        self.backbone = ShuffleNet(groups=1, width_mult=1, out_dim=512)

        # self.backbone.fc = nn.Identity()
        
        self.fc_clinical_data = nn.Linear(clinical_data_input_dim, clinical_data_output_dim)

        if rnn_type == 'rnn':
            self.rnn = RNNModel(512, hidden_dim=hidden_size, layer_dim=1, output_dim=1, dropout_prob=dropout)
        elif rnn_type == 'lstm':
            self.rnn = LSTMModel(512, hidden_dim=hidden_size, layer_dim=1, output_dim=1, dropout_prob=dropout)
        elif rnn_type == 'gru':
            self.rnn = GRUModel(512, hidden_dim=hidden_size, layer_dim=1, output_dim=1, dropout_prob=dropout)
        elif rnn_type == 'gru2':
            self.rnn = GRUModel_atn(512, hidden_dim=hidden_size, layer_dim=1, output_dim=1, dropout_prob=dropout)
    
    def forward(self, mr, rtd, clinical_data):
        
        mr.unsqueeze_(1)
        rtd.unsqueeze_(1)
        
        batch_size, _, depth, height, width = mr.shape
        
        features_clinical_data = torch.relu(self.fc_clinical_data(clinical_data)).unsqueeze(1).repeat(1, depth, 1)
        
        feat_mr = self.backbone(mr)
        feat_rtd = self.backbone(rtd)
        
        feat_mr = torch.cat((feat_mr, features_clinical_data), dim=-1)
        feat_rtd = torch.cat((feat_rtd, features_clinical_data), dim=-1)
        
        feats = [feat_mr, feat_rtd]

        feats = torch.stack(feats, dim=1)
        
        out = self.rnn(feats)

        return out