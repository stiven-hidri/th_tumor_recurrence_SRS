import torch
import torch.nn as nn
from .resnet3d import generate_model as generate_model
from .rnn import RNNModel, LSTMModel, GRUModel, GRUModel_atn



class ConvRNN(nn.Module):
    def __init__(self, rnn_type='rnn', dropout = .1):
        super(ConvRNN, self).__init__()
        
        self.backbone = generate_model(10) # [10, 18, 34, 50, 101, 152, 200]
        #resnet_3d(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], n_input_channels=1)

        self.backbone.fc = nn.Identity()

        if rnn_type == 'rnn':
            self.rnn = RNNModel(512, hidden_dim=256, layer_dim=1, output_dim=1, dropout_prob=dropout)
        elif rnn_type == 'lstm':
            self.rnn = LSTMModel(512, hidden_dim=256, layer_dim=1, output_dim=1, dropout_prob=dropout)
        elif rnn_type == 'gru':
            self.rnn = GRUModel(512, hidden_dim=256, layer_dim=1, output_dim=1, dropout_prob=dropout)
        elif rnn_type == 'gru2':
            self.rnn = GRUModel_atn(512, hidden_dim=256, layer_dim=1, output_dim=1, dropout_prob=dropout)
            
        
    
    def forward(self, mr, rtd):
        
        mr.unsqueeze_(1)
        rtd.unsqueeze_(1)
        
        feat_mr = self.backbone(mr)
        feat_rtd = self.backbone(rtd)
        
        feats = [feat_mr, feat_rtd]

        feats = torch.stack(feats, dim=1)
        
        out = torch.sigmoid(self.rnn(feats))

        return out