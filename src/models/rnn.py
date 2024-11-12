import torch.nn as nn
import torch

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout_prob):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda().requires_grad_()
        out, h0 = self.rnn(x, h0.detach())

        # out, _ = self.rnn(x)

        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

    def forward(self, x):
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda().requires_grad_()
        # c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda().requires_grad_()
        
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        out, _ = self.lstm(x)

        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout_prob):
        super(GRUModel, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda().requires_grad_()

        out, _ = self.gru(x, h0.detach())
        
        # out, _ = self.gru(x)

        return out
