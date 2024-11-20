import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpCD(nn.Module):
    def __init__(self, pretrained=False, input_size=47, hidden_size1=128, hidden_size2=64, output_size=1, dropout=.1):
        super(MlpCD, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout)
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # Output layer
        self.final_fc = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        if x.shape[0] == 1:
            x = self.dropout1(F.relu(self.fc1(x)))
            x = self.dropout2(F.relu(self.fc2(x)))
        else:
            x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
            x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        
        out = self.final_fc(x)  # Use sigmoid or softmax if needed outside the model
        return out
    