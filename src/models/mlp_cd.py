import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpCD(nn.Module):
    def __init__(self, dropout=.1, input_size=47, hidden_size1=18, hidden_size2=10):
        super(MlpCD, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.dropout1 = nn.Dropout(dropout)      # Dropout after first hidden layer
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # First hidden layer
        self.dropout2 = nn.Dropout(dropout)      # Dropout after first hidden layer
        
        self.final_fc = nn.Linear(hidden_size2, 1)   # Output layer
        
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
        
    def forward(self, clinical_input):
        
        x = self.dropout1(F.relu(self.fc1(clinical_input)))
        
        x = self.dropout2(F.relu(self.fc2(x)))
        
        output = self.final_fc(x)
        
        return output