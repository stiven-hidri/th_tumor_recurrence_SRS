import torch
import torch.nn as nn
import torch.nn.functional as F

class MlpCD(nn.Module):
    def __init__(self, dropout=.5, input_size=48, hidden_size1=10, hidden_size2=128):
        super(MlpCD, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
        self.bn1 = nn.BatchNorm1d(hidden_size1)        # Batch normalization after first hidden layer
        self.dropout1 = nn.Dropout(dropout)      # Dropout after first hidden layer
        
        # self.fc2 = nn.Linear(hidden_size1, hidden_size2) # Second hidden layer
        # self.bn2 = nn.BatchNorm1d(hidden_size2)         # Batch normalization after second hidden layer
        # self.dropout2 = nn.Dropout(dropout) 
        
        self.final_fc = nn.Linear(hidden_size1, 1)   # Output layer
        
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
        
    def forward(self, les_input, dose_input, clinical_input):
        x = clinical_input.squeeze()
            
        x = self.dropout1(self.bn1(F.relu(self.fc1(x))))
        
        # x = self.dropout2(self.bn2(F.relu(self.fc2(x))))
        
        output = self.final_fc(x)
        
        return output