import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# def load_averaged_weights():
#     path_to_weights = os.path.join(os.path.dirname(__file__), 'saved_models', 'mlp_cd')
    
#     # Get all model weight files in the specified folder
    
#     weight_files = [os.path.join(path_to_weights, f) for f in os.listdir(path_to_weights) if f.endswith('.ckpt')]
    
#     # Initialize a dictionary to accumulate weights
#     avg_state_dict = None
    
#     # Load and sum up weights from each file
#     for file in weight_files:
#         state_dict_model = torch.load(file)['state_dict']
#         state_dict = {f'{k.split("model.")[-1]}': v for k, v in state_dict_model.items()}
#         if avg_state_dict is None:
#             avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
#         else:
#             for k in avg_state_dict.keys():
#                 avg_state_dict[k] += state_dict[k]

#     # Average the weights
#     num_files = len(weight_files)
#     for k in avg_state_dict.keys():
#         avg_state_dict[k] /= num_files

#     # Load the averaged weights into the model
#     return avg_state_dict

# class MlpCD(nn.Module):
#     def __init__(self, dropout=.1, input_size=47, hidden_size1=20, hidden_size2=10, pretrained=True):
#         super(MlpCD, self).__init__()

#         self.fc1 = nn.Linear(input_size, hidden_size1)  # First hidden layer
#         self.dropout1 = nn.Dropout(dropout)      # Dropout after first hidden layer
        
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # First hidden layer
#         self.dropout2 = nn.Dropout(dropout)      # Dropout after first hidden layer
        
#         self.final_fc = nn.Linear(hidden_size2, 1)   # Output layer
        
#         # if pretrained:
#         #     self.load_state_dict(load_averaged_weights())
#         #     for param in self.fc1.parameters():
#         #         param.requires_grad = False
#         #     for param in self.fc2.parameters():
#         #         param.requires_grad = False
        
#     def forward(self, clinical_input):
        
#         x = self.dropout1(F.relu(self.fc1(clinical_input)))
        
#         x = F.relu(self.fc2(x))
        
#         output = self.final_fc(x)
        
#         return output

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
        
        # Input -> First hidden layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Second hidden layer
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Output layer
        x = self.final_fc(x)  # Use sigmoid or softmax if needed outside the model
        return x