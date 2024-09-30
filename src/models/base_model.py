import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, dropout=.3):
        super(BaseModel, self).__init__()
        
        # Lesion input branch
        self.les_conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.les_pool1 = nn.MaxPool3d(2)
        self.les_bn1 = nn.BatchNorm3d(32)
        
        self.les_conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.les_pool2 = nn.MaxPool3d(2)
        self.les_bn2 = nn.BatchNorm3d(64)
        
        self.les_conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.les_pool3 = nn.MaxPool3d(2)
        self.les_bn3 = nn.BatchNorm3d(128)
        
        self.les_global_pool = nn.AdaptiveAvgPool3d(1)
        self.les_fc1 = nn.Linear(128, 512)
        self.les_dropout = nn.Dropout(dropout)
        self.les_fc2 = nn.Linear(512, 128)
        
        # Dose input branch
        self.dose_conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.dose_pool1 = nn.MaxPool3d(2)
        self.dose_bn1 = nn.BatchNorm3d(32)
        
        self.dose_conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.dose_pool2 = nn.MaxPool3d(2)
        self.dose_bn2 = nn.BatchNorm3d(64)
        
        self.dose_conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.dose_pool3 = nn.MaxPool3d(2)
        self.dose_bn3 = nn.BatchNorm3d(128)
        
        self.dose_global_pool = nn.AdaptiveAvgPool3d(1)
        self.dose_fc1 = nn.Linear(128, 512)
        self.dose_dropout = nn.Dropout(dropout)
        self.dose_fc2 = nn.Linear(512, 128)
        
        # Clinical input branch
        self.clinical_fc1 = nn.Linear(216, 128)  # Placeholder for dynamic input layer
        
        # Final output layer
        self.final_fc = nn.Linear(128 * 3, 1)
        
        # Xavier initialization
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
        
    def forward(self, les_input, dose_input, clinical_input):
        
        les_input = les_input.unsqueeze(1)
        dose_input = dose_input.unsqueeze(1)
        clinical_input = clinical_input.squeeze()
        
        # Lesion branch
        x = F.relu(self.les_conv1(les_input))
        x = self.les_pool1(x)
        x = self.les_bn1(x)
        
        x = F.relu(self.les_conv2(x))
        x = self.les_pool2(x)
        x = self.les_bn2(x)
        
        x = F.relu(self.les_conv3(x))
        x = self.les_pool3(x)
        x = self.les_bn3(x)
        
        x = self.les_global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.les_fc1(x))
        x = self.les_dropout(x)
        les_output = F.relu(self.les_fc2(x))
        
        # Dose branch
        x = F.relu(self.dose_conv1(dose_input))
        x = self.dose_pool1(x)
        x = self.dose_bn1(x)
        
        x = F.relu(self.dose_conv2(x))
        x = self.dose_pool2(x)
        x = self.dose_bn2(x)
        
        x = F.relu(self.dose_conv3(x))
        x = self.dose_pool3(x)
        x = self.dose_bn3(x)
        
        x = self.dose_global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dose_fc1(x))
        x = self.dose_dropout(x)
        dose_output = F.relu(self.dose_fc2(x))
        
        if self.clinical_fc1.in_features != clinical_input.size(1):
            # Create the linear layer dynamically based on the input size
            num_features = clinical_input.size(1)  # Get number of features dynamically
            self.clinical_fc1 = nn.Linear(num_features, 128)  # Initialize layer with dynamic input size
            
        
        # Clinical branch
        clinical_output = F.relu(self.clinical_fc1(clinical_input))
        # clinical_output = F.relu(self.clinical_fc2(x))
        
        # If clinical_output ends up being [128] instead of [8, 128], add the batch dimension
        if clinical_output.dim() == 1:
            clinical_output = clinical_output.unsqueeze(0)  # Add batch dimension

        # Concatenate outputs
        # combined = torch.cat((les_output, dose_output), dim=1)
        combined = torch.cat((les_output, dose_output, clinical_output), dim=1)
        
        output = torch.sigmoid(self.final_fc(combined))
        
        return output