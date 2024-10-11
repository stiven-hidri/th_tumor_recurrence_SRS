import torch
import torch.nn as nn
import torchvision.models as models

# InceptionV2 Feature Extractor
class InceptionV2Features(nn.Module):
    def __init__(self):
        super(InceptionV2Features, self).__init__()
        resnet = models.resnet18(pretrained=True)
        
        # Modify the first convolutional layer to accept 2-channel input instead of 3-channel RGB
        resnet.conv1 = nn.Conv2d(
            in_channels=2,  # 2 channels for MR and RTD
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Remove the classifier layers (we only need the feature extractor part)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.feature_extractor(x)

# LSTM Model
class ConvLongRNN(nn.Module):
    def __init__(self, dropout:.3, hidden_size=128, num_layers = 1, output_dim=1):
        super(ConvLongRNN, self).__init__()
        
        self.inception_v2 = InceptionV2Features()
        self.lstm = nn.LSTM(input_size=2048,  # Assumes InceptionV2 outputs 2048 features
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_dim)  # Binary classification (recurrence or stability)
        
    # def forward(self, mr, rtd):
        
    #     mr = mr.unsqueeze(1)
    #     rtd = rtd.unsqueeze(1)
        
    #     batch_size = mr.size()[0]
            
    #     combined_features = [ self.inception_v2(torch.cat((mr[b, :, :, :, :], rtd[b, :, :, :, :]), dim=1)) for b in range(batch_size)]

    #     data_tensor = torch.stack(combined_features, dim=0)
        
    #     data_tensor = data_tensor.squeeze(2)
    #     data_tensor = data_tensor.view(data_tensor.size(0), data_tensor.size(1), -1) 
        
    #     lstm_out, _ = self.lstm(data_tensor)
        
    #     output = self.fc(lstm_out[:, -1, :])
        
    #     return output

    def forward(self, mr, rtd):
        # Unsqueeze along the channel dimension to prepare for concatenation
        mr = mr.unsqueeze(1)  # Shape: [batch_size, 1, 40, 40, 40]
        rtd = rtd.unsqueeze(1)  # Shape: [batch_size, 1, 40, 40, 40]

        # Concatenate along the channel dimension (now the tensor has 2 channels)
        combined = torch.cat((mr, rtd), dim=1)  # Shape: [batch_size, 2, 40, 40, 40]

        # Reshape to process each 40x40 slice as a 2D image
        # Permute to move the slice dimension to the batch dimension
        # (batch_size * num_slices, num_channels, height, width)
        batch_size = combined.size(0)
        num_slices = combined.size(2)
        
        combined = combined.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [batch_size, 40, 2, 40, 40]
        combined = combined.view(-1, 2, 40, 40).squeeze()  # Shape: [batch_size * 40, 2, 40, 40]

        # Feed the entire batch of 2D slices to InceptionV2
        combined_features = self.inception_v2(combined)  # Shape: [batch_size * 40, 512, h', w']

        # Reshape back to (batch_size, num_slices, features)
        combined_features = combined_features.view(batch_size, num_slices, -1)  # Shape: [batch_size, 40, feature_size]

        # Feed into the LSTM
        lstm_out, _ = self.lstm(combined_features)  # Shape: [batch_size, num_slices, hidden_size]

        # Get the output of the last time step
        output = self.fc(lstm_out[:, -1, :])  # Shape: [batch_size, output_size]

        return output


