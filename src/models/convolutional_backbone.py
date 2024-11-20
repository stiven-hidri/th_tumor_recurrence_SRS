from torch import nn
import torch.nn.functional as F
import torch
from utils import weight_init

class ConvBackbone3D(nn.Module):
    def __init__(self, out_dim_backbone=512, dropout=0.1):
        super(ConvBackbone3D, self).__init__()
            # Define convolutional layers
            
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
        self.les_fc2 = nn.Linear(512, out_dim_backbone)
        
        weight_init(self)
    
    def forward(self, x):
        x = self.les_bn1(self.les_pool1(F.relu(self.les_conv1(x))))        
        x = self.les_bn2(self.les_pool2(F.relu(self.les_conv2(x))))
        x = self.les_bn3(self.les_pool3(F.relu(self.les_conv3(x))))
        
        x = self.les_global_pool(x).view(x.size(0), -1)
        
        x = self.les_dropout(F.relu(self.les_fc1(x)))
        wdt_output = F.relu(self.les_fc2(x))  
        
        return wdt_output
        
class ConvBackbone(nn.Module):
    def __init__(self, in_channels=3,dropout=0.1, out_dim_backbone=256):
        super(ConvBackbone, self).__init__()
        self.in_channels = in_channels
        
        # Define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 40 x 40
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 20 x 20
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 20 x 20
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 10 x 10
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 10 x 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 10 x 10
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Output: 128 x 10 x 10
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 10 x 10
            
        )
        
        self.les_global_pool = nn.AdaptiveAvgPool2d(1)
        self.les_fc1 = nn.Linear(256, 512)
        self.les_dropout = nn.Dropout(dropout)
        if out_dim_backbone != 512:
            self.les_fc2 = nn.Linear(512, out_dim_backbone)
        else:
            self.les_fc2 = nn.Identity()
        
        
        weight_init(self)
    
    def forward(self, x):
        x = self.conv_layers(x)  # Shape: (batch_size, 256, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Shape: (batch_size, 256)
        x = self.fc(x)
        return x
    
class MobileNet(nn.Module):
    def __init__(self, in_channels=3, dropout=0.1, out_dim_backbone=512):
        super(MobileNet, self).__init__()
        self.in_channels = in_channels
    
        self.mobile_net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)

        old_conv1 = self.mobile_net.features[0][0]  # Get the first convolution layer
        self.mobile_net.features[0][0] = nn.Conv2d(
            in_channels=in_channels,  # Change from 3 to 2 channels
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias is not None
        )
        
        # Initialize the new convolution weights
        nn.init.kaiming_normal_(self.mobile_net.features[0][0].weight, mode='fan_out')

        self.feature_extractor = self.mobile_net.features
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Ensure fixed-size output
            nn.Flatten(),
            nn.Linear(1280, out_dim_backbone)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.projector(x)
        return x
    
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet3D(nn.Module):
    def __init__(self, out_dim_backbone=512, sample_size=224, width_mult=1., dropout=.1):
        super(MobileNet3D, self).__init__()
        block = InvertedResidual
        input_channel = 1
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1,1,1)],
            [6,  24, 2, (2,2,2)],
            [6,  32, 3, (2,2,2)],
            [6,  64, 4, (2,2,2)],
            [6,  96, 3, (1,1,1)],
            [6, 160, 3, (2,2,2)],
            [6, 320, 1, (1,1,1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, (1,2,2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1,1,1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, out_dim_backbone),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")

