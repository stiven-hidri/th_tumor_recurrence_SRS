import torch
import torch.nn as nn
import torchvision.models as models
import math

# Custom spatial pyramid pooling function
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a list of int specifying the output size for each level of pooling
    
    returns: a tensor vector with shape [num_sample, n] where n is the concatenation of multi-level pooling outputs
    '''    
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) // 2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        
        if i == 0:
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp

# Custom ResNet34 with Spatial Pyramid Pooling
class ResNet34_SPP(nn.Module):
    def __init__(self, out_pool_size=[1, 2, 4], num_classes=1024):
        super(ResNet34_SPP, self).__init__()
        
        # Load pretrained ResNet34 model
        self.backbone = models.resnet34(pretrained=True)
        
        self.backbone.conv1 = nn.Conv2d(
            in_channels=2,out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride, 
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias
        )
        
        with torch.no_grad():
            original_weights = self.backbone.conv1.weight
            self.backbone.conv1.weight[:, :2, :, :] = original_weights[:, :2, :, :]
        
        self.out_pool_size = out_pool_size
        
        # Remove the avgpool layer to replace it with SPP
        self.backbone.avgpool = nn.Identity()  # No operation layer
        
        # Adjust the fully connected layer's input size
        num_spp_features = sum([size * size for size in out_pool_size]) * 512  # 512 is from ResNet34 last conv channels
        self.backbone.fc = nn.Linear(num_spp_features, num_classes)

    def forward(self, x):
        # Extract features from the convolutional part of ResNet34
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Get dimensions for SPP and apply it
        num_samples = x.size(0)
        previous_conv_size = [x.size(2), x.size(3)]
        x = spatial_pyramid_pool(x, num_samples, previous_conv_size, self.out_pool_size)
        
        # Fully connected layer
        x = self.backbone.fc(x)
        return x
