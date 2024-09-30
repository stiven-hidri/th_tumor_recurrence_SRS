import torch
import torch.nn as nn

from torchvision.models import resnet34
from resnet3d import BasicBlock, ResNet as resnet_3d

class ResNet(nn.Module):
    def __init__(self, opt):
        super(ResNet, self).__init__()
        self.backbone = resnet34()
        self.backbone.conv1 = nn.Conv2d(
            opt.in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.backbone.fc = nn.Identity()

        fc_in_channels = 512*4 + 4 if opt.with_vol else 512*4
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_channels, 128),
            nn.Dropout(0.3),
            nn.Linear(128, opt.n_classes)
        )
    
    def forward(self, mr, rtd):
        feat_mr = self.backbone(mr)
        feat_rtd = self.backbone(rtd)

        feat = torch.cat([feat_mr, feat_rtd], 1)

        out = self.classifier(feat)

        return out

class ResNet3D(nn.Module):
    def __init__(self, opt):
        super(ResNet3D, self).__init__()
        self.backbone = resnet_3d(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], n_input_channels=128)
        self.backbone.fc = nn.Identity()

        fc_in_channels = 512*4 + 4 if opt.with_vol else 512*4
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_channels, 128),
            nn.Dropout(0.3),
            nn.Linear(128, opt.n_classes)
        )

        self.opt = opt
    
    def forward(self, mr, rtd):
        feat_mr = self.backbone(mr)
        feat_rtd = self.backbone(rtd)

        feat = torch.cat([feat_mr, feat_rtd], 1)

        out = self.classifier(feat)

        return out