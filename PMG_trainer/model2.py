import torch.nn as nn
import torch
    

class PMG(nn.Module):
    def __init__(self, model, feature_size, classes_num):
        super(PMG, self).__init__()

        self.features = model
        self.max1 = nn.MaxPool2d(kernel_size=7, stride=7)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1024 * 2),
            nn.Linear(1024 * 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )

    def forward(self, x):
        xf5 = self.features(x)
        if len(xf5.shape) >= 3:
            xf = self.max1(xf5)
        else:
            xf = xf5
        xf = xf.view(-1, 2048)
        res = self.classifier(xf)
          
        return res, xf
    
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
