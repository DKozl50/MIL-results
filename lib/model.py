from typing import Optional, List, Type

import torch
from torch import nn
from torch.nn import functional as F


# slightly modified ResNet code from pytorch
# as ResNet is only used as a backbone and is a subject of the task, I did not write it myself
# in this modification I do not perform classification and make layer3 and layer4 "milder" 
# so that ResNet output will be 1/8 of it's original size, as it is in paper
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        groups: int = 1,
        width_per_group: int = 64,
) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        dilation: int = 1,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, dilation=1
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


# -------------------- END OF PYTORCH COPY-ish --------------------
# =================================================================
# -------------------- END OF PYTORCH COPY-ish --------------------


class PyramidParser(nn.Module):
    def __init__(self, in_features, out_features, bin_sizes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bin_extractors = []
        for bin_size in bin_sizes:
            self.bin_extractors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_features, in_features, 1),
            ))
        self.bin_extractors = nn.ModuleList(self.bin_extractors)
        self.bottleneck = nn.Conv2d(in_features * (len(bin_sizes)+1), out_features, 1)  # +1 for untransformed x
        self.bn = nn.BatchNorm2d(out_features)
        
    def forward(self, x):
        curr = []
        interp_size = x.shape[2:]
        for extractor in self.bin_extractors:
            curr.append(F.interpolate(extractor(x), interp_size, mode='bilinear', align_corners=True))
        curr.append(x)
        curr = torch.cat(curr, 1)
        curr = self.bottleneck(curr)
        curr = self.bn(curr)
        return self.relu(curr)
    
    
class PSPUpsampler(nn.Module):
    def __init__(self, in_features, out_features, multiplier):
        super().__init__()
        self.multiplier = multiplier
        
        self.conv = nn.Conv2d(in_features, out_features, 1)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        h, w = x.shape[2:]
        h *= self.multiplier
        w *= self.multiplier
        x = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
    

class PSPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2])  # resnet18
        self.pp = PyramidParser(512, 256, [1, 2, 3, 6])
        self.drop = nn.Dropout2d(p=0.1)

        self.up1 = PSPUpsampler(256, 128, 4)
        self.up2 = PSPUpsampler(128, 64, 2)
        self.pred = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.resnet(x) 
        x = self.pp(x)
        x = self.drop(x)
        
        x = self.drop(self.up1(x))
        x = self.drop(self.up2(x))
        x = self.pred(x)
        return x
