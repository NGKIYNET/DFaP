import torch
from torch import nn

class wn(nn.Module):

    def __init__(self, num_class, num_featrues):
        super(wn, self).__init__()
        self.num_class = num_class
        self.num_features = num_featrues
        self._w = torch.ones([self.num_class, self.num_features])
        self.wn = nn.Parameter(data=self._w, requires_grad=True)

    def forward(self, input):
        return torch.matmul(input, self.wn)


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)



class residual_block(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if self.downsample:
            residual = self.downsample(x)
        out = out+residual
        out = F.relu(out, True)
        return out


class ResNet34(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(16, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=1)
        self.layer3 = self._make_layer(128, 256, 6, stride=1)
        self.layer4 = self._make_layer(256, 256, 3, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)                      
        self.wn = wn(256, num_classes)                              


    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(nn.Conv2d(
            inchannel, outchannel, 1, stride, bias=False), nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(residual_block(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(residual_block(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        filter = []
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        filter.append(x.cpu())
        x = self.pool(x)
        x = torch.reshape(x, [-1, 256])
        x = self.wn(x)
        return x,filter