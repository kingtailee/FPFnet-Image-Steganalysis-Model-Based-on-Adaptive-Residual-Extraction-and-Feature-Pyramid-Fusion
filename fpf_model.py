import torch.nn as nn
import torch.nn.functional as F
import torch
from srm_filter_kernel import AllConnectFilter_bn_3


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=1, padding=size // 2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, feature_map):
        x = self.conv(feature_map)
        x = self.norm(x)
        x = self.act(x)
        return x


class AvgConv(nn.Module):
    def __init__(self, inchannel, outchannel, poolsize=3):
        super(AvgConv, self).__init__()
        self.avgpool = nn.AvgPool2d(poolsize, stride=2,padding=1)
        self.conv1 = Conv(inchannel, outchannel)

    def forward(self, input):
        x = self.avgpool(input)
        x = self.conv1(x)
        return x


class UnConvolution(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(UnConvolution, self).__init__()
        self.unconv = nn.ConvTranspose2d(inchannel, outchannel, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv=Conv(64, 32)
        self.norm = nn.BatchNorm2d(outchannel)
        self.act = nn.ReLU()

    def forward(self, input1, input2):
        input1 = self.unconv(input1)
        x = torch.cat((input1, input2), dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FeatureExtract_relu(nn.Module):
    def __init__(self):
        super(FeatureExtract_relu, self).__init__()

        self.conv1 = Conv(64, 32)
        self.conv2 = Conv(32, 32)
        self.conv3 = Conv(32, 32)

        self.downconv1 = AvgConv(32, 32)
        self.downconv2 = AvgConv(32, 32)
        self.downconv3 = AvgConv(32, 32)

        self.upconv1 = UnConvolution(32, 32)
        self.upconv2 = UnConvolution(32, 32)
        self.upconv3 = UnConvolution(32, 32)

        self.downconv4 = AvgConv(32, 32)
        self.downconv5 = AvgConv(32, 32)

        self.convglobal = Conv(32, 128)
        self.global_avg = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(128, 512)
        self.fc2 = torch.nn.Linear(512, 1024)
        self.fc3 = torch.nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x1 = x
        x = self.downconv1(x)
        x2 = x
        x = self.downconv2(x)
        x3 = x
        x = self.downconv3(x)
        x4 = x

        x3 = self.upconv3(x4, x3)
        x2 = self.upconv2(x3, x2)
        x = self.upconv1(x2, x1)

        x = self.downconv4(x)
        x = self.downconv5(x)

        x = self.convglobal(x)
        x = self.global_avg(x)

        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class net_exam(torch.nn.Module):
    def __init__(self):
        super(net_exam, self).__init__()
        self.pre = AllConnectFilter_bn_3()
        self.feature_extract = FeatureExtract_relu()
        self.init_weights()

    def forward(self, x):
        x = self.pre(x)
        x = self.feature_extract(x)
        return x

    def init_weights(self):
        for module in self.modules():
            if type(module) == nn.Conv2d:
                if module.weight.requires_grad:
                    nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

            if type(module) == nn.Linear:
                nn.init.normal_(module.weight.data, mean=0, std=0.01)
                nn.init.constant_(module.bias.data, val=0)
