import numpy as np
import torch.nn as nn
import torch

filter_class_3 = [
    np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1],
        [0, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [1, -1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 1],
        [0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
]

filter_class_5 = [
    np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ], dtype=np.float32),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1]
    ], dtype=np.float32)
]


class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output


class AllConnectFilter_bn_3(nn.Module):
    def __init__(self, threshold=1, outchannel=64):
        super(AllConnectFilter_bn_3, self).__init__()
        self.threshold = threshold
        self.outchannels = outchannel

        # requires_gard=false
        hpf_weight = nn.Parameter(torch.Tensor(filter_class_3).view(8, 1, 3, 3), requires_grad=False)
        # no padding
        self.hpf = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
        self.hpf.weight = hpf_weight

        self.conv1 = torch.nn.Conv2d(in_channels=8, out_channels=outchannel, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(outchannel)
        self.tlu = TLU(self.threshold)

    def forward(self, image_in):
        output = self.hpf(image_in)
        output = self.conv1(output)
        output = self.bn(output)
        output = self.tlu(output)
        return output


class AllConnectFilter_bn_5(nn.Module):
    def __init__(self, threshold=1, outchannel=64):
        super(AllConnectFilter_bn_5, self).__init__()
        self.threshold = threshold
        self.outchannels = outchannel

        # requires_gard=false
        hpf_weight = nn.Parameter(torch.Tensor(filter_class_5).view(8, 1, 5, 5), requires_grad=False)
        # no padding
        self.hpf = nn.Conv2d(1, 8, kernel_size=5, padding=0, bias=False)
        self.hpf.weight = hpf_weight

        self.conv1 = torch.nn.Conv2d(in_channels=8, out_channels=outchannel, kernel_size=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(outchannel)
        self.tlu = TLU(self.threshold)

    def forward(self, image_in):
        output = self.hpf(image_in)
        output = self.conv1(output)
        output = self.bn(output)
        output = self.tlu(output)
        return output
