from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_any_t
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
# Model for MS-ResNet

thresh = 0.5  # 0.5 # neuronal threshold
lens = 0.5  # 0.5 # hyper-parameters of approximate function
decay = 0.25  # 0.25 # decay constants
time_window = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update


class mem_update(nn.Module):
    def __init__(self):
        super(mem_update, self).__init__()

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(device)
        spike = torch.zeros_like(x[0]).to(device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1-spike.detach()) + x[i]
            else:
                mem = x[i]
            spike = act_fun(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__()
        self.bn = BatchNorm3d1(num_features) 

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2) 


class batch_norm_2d1(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)
            nn.init.zeros_(self.bias)


class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, 0.2*thresh)
            nn.init.zeros_(self.bias)

class Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
        weight = self.weight
        h = (input.size()[3]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        w = (input.size()[4]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        for i in range(time_window):
            b = time.time()
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return c1
    

class Recon_Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Recon_Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker
        self.pool = nn.MaxPool2d(2,stride=2)

    def forward(self, input):
        weight = self.weight         
        
        h = (input.size()[3]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        w = (input.size()[4]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        if self.in_channels != self.out_channels:
            h = int(h/2)
            w = int(w/2)
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        for i in range(time_window):
            out = F.conv2d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.in_channels != self.out_channels:
                c1[i] = self.pool(out)
            else:
                c1[i] = out
        return c1


######################################################################################################################
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            mem_update(),
            Snn_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            batch_norm_2d(out_channels),
            mem_update(),
            Snn_Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            batch_norm_2d1(out_channels),
            )
        # shortcut
        self.shortcut = nn.Sequential(
            )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super().__init__()
        nChannels = [64, 64, 128, 256, 512]
        self.in_channels = nChannels[0]
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, nChannels[0], kernel_size=3, padding=1, bias=False),
            batch_norm_2d(nChannels[0]))
        self.mem_update = mem_update()
        self.stage1 = self._make_layer(block, nChannels[1], num_block[0], 1)
        self.stage2 = self._make_layer(block, nChannels[2], num_block[1], 2)
        self.stage3 = self._make_layer(block, nChannels[3], num_block[2], 2)
        self.stage4 = self._make_layer(block, nChannels[4], num_block[3], 2)
        self.fc = nn.Linear(nChannels[-1], num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        input = torch.zeros(time_window, x.size()[0], 3, x.size()[2], x.size()[3], device=device)
        for i in range(time_window):
            input[i] = x
        # input = x
        output = self.conv1(input)
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = self.mem_update(output)
        output = F.adaptive_avg_pool3d(output, (None, 1, 1))
        output = output.view(output.size()[0], output.size()[1], -1)
        output = output.sum(dim=0)/output.size()[0]
        output = self.fc(output)
        return output


def resnet18(num_class):
    return ResNet_CIFAR(BasicBlock, [2, 2, 2, 2], num_class)

def resnet34(num_class):
    return ResNet_CIFAR(BasicBlock, [3, 4, 6, 3], num_class)