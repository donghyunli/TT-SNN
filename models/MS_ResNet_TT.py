import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import sys
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

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)


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


class PTT_Snn_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None, padding_mode='zeros', marker='b'):
        super(PTT_Snn_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.marker = marker
        self.rank = rank
        
        if in_channels != out_channels:
            stride_ = 1
        else:
            stride_ = 1

        self.first_layer = nn.Conv2d(in_channels, rank, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        self.second_layer = nn.Conv2d(rank, rank, kernel_size=(3,1), stride=(stride_,stride_), padding='same', bias=False)
        self.third_layer = nn.Conv2d(rank, rank, kernel_size=(1,3), stride=(stride_,stride_), padding=(0,1), bias=False)
        self.fourth_layer = nn.Conv2d(rank, out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.first_layer.apply(init_weights)
        self.second_layer.apply(init_weights)
        self.third_layer.apply(init_weights)
        self.fourth_layer.apply(init_weights)


    def forward(self, input):
        if type(self.stride) == tuple:
            self.stride = self.stride[0]
        h = (input.size()[3]-self.kernel_size+2*self.padding)//self.stride+1
        w = (input.size()[4]-self.kernel_size+2*self.padding)//self.stride+1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        
        for i in range(time_window):
            out1 = self.first_layer(input[i])
            out2_1 = self.second_layer(out1)
            out2_2 = self.third_layer(out1)
            out2 = (out2_1 + out2_2) / 2  
            out3 = self.fourth_layer(out2)
            if self.in_channels != self.out_channels:
                out3 = self.pool(out3)
            c1[i] = out3
        return c1
    

class STT_Snn_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None, padding_mode='zeros', marker='b'):
        super(STT_Snn_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.marker = marker
        self.rank = rank

        if in_channels != out_channels:
            stride_ = 1
        else:
            stride_ = 1

        self.first_layer = nn.Conv2d(in_channels, rank, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        self.second_layer = nn.Conv2d(rank, rank, kernel_size=(3,1), stride=(stride_,stride_), padding=(1,0), bias=False)
        self.third_layer = nn.Conv2d(rank, rank, kernel_size=(1,3), stride=(stride_,stride_), padding=(0,1), bias=False)
        self.fourth_layer = nn.Conv2d(rank, out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.first_layer.apply(init_weights)
        self.second_layer.apply(init_weights)
        self.third_layer.apply(init_weights)
        self.fourth_layer.apply(init_weights)

    def forward(self, input):
        if type(self.stride) == tuple:
            self.stride = self.stride[0]
        h = (input.size()[3]-self.kernel_size+2*self.padding)//self.stride+1
        w = (input.size()[4]-self.kernel_size+2*self.padding)//self.stride+1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)

        for i in range(time_window):

            out1 = self.first_layer(input[i])           
            out2 = self.second_layer(out1)
            out3 = self.third_layer(out2)
            out4 = self.fourth_layer(out3)
            if self.in_channels != self.out_channels:
                out4 = self.pool(out4)
            c1[i] = out4

        return c1
    

class HTT_Snn_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=None, padding_mode='zeros', marker='b'):
        super(HTT_Snn_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.marker = marker
        self.rank = rank

        if in_channels != out_channels:
            stride_ = 1
        else:
            stride_ = 1

        self.first_layer = nn.Conv2d(in_channels, rank, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        self.second_layer = nn.Conv2d(rank, rank, kernel_size=(3,1), stride=(stride_,stride_), padding=(1,0), bias=False)
        self.third_layer = nn.Conv2d(rank, rank, kernel_size=(1,3), stride=(stride_,stride_), padding=(0,1), bias=False)
        self.fourth_layer = nn.Conv2d(rank, out_channels, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=False)
        # self.extra = nn.Conv2d(rank, rank, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.first_layer.apply(init_weights)
        self.second_layer.apply(init_weights)
        self.third_layer.apply(init_weights)
        self.fourth_layer.apply(init_weights)   

    def forward(self, input): 
        if type(self.stride) == tuple:
            self.stride = self.stride[0]
        h = (input.size()[3]-self.kernel_size+2*self.padding)//self.stride+1
        w = (input.size()[4]-self.kernel_size+2*self.padding)//self.stride+1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        
        
        for i in range(time_window):
            if i == 0 or i == 1: 
                out1 = self.first_layer(input[i])
                out2_1 = self.second_layer(out1)
                out2_2 = self.third_layer(out1)
                out2 = (out2_1 + out2_2) / 2  
                out3 = self.fourth_layer(out2)
                if self.in_channels != self.out_channels:
                    out3 = self.pool(out3)
                c1[i] = out3
            
            elif i == 2 or i == 3:  
                out1 = self.first_layer(input[i])
                out3 = self.fourth_layer(out1)
                if self.in_channels != self.out_channels:
                    out3 = self.pool(out3)
                c1[i] = out3
            
            else:
                assert "Timestep error!"

        return c1

######################################################################################################################
class BasicBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, rank, stride=1):
        super().__init__()
        rank1 = rank[0]
        rank2 = rank[1]
        print('rank1:' , rank1)
        print('rank2:' , rank2)
        if args.tt_mode == 'STT':
            self.residual_function = nn.Sequential(
                mem_update(),
                STT_Snn_Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, rank=rank1, stride=stride, padding=1, bias=None),
                batch_norm_2d(out_channels),
                mem_update(),
                STT_Snn_Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, rank=rank2, padding=1, bias=None),
                batch_norm_2d1(out_channels),
                )
        if args.tt_mode == 'PTT':
            self.residual_function = nn.Sequential(
                mem_update(),
                PTT_Snn_Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, rank=rank1, stride=stride, padding=1, bias=None),
                batch_norm_2d(out_channels),
                mem_update(),
                PTT_Snn_Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, rank=rank2, padding=1, bias=None),
                batch_norm_2d1(out_channels),
                )
        if args.tt_mode == 'HTT':
            self.residual_function = nn.Sequential(
                mem_update(),
                HTT_Snn_Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, rank=rank1, stride=stride, padding=1, bias=None),
                batch_norm_2d(out_channels),
                mem_update(),
                HTT_Snn_Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, rank=rank2, padding=1, bias=None),
                batch_norm_2d1(out_channels),
                )
        # shortcut
        self.shortcut = nn.Sequential(
            )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=None),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_block, args, rank_list, num_classes):
        super().__init__()
        nChannels = [64, 64, 128, 256, 512]
        self.in_channels = nChannels[0]
        self.conv1 = nn.Sequential(
            Snn_Conv2d(3, nChannels[0], kernel_size=3, padding=1, bias=False),
            batch_norm_2d(nChannels[0]))
        self.mem_update = mem_update()
        if len(rank_list) == 16: # for ResNet18
            stage1_rank = rank_list[0:4]
            stage2_rank = rank_list[4:8]
            stage3_rank = rank_list[8:12]
            stage4_rank = rank_list[12:16]
        elif len(rank_list) == 32: # for ResNet34
            stage1_rank = rank_list[0:6]
            stage2_rank = rank_list[6:14]
            stage3_rank = rank_list[14:26]
            stage4_rank = rank_list[26:32]
        self.stage1 = self._make_layer(block, args, nChannels[1], num_block[0], 1, stage1_rank)
        self.stage2 = self._make_layer(block, args, nChannels[2], num_block[1], 2, stage2_rank)
        self.stage3 = self._make_layer(block, args, nChannels[3], num_block[2], 2, stage3_rank)
        self.stage4 = self._make_layer(block, args, nChannels[4], num_block[3], 2, stage4_rank)
        self.fc = nn.Linear(nChannels[-1], num_classes)

    def _make_layer(self, block, args, out_channels, num_blocks, stride, rankList):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        i = 0
        for stride in strides:
            rank = rankList[i:i+2]
            layers.append(block(args, self.in_channels, out_channels, rank, stride))
            self.in_channels = out_channels
            i = i + 2
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


def tt_resnet18(args, rank_list, num_class):
    return ResNet_CIFAR(BasicBlock, [2, 2, 2, 2], args, rank_list, num_class)

def tt_resnet34(args, rank_list, num_class):
    return ResNet_CIFAR(BasicBlock, [3, 4, 6, 3], args, rank_list, num_class)
