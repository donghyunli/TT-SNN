import sys
import tensorly as tl
from torch.autograd import Variable
from tensorly.decomposition import tensor_train
import numpy as np
import torch
import torch.nn as nn
import decompose.VBMF as VBMF
from models.MS_ResNet_TT import Snn_Conv2d, TT_Snn_Conv2d

tl.set_backend("pytorch")


def tt_decomposition_1(layer, rank):
    """Gets a conv layer,
    returns a nn.Sequential object with the CP decomposition.
    """

    W = layer.weight.data
    tt_cores = tensor_train(W.permute(1,2,3,0), rank=rank)
    ttCore1 = tt_cores.factors[0] # 1, in_channel, r1
    ttCore2 = tt_cores.factors[1] # r1, kW, r2
    ttCore3 = tt_cores.factors[2] # r2, kH, r3
    ttCore4 = tt_cores.factors[3] # r3, out_channel, 1

    TT_layer = TT_Snn_Conv2d(
        in_channels=ttCore1.shape[1],
        out_channels=ttCore4.shape[1],
        kernel_size=3,
        rank=rank,
        stride=layer.stride, 
        padding=1, 
        bias=None
    )

    TT_layer.first_layer.data = torch.permute(ttCore1, [2,1,0]).unsqueeze(-1)
    TT_layer.second_layer.data = torch.permute(ttCore2, [2,0,1]).unsqueeze(-1)
    TT_layer.third_layer.data = torch.permute(ttCore3, [2,0,1]).unsqueeze(2)
    TT_layer.fourth_layer.data = torch.permute(ttCore4, [1,0,2]).unsqueeze(-1)

    new_layers = [TT_layer]
    return nn.Sequential(*new_layers)


def tt_decomposition_2(layer, rank):
    """Gets a conv layer,
    returns a nn.Sequential object with the CP decomposition.
    """
    W = layer.weight.data

    tt_cores = tensor_train(W.permute(1,2,3,0), rank=rank)
    ttCore1 = tt_cores.factors[0] # 1, in_channel, r1
    ttCore2 = tt_cores.factors[1] # r1, kW, r2
    ttCore3 = tt_cores.factors[2] # r2, kH, r3
    ttCore4 = tt_cores.factors[3] # r3, out_channel, 1

    TT_layer = TT_Snn_Conv2d(
        in_channels=ttCore1.shape[1],
        out_channels=ttCore4.shape[1],
        kernel_size=3,
        rank=rank,
        padding=1, 
        bias=None
    )       

    TT_layer.first_layer.data = torch.permute(ttCore1, [2,1,0]).unsqueeze(-1)
    TT_layer.second_layer.data = torch.permute(ttCore2, [2,0,1]).unsqueeze(-1)
    TT_layer.third_layer.data = torch.permute(ttCore3, [2,0,1]).unsqueeze(2)
    TT_layer.fourth_layer.data = torch.permute(ttCore4, [1,0,2]).unsqueeze(-1)

    new_layers = [TT_layer]
    return nn.Sequential(*new_layers)


def estimate_ranks(layer):
    """Unfold the 2 modes of the Tensor the decomposition will
    be performed on, and estimates the ranks of the matrices using VBMF
    """
    print('decompose layer: ', layer)
    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0)
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks


def decompose_layer1(type, layer):
    """
    return the new decomposed conv2d layer
    """
    if type == "tt":
        ranks = estimate_ranks(layer)
        rank = ranks[0] if ranks[0] != 0 else ranks[1]
        new_layer = tt_decomposition_1(layer, rank)
    else:
        assert "only tt available"

    return new_layer, rank

def decompose_layer2(type, layer):
    """
    return the new decomposed conv2d layer
    """
    if type == "tt":
        ranks = estimate_ranks(layer)
        rank = ranks[0] if ranks[0] != 0 else ranks[1]
        new_layer = tt_decomposition_2(layer, rank)
    else:
        assert "only tt available"

    return new_layer, rank