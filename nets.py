import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from torch.distributions import Normal, Uniform, Bernoulli, Laplace
import torch.nn as nn
import torch.optim as optim
import torch.sparse
from math import sqrt
from tqdm import tqdm

import copy
from torch import squeeze, unsqueeze
import lib.layers.diffeq_layers as diffeq_layers


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        time = t.view(1, 1).to(x.device)
        return self._layer(x) * torch.sigmoid(self._hyper_gate(time)) \
            + self._hyper_bias(time)


class ConcatSquashLinearSparse(nn.Module):
    def __init__(self, dim_in, dim_out, adjacency, device):
        super(ConcatSquashLinearSparse, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self._adjacency = adjacency
        _weight_mask = torch.zeros([dim_out, dim_in])
        for [a, b] in adjacency:
            if b < dim_in:
                _weight_mask[a, b] = 1.0
        print("weight_mask: ")
        print(_weight_mask)
        self._weight_mask = _weight_mask.to(device)

        lin = nn.Linear(dim_in, dim_out)
        self._weights = lin.weight
        self._bias = lin.bias

        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        w = torch.mul(self._weight_mask, self._weights)
        res = torch.addmm(self._bias, x, w.transpose(0,1))

        return res * torch.sigmoid(self._hyper_gate(t.view(1, 1))) \
            + self._hyper_bias(t.view(1, 1))


class LinearSparse(nn.Module):
    def __init__(self, dim_in, dim_out, adjacency, device):
        super(LinearSparse, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        self._adjacency = adjacency
        _weight_mask = torch.zeros([dim_out, dim_in])
        #weight_rescale = dim_out*(dim_in + dim_out)/len(adjacency)
        for [a, b] in adjacency:
            if b < dim_in:
                _weight_mask[a, b] = 1.0 # weight_rescale
        print("weight_mask: ")
        print(_weight_mask)

        # replicate x dimensions for dx part
        self._weight_mask = _weight_mask

        lin = nn.Linear(dim_in, dim_out)
        self._weights = lin.weight
        self._bias = lin.bias


    def forward(self, x):
        w = torch.mul(self._weight_mask, self._weights)
        res = torch.addmm(self._bias, x, w.transpose(0,1))

        return res


class AdaptedODENet(nn.Module):
    def __init__(self, dims, conditional_dims, num_layers=4):
        super(AdaptedODENet, self).__init__()
        self.num_squeeze=0
        layers = [ConcatSquashLinear(dims + conditional_dims + 1, dims)] + [ConcatSquashLinear(dims, dims) for _ in range(num_layers-1)]
        activation_fns = [nn.Tanh() for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.record = []
        self.is_recording = False

    def forward(self, t, x):
        batch_dim = x.shape[0]
        #dx = torch.cat([x, self.conditioned], dim=1)
        dx = torch.cat([x, self.conditioned, t * torch.ones([batch_dim, 1]).to(x.device)], dim=1)
        for l, layer in enumerate(self.layers):
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                acti = layer(t, dx)
                if l == 0:
                    dx = self.activation_fns[l](acti)
                else:
                    dx = self.activation_fns[l](acti) + dx
            else:
                dx = layer(t, dx)

        if self.is_recording:
            self.record.append((t, dx.clone().detach(), x.clone().detach()))
        return dx



class SparseODENet(nn.Module):
    def __init__(self, dims, conditional_dims, full_adjacency, device, num_layers=4):
        super(SparseODENet, self).__init__()
        self.num_squeeze=0
        layers = [ConcatSquashLinearSparse(dims + conditional_dims + 1, dims, full_adjacency, device)] + [ConcatSquashLinearSparse(dims, dims, full_adjacency, device) for _ in range(num_layers-1)]
        activation_fns = [nn.Tanh() for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.record = []
        self.is_recording = False

    def forward(self, t, x):
        batch_dim = x.shape[0]
        #dx = torch.zeros(x.shape).to(x)
        dx = torch.cat([x, self.conditioned, t * torch.ones([batch_dim, 1]).to(x.device)], dim=1)
        for l, layer in enumerate(self.layers):
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                acti = layer(t, dx)
                if l == 0:
                    dx = self.activation_fns[l](acti)
                else:
                    dx = self.activation_fns[l](acti) + dx
            else:
                dx = layer(t, dx)

        if self.is_recording:
            self.record.append((t, dx.clone().detach(), x.clone().detach()))
        return dx


################################################
# ffjord network
################################################

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
}



class ODENet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(
            self, hidden_dims, input_shape, strides, conv, layer_type="concat", nonlinearity="softplus", num_squeeze=0, conditional_dims=0):
        super(ODENet, self).__init__()
        self.num_squeeze = num_squeeze
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "squash": diffeq_layers.SquashConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "concat_v2": diffeq_layers.ConcatConv2d_v2,
                "concatsquash": diffeq_layers.ConcatSquashConv2d,
                "blend": diffeq_layers.BlendConv2d,
                "concatcoord": diffeq_layers.ConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "squash": diffeq_layers.SquashLinear,
                "concat": diffeq_layers.ConcatLinear,
                "concat_v2": diffeq_layers.ConcatLinear_v2,
                "concatsquash": diffeq_layers.ConcatSquashLinear,
                "blend": diffeq_layers.BlendLinear,
                "concatcoord": diffeq_layers.ConcatLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        first = True
        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0] + conditional_dims if first else hidden_shape[0],
                               dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2
            first = False

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = torch.cat([y, self.conditioned], dim=1)
        # squeeze
        for _ in range(self.num_squeeze):
            dx = squeeze(dx, 2)
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # unsqueeze
        for _ in range(self.num_squeeze):
            dx = unsqueeze(dx, 2)
        return dx

