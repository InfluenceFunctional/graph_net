'''Import statements'''
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric.nn as gnn
from CustomSchNet import CustomSchNet
import sys
import itertools


class molecule_graph_model(nn.Module):
    def __init__(self, config, dataDims, return_latent=False):
        super(molecule_graph_model, self).__init__()
        # initialize constants and layers
        self.return_latent = return_latent
        self.activation = config.activation
        self.num_fc_layers = config.num_fc_layers
        self.fc_depth = config.fc_depth
        self.graph_model = config.graph_model
        self.classes = dataDims['output classes']
        self.graph_filters = config.graph_filters
        self.n_mol_feats = dataDims['mol features']
        self.n_atom_feats = dataDims['atom features']
        if not config.concat_mol_to_atom_features:  # if we are not adding molwise feats to atoms, subtract the dimension
            self.n_atom_feats -= self.n_mol_feats
        self.pool_type = config.pooling
        self.fc_norm_mode = config.fc_norm_mode
        self.embedding_dim = config.atom_embedding_size

        torch.manual_seed(config.model_seed)

        if config.graph_model == 'schnet':
            self.graph_net = CustomSchNet(
                hidden_channels=config.fc_depth,
                num_filters=config.graph_filters,
                num_interactions=config.graph_convolution_layers,
                num_gaussians=config.num_radial,
                cutoff=config.graph_convolution_cutoff,
                max_num_neighbors=config.max_num_neighbors,
                readout='mean',
                num_atom_features=self.n_atom_feats,
                embedding_hidden_dimension=config.atom_embedding_size,
                atom_embedding_dims=dataDims['atom embedding dict sizes'],
                norm=config.graph_norm,
                dropout=config.fc_dropout_probability
            )
        else:
            print(config.graph_model + ' is not a valid graph model!!')
            sys.exit()


    def forward(self, data):
        x = data.x
        pos = data.pos
        x = self.graph_net(x[:, :self.n_atom_feats], pos, data.batch)  # get atoms encoding

        return x # output atom-wise feature vectors


class kernelActivation(nn.Module):  # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[-2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1, 1), groups=int(channels), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        # y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        # for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
        super().__init__()
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self, input):
        return self.norm(input)
