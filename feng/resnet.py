import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from feng.mhcmodel import *


######################################################
# CNN 1D ResNet
######################################################

class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout=None):
        super(BottleneckBlock, self).__init__()

        self.dropout_pr = dropout
        if dropout:
            self.drop1 = nn.Dropout(self.dropout_pr)
            self.drop2 = nn.Dropout(self.dropout_pr)

        # resblock and resblock + bn
        self.conv1 = nn.Conv1d(in_channels, in_channels // 4, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm1d(in_channels // 4)

        self.conv2 = nn.Conv1d(in_channels // 4, in_channels // 4, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels // 4)

        self.conv3 = nn.Conv1d(in_channels // 4, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # self.activation = F.rrelu

        self.init_weights()


    def forward(self, input):
        residual = input

        x = self.conv1(input)
        x = F.rrelu(self.bn1(x))
        if self.dropout_pr:
            x = self.drop1(x)

        x = self.conv2(x)
        x = F.rrelu(self.bn2(x))
        if self.dropout_pr:
            x = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += residual
        x = F.rrelu(x)

        return x


    def init_weights(self):
        init.kaiming_uniform_(self.conv1.weight)
        init.kaiming_uniform_(self.conv2.weight)
        init.kaiming_uniform_(self.conv3.weight)


class ResNet(MhcModel):
    
    def __init__(self, n_filters, mhc_dim, pep_dim, mhc_blocks, pep_blocks, aa_channels, kernel_size, lin_sizes, dropout_lin, dropout_conv, dropout_inp, comment="elu_block"):
        super(ResNet, self).__init__("resnet_1d", comment)
        self._main_pars["mhc_blocks"] = mhc_blocks
        self._main_pars["pep_blocks"] = pep_blocks
        self._main_pars["kernel"] = kernel_size
        self._main_pars["lin_sizes"] = "-".join(map(lambda x: str(x), lin_sizes))
        self._add_pars["dr_dense"] = dropout_lin
        self._add_pars["dr_conv"] = dropout_conv
        self._add_pars["dr_input"] = dropout_inp

        # self.bottleneck_dim = 8
        self.alphabet_size = 21

        self.mhc_emb = nn.Embedding(self.alphabet_size, aa_channels)
        self.pep_emb = nn.Embedding(self.alphabet_size, aa_channels)
        self.mhc_branch = self._make_branch(mhc_dim, mhc_blocks, aa_channels, n_filters, kernel_size, dropout_inp, dropout_conv)
        self.pep_branch = self._make_branch(pep_dim, pep_blocks, aa_channels, n_filters, kernel_size, dropout_inp, dropout_conv)
        self.dense_block = self._make_dense_block(mhc_dim*n_filters + pep_dim*n_filters, lin_sizes, dropout_lin)
        # self.dense_block = self._make_dense_block(self.bottleneck_dim*n_filters*2, lin_sizes, dropout_lin)

        if lin_sizes:
            prev_dim = lin_sizes[-1]
        else:
            prev_dim = mhc_dim*n_filters + pep_dim*n_filters
        
        self.final_layer = self._make_final_layer(prev_dim)


    def forward(self, input_mhc, input_pep, mode="reg"):
        x_mhc = self.mhc_emb(input_mhc.squeeze(1).long()).transpose(1, 2)
        x_mhc = self.mhc_branch(x_mhc)

        x_pep = self.pep_emb(input_pep.squeeze(1).long()).transpose(1, 2)
        x_pep = self.pep_branch(x_pep)

        # x_mhc = self.mhc_branch(input_mhc)
        # x_pep = self.pep_branch(input_pep)

        x = torch.cat([x_mhc.view((x_mhc.size(0), -1)), x_pep.view((x_pep.size(0), -1))], 1)
        x = self.dense_block(x)
        return self.final_layer[mode](x)


    def get_embeddings(self, input_mhc, input_pep):
        x_mhc = self.mhc_branch(input_mhc)
        x_pep = self.pep_branch(input_pep)
        x = torch.cat([x_mhc.view((x_mhc.size(0), -1)), x_pep.view((x_pep.size(0), -1))], 1)
        x = self.dense_block(x)
        return {"mhc": x_mhc.view((x_mhc.size(0), -1)).data, "pep": x_pep.view((x_pep.size(0), -1)).data, "lin": x.data}


    def _make_branch(self, block_dim, n_blocks, aa_channels, n_filters, kernel_size, drop_in, drop_conv):
        layers = []

        if drop_in:
            layers.append(nn.Dropout(drop_in))
        layers.append(nn.Conv1d(aa_channels, n_filters, kernel_size=3, padding=1))
        init.kaiming_uniform_(layers[-1].weight)

        layers.append(nn.BatchNorm1d(n_filters))
        layers.append(nn.RReLU())
        layers.append(nn.Dropout(drop_conv))

        for i in range(n_blocks):
            layers.append(BottleneckBlock(n_filters, n_filters, kernel_size, drop_conv))

        layers.append(nn.BatchNorm1d(n_filters))
        layers.append(nn.RReLU())
        layers.append(nn.Dropout(drop_conv))

        # layers.append(nn.AdaptiveMaxPool1d(self.bottleneck_dim))

        return nn.Sequential(*layers)


    def _make_dense_block(self, prev_size, sizes, dropout):
        def add_block(layers, prev_size, next_size, dropout):
            layers.append(nn.Linear(prev_size, next_size))
            init.kaiming_uniform_(layers[-1].weight)

            # layers.append(nn.ELU())
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(nn.RReLU())

            if dropout:
                layers.append(nn.Dropout(dropout))

        layers = []

        add_block(layers, prev_size, sizes[0], dropout)

        for i in range(1, len(sizes)):
            add_block(layers, sizes[i-1], sizes[i], dropout)

        return nn.Sequential(*layers)


    def _make_final_layer(self, prev_dim):
        final_layer = {}

        layers = []
        layers.append(nn.Linear(prev_dim, 1))
        init.kaiming_uniform_(layers[-1].weight)
        layers.append(nn.LeakyReLU())
        final_layer["reg"] = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Linear(prev_dim, 1))
        init.kaiming_uniform_(layers[-1].weight)
        layers.append(nn.Sigmoid())
        final_layer["clf"] = nn.Sequential(*layers)

        return final_layer