import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from feng.data import *

from feng.resnet import *
from feng.attnrnn import *
from feng.aae import *


######################################################
# Simple paired RNNs
######################################################


class PairRNN(MhcModel):

    def __init__(self, aa_channels, hidden_dim, n_layers, linear_dim, dropout_rnn, dropout_lin, dropout_inp_mhc=0, dropout_inp_pep=0, comment=""):
        super(PairRNN, self).__init__("pair_rnn", comment)
        self._main_pars["mhc_hid"] = hidden_dim
        self._main_pars["pep_hid"] = hidden_dim
        self._main_pars["lay"] = n_layers
        self._main_pars["lin_sizes"] = "-".join(map(lambda x: str(x), linear_dim))
        self._add_pars["dr_rnn"] = dropout_rnn
        self._add_pars["dr_dense"] = dropout_lin
        self._add_pars["dr_inp_mhc"] = dropout_inp_mhc
        self._add_pars["dr_inp_pep"] = dropout_inp_pep

        linear_dim = linear_dim[0]

        self._hidden_dim = hidden_dim

        # Dropout on input data
        self._drop_in_mhc = nn.Dropout(dropout_inp_mhc)
        self._drop_in_pep = nn.Dropout(dropout_inp_pep)

        self._mhc_rnn = nn.GRU(aa_channels, hidden_dim, n_layers, bidirectional=True, dropout=dropout_rnn, batch_first=True)
        self._pep_rnn = nn.GRU(aa_channels, hidden_dim, n_layers, bidirectional=True, dropout=dropout_rnn, batch_first=True)

        self._linear1 = nn.Linear(hidden_dim*2, linear_dim)
        self._drop1 = nn.Dropout(dropout_lin)
        self._final = nn.Linear(linear_dim, 1)

        print(self.name())

        # TODO: layer norm
        # TODO: weight norm


    def forward(self, input_mhc, input_pep):
        _, mhc = self._mhc_rnn(input_mhc)
        mhc = (mhc[-1] + mhc[-2]).mul_(.5).view((-1, self._hidden_dim))

        _, pep = self._pep_rnn(input_pep)
        pep = (pep[-1] + pep[-2]).mul_(.5).view((-1, self._hidden_dim))

        x = torch.cat([mhc, pep], 1)

        x = self._linear1(x)
        x = F.elu(x)
        x = self._drop1(x)

        x = self._final(x)
        x = F.elu(x)

        return x


    def init_weights(self):
        init.kaiming_uniform_(self._mhc_rnn.weight_ih_l0)
        init.kaiming_uniform_(self._mhc_rnn.weight_hh_l0)

        init.kaiming_uniform_(self._pep_rnn.weight_ih_l0)
        init.kaiming_uniform_(self._pep_rnn.weight_hh_l0)

        init.kaiming_uniform_(self._linear1.weight)
        init.kaiming_uniform_(self._final.weight)


######################################################
# Paired RNNs with attention
######################################################


class PairAttentiveRNN(nn.Module):

    def __init__(self):
        super(PairAttentiveRNN, self).__init__()
    

    def forward(self, input):
        pass        


#
# Slowly implement
#
# class EchoBlock(nn.Module):
#     def __init__(self, input_dim, other_dim, hidden_dim, n_layers, dropout_mem, dropout_attn, dropout_rnn, dropout_out, rnn_class="gru"):
#         super(ARCBlock, self).__init__()
        
#         self._rnn_class = rnn_class
#         if rnn_class == "gru":
#             rnn_class = nn.GRU
#         else:
#             rnn_class = nn.LSTM

#         self._hidden_dim = hidden_dim

#         self._mem = nn.Linear(hidden_dim * 2, hidden_dim)
#         self._mem_drop = nn.Dropout(dropout_mem)

#         self._attention = nn.Linear(hidden_dim*2 + my_dim[0]*my_dim[2], my_dim[0])
#         self._attention_cmb = nn.Linear(??? hidden_dim, hidden_dim)
#         self._attn_drop = nn.Dropout(dropout_attn)

#         self._rnn = rnn_class(input_dim, hidden_dim, n_layers, bidirectional=True, dropout=dropout_rnn)
#         self._rnn_drop = nn.Dropout(dropout_out)


#     def forward(self, input, h_my, h_other):
#         # Memory cell
#         h = torch.cat([h_my, h_other], 1)
#         h = self._mem(h)
#         h = F.elu(x)
#         h = self._mem_drop(h)

#         # Attention on the input data
#         ???
#         self._attn_drop
#         torch.cat([h_my, h_other, input.view((-1, my_dim[0] * my_dim[2]))])
#         attn_input = F.softmax(self._attention())
#         attn_input = torch.bmm(attn_input, input)

#         # Run through RNN
#         _, x = self._rnn(attn_input, h)
#         x = (x[-1] + x[-2]).mul_(.5).view((-1, self._hidden_dim))
#         x = F.elu(x)
#         x = self._rnn_drop(x)

#         return x


# class EchoComparator(nn.Module):
#     def __init__(self, n_blocks=5, rnn_class="gru"):
#         super(ARC, self).__init__()

#         self._n_blocks = n_blocks

#         self._mhc_in = self.make_input_block()
#         self._pep_in = self.make_input_block()

#         self._mhc_block = EchoBlock(rnn_class=rnn_class)
#         self._pep_block = EchoBlock(rnn_class=rnn_class)

#         self._final_block = self.make_final_block()
#         self._final_activation = nn.ELU()


#     def make_input_block(self, input_dim, dropout_inp, dropout_lin, hidden_dim):
#         layers = []
#         if dropout_inp:
#             layers.append(nn.Dropout(dropout_inp))
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ELU())
#         if dropout_lin:
#             layers.append(nn.Dropout(dropout_lin))
#         return nn.Sequential(*layers)


#     def make_final_block(self, prev_size, sizes, dropout):
#         layers = []
#         layers.append(nn.Linear(prev_size, sizes[0]))
#         layers.append(nn.ELU())
#         if dropout:
#             layers.append(nn.Dropout(dropout))
#         for i in range(1, len(sizes)):
#             layers.append(nn.Linear(sizes[i-1], sizes[i]))
#             layers.append(nn.ELU())
#             if dropout:
#                 layers.append(nn.Dropout(dropout))
#         layers.append(nn.Linear(sizes[-1], 1))
#         layers.append(nn.ELU())
#         return nn.Sequential(*layers)


#     def forward(self, input_mhc, input_pep):
#         h_mhc = self._mhc_in(input_mhc)
#         h_pep = self._pep_in(input_pep)

#         for i in range(self._n_blocks):
#             h_mhc = self._mhc_block(input_mhc, h_mhc, h_pep)
#             h_pep = self._pep_block(input_pep, h_pep, h_mhc)

#         x = torch.cat([h_mhc, h_pep], 1)
#         x = self._final_block(x)
#         return self._final_activation(x)


# TODO:
# Check ARC - with single RNN
# Check Echo - with two RNN