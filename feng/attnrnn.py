import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from feng.mhcmodel import *


class AttnRNN(MhcModel):

    def __init__(self, pep_dim, hidden_dim, n_layers, aa_channels, lin_sizes, dropout_lin, dropout_rnn, dropout_inp, comment="simple_attn"):
        super(AttnRNN, self).__init__("attn_rnn", comment)
        self._main_pars["mhc_hid"] = hidden_dim
        self._main_pars["pep_hid"] = hidden_dim
        self._main_pars["lay"] = n_layers
        self._main_pars["lin_sizes"] = "-".join(map(lambda x: str(x), lin_sizes))
        self._add_pars["dr_rnn"] = dropout_rnn
        self._add_pars["dr_dense"] = dropout_lin
        self._add_pars["dr_inp_mhc"] = dropout_inp
        self._add_pars["dr_inp_pep"] = dropout_inp

        self.hidden_dim = hidden_dim
        self.pep_dim = pep_dim
        self.aa_channels = aa_channels

        # Dropout on input data
        self.drop_in_mhc = nn.Dropout(dropout_inp)
        self.drop_in_pep = nn.Dropout(dropout_inp)

        self.mhc_rnn = nn.GRU(aa_channels, hidden_dim, n_layers, bidirectional=True, dropout=dropout_rnn, batch_first=True)
        self.pep_rnn = nn.GRU(aa_channels, hidden_dim, n_layers, bidirectional=True, dropout=dropout_rnn, batch_first=True)

        self.attn = nn.Linear(pep_dim*aa_channels + hidden_dim, pep_dim)

        self.dense_block = self._make_dense_block(hidden_dim, lin_sizes, dropout_lin)

        if lin_sizes:
            prev_dim = lin_sizes[-1]
        else:
            prev_dim = self.hidden_dim
        
        self.final_layer = self._make_final_layer(prev_dim)

        self.init_weights()

        print(self.name())

        # TODO: layer norm
        # TODO: weight norm


    def forward(self, input_mhc, input_pep, mode="reg"):
        input_mhc = pack_padded_sequence(input_mhc[0], input_mhc[1], batch_first=True)
        input_pep, lens = input_pep

        # input dropout here

        _, mhc_h = self.mhc_rnn(input_mhc)
        mhc_h = (mhc_h[-1] + mhc_h[-2]).mul_(.5).view((-1, self.hidden_dim))

        attn_weights = F.softmax(self.attn(torch.cat([input_pep.view((-1, self.pep_dim * self.aa_channels)), mhc_h], 1)), dim=1)
        attn_weights = attn_weights.unsqueeze(2).expand((-1, -1, attn_weights.size(1)))

        attn_applied = torch.bmm(attn_weights, input_pep)
        attn_applied = pack_padded_sequence(attn_applied, lens, batch_first=True)
        _, pep_h = self.pep_rnn(attn_applied)
        pep_h = (pep_h[-1] + pep_h[-2]).mul_(.5).view((-1, self.hidden_dim))

        x = self.dense_block(pep_h)

        return self.final_layer[mode](x)


    def get_embeddings(self, input_mhc, input_pep):
        input_mhc = pack_padded_sequence(input_mhc[0], input_mhc[1], batch_first=True)
        input_pep, lens = input_pep

        # input dropout here

        _, mhc_h = self.mhc_rnn(input_mhc)
        mhc_h = (mhc_h[-1] + mhc_h[-2]).mul_(.5).view((-1, self.hidden_dim))

        attn_weights = F.softmax(self.attn(torch.cat([input_pep.view((-1, self.pep_dim * self.aa_channels)), mhc_h], 1)), dim=1)
        attn_weights = attn_weights.unsqueeze(2).expand((-1, -1, attn_weights.size(1)))
        # TODO: check if multiplications / dimensions are OK
        attn_applied = torch.bmm(attn_weights, input_pep)
        attn_applied = pack_padded_sequence(attn_applied, lens, batch_first=True)
        _, pep_h = self.pep_rnn(attn_applied)
        pep_h = (pep_h[-1] + pep_h[-2]).mul_(.5).view((-1, self.hidden_dim))

        x = self.dense_block(pep_h)

        return {"mhc": mhc_h, "pep": pep_h, "lin": x, "attn": attn_weights}


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


    def init_weights(self):
        init.kaiming_uniform_(self.mhc_rnn.weight_ih_l0)
        init.kaiming_uniform_(self.mhc_rnn.weight_hh_l0)

        init.kaiming_uniform_(self.pep_rnn.weight_ih_l0)
        init.kaiming_uniform_(self.pep_rnn.weight_hh_l0)

        init.kaiming_uniform_(self.attn.weight)
        init.kaiming_uniform_(self.attn.weight)