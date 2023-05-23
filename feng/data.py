import re
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset


ALPHABET = np.array(list("ALRKNMDFCPQSETGWHYIV"), dtype=str)

unify_alleles = lambda x: re.sub('[*|:|-]', '', x)

# 224 ~ 0.5
BIND_THR = 1 - (np.log(500) / np.log(50000))

USE_CUDA = torch.cuda.is_available()

binarize_aff = lambda x: 1 if x >= BIND_THR else 0

unlog_aff = lambda x: np.exp((1 - x) * np.log(50000))

log_aff = lambda x: 1 - (np.log(np.clip(x, 1, 50000)) / np.log(50000))

# PSEUDO_SEQ_FILE = "/pseudoseqs.csv.gz"
# PSEUDO_SEQ_FILE = "/pseudoseqs_hs.csv"
# PSEUDO_SEQ_FILE = "/pseudoseqs_freq.csv"
# PSEUDO_SEQ_FILE = "/pseudoseqs_couple.csv"


def log_meas(meas_values):
    meas_values = np.clip(meas_values, 1, 50000)
    return 1 - (np.log(meas_values) / np.log(50000))


def compute_time(fun, *args, **kwargs):
    start = time.time()
    res = fun(*args, **kwargs)
    end = time.time()
    return res, end - start


def clean_mhc_df(mhc_df_path, mhcs, log=True, lens=[8, 9, 10, 11, 12, 13, 14, 15]):
    mhc_df = pd.read_csv(mhc_df_path)#, sep="\t")
    mhc_df.meas = log_meas(mhc_df.meas) if log else mhc_df.meas
    mhc_df.mhc = mhc_df.mhc.apply(unify_alleles)
    mhc_df["lenghts"] = np.array([len(i) for i in mhc_df.sequence.values])
    pred_len = len(mhc_df)
    bad_mhc = pred_len - mhc_df.mhc.isin(mhcs).sum()
    bad_len = pred_len - mhc_df["lenghts"].isin(lens).sum()
    mhc_df = mhc_df[(mhc_df.mhc.isin(mhcs)) & (mhc_df["lenghts"].isin(lens))]
    return mhc_df[["mhc", "sequence", "meas"]].reset_index(drop=True), pred_len, bad_mhc, bad_len


def embed_pseudo(pseudo_path, emb_path):
    aa_model = np.load(emb_path)

    pseudo_df = pd.read_csv(pseudo_path)
    pseudo_df.mhc = pseudo_df.mhc.apply(unify_alleles)
    pseudo_dict = {}
    for i, seq in enumerate(pseudo_df.pseudo):
        acid_embs = np.array([aa_model[char] for char in seq])
        pseudo_dict[pseudo_df.mhc[i]] = np.array(acid_embs)
    return pseudo_dict


def load_pseudo(pseudo_path, emb_path):
    print("Embedding file:", emb_path)
    print("Loading pseudosequences...")
    pseudo_mhc = embed_pseudo(pseudo_path, emb_path)
    print("Number of pseudo MHC: {}".format(len(pseudo_mhc)))
    return pseudo_mhc


def clean_embs2(emb_path, sequences, max_len, pad_char=None, pad_centre=True):
    aa_model = np.load(emb_path)

    seq_data = np.zeros((len(sequences), max_len, len(aa_model['A'])))
    for i, seq in enumerate(sequences):
        if not pad_centre:
            seq_data[i, :len(seq), :] = np.array([aa_model[char] for char in seq])
            if pad_char and len(seq) != max_len:
                seq_data[i, len(seq):, :] = np.array([aa_model[pad_char] for _ in range(len(seq), max_len)])
        else:
            left_pos, right_pos = len(seq) // 2, len(seq) - (len(seq) // 2)
            seq_data[i, :left_pos, :] = np.array([aa_model[char] for char in seq[:left_pos]])
            seq_data[i, -right_pos:, :] = np.array([aa_model[char] for char in seq[-right_pos:]])
            if pad_char and len(seq) != max_len:
                seq_data[i, left_pos:-right_pos, :] = np.array([aa_model[pad_char] for _ in range(len(seq), max_len)])
    return seq_data


def get_embeddings2(emb_path, mhc_df, pseudo_dict, max_len, pad_char):
    pep_data = clean_embs2(emb_path, mhc_df["sequence"], max_len, pad_char)
    mhc_data = np.array([pseudo_dict[mhc] for mhc in mhc_df["mhc"]])
    return mhc_data, pep_data, [len(x) for x in mhc_df["sequence"]], np.array(mhc_df.meas)


def load_iedb(nn_mode, filename, emb_path, pseudo_sequences, min_len=8, max_len=15, filter_unused=True, head=None, pad_char=None): 
    """

    Arguments:
        min_len: Filter out peptide sequences with length lesser than that.
        max_len: Filter out peptide sequences with length greater than that.
        filter_unused: If True remove pseudosequences from synthetic batch generation 
        that are not used in the input Dataset. Default: True.
    """
    # folderpath = folderpath + "/curated.csv.gz"
    # folderpath = folderpath + "/bdata2009_curated.csv.gz"
    folderpath = filename

    ps_mhcs = list(pseudo_sequences.keys())

    print("File:", folderpath)
    print("Loading the data...", end="\t")
    (train_df, orig_len, bad_mhc, bad_len), sec = compute_time(clean_mhc_df, folderpath, ps_mhcs, log=True, lens=[i for i in range(min_len, max_len+1)])
    if head:
        train_df = train_df.head(head)
    print("Done in", round(sec, 2), "s")
    print("  Source size:    {}\n  #Unknown MHC:   {}\n  #Wrong lengths: {}".format(orig_len, bad_mhc, bad_len))

    print("Creating embeddings...", end="\t")
    (mhc_data, pep_data, pep_lengths, affinity_data), sec = compute_time(get_embeddings2, emb_path, train_df, pseudo_sequences, max_len, pad_char) # 32 sec
    print("Done in", round(sec, 2), "s")

    print("Shape of the data:")
    print("  MHC embeddings:     {}".format(mhc_data.shape))
    print("  peptide embeddings: {}".format(pep_data.shape))
    print("  binding affinities: {}".format(affinity_data.shape))

    dataset = MhcDataset(mhc_data, pep_data, pep_lengths, affinity_data, nn_mode)
    # return mhc_data, pep_data, pep_lengths, affinity_data, dataset
    return dataset


def process_allele(args):
    nn_mode, df, pseudo_sequences, max_len, emb_path, pad_char = args
    (mhc_data, pep_data, pep_lengths, classes_data), sec = compute_time(get_embeddings2, emb_path, df, pseudo_sequences, max_len, pad_char)
    # mhc_data = torch.from_numpy(mhc_data).float()
    # pep_data = torch.from_numpy(pep_data).float()
    # classes_data = torch.from_numpy(classes_data).float()
    # pep_lengths = np.array(pep_lengths)
    return MhcDataset(mhc_data, pep_data, pep_lengths, classes_data, nn_mode)


def load_abelin(nn_mode, folderpath, emb_path, pseudo_sequences, min_len=8, max_len=15, num_workers=mp.cpu_count(), head=None, pad_char=None):
    folderpath = folderpath

    ps_mhcs = list(pseudo_sequences.keys())

    print("File:", folderpath)
    print("Loading the data...", end="\t")
    (abelin_df, orig_len, bad_mhc, bad_len), sec = compute_time(clean_mhc_df, folderpath, ps_mhcs, log=False, lens=[i for i in range(min_len, max_len+1)])
    if head:
        abelin_df = abelin_df.head(head)
    print("Done in", round(sec, 2), "s")
    print("  Source size:    {}\n  #Unknown MHC:   {}\n  #Wrong lengths: {}".format(orig_len, bad_mhc, bad_len))

    print("Number of rows: {}".format(len(abelin_df)))
    print("Unique HLAs:    {}".format(len(abelin_df["mhc"].unique())))

    print("Creating embeddings...", end="\t")
    sec_start = time.time()
    abelin_dataset = {}

    with mp.Pool(num_workers) as pool:
        res = pool.map(process_allele, [(nn_mode, abelin_df.loc[abelin_df["mhc"] == allele, :], pseudo_sequences, max_len, emb_path, pad_char) for allele in abelin_df["mhc"].unique()])
        for i, allele in enumerate(abelin_df["mhc"].unique()):
            abelin_dataset[allele] = res[i]

    print("Done in", round(time.time() - sec_start, 2), "s")

    return abelin_dataset



class MhcDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, mhc_array, pep_array, pep_lengths, affinities, mode=["cnn", "rnn"][0], tensors=False):
        # CNN: batch - channels - length
        # RNN: batch - length - channels
        super(MhcDataset, self).__init__()

        assert mode in ["cnn", "rnn"]

        self.mode = mode

        # TODO: make good constructor, not this "thing"
        if not tensors:
            assert mhc_array.shape[0] == pep_array.shape[0] == len(pep_lengths) == len(affinities)

            # Sort by peptide lengths in reversed order
            # in order to support slicing for RNN
            sorted_inds = np.flip(np.argsort(pep_lengths), axis=0)
            mhc_array = mhc_array[sorted_inds]
            pep_array = pep_array[sorted_inds]
            pep_lengths = np.array(pep_lengths, dtype=np.int16)[sorted_inds]
            affinities = affinities[sorted_inds]

            mhc_tensor = torch.from_numpy(mhc_array).float()
            pep_tensor = torch.from_numpy(pep_array).float()
            aff_tensor = torch.from_numpy(affinities).float()

            self.len = pep_lengths
            self.aff = aff_tensor
            # self.aff_backup = aff_tensor.clone()
            if self.mode == "cnn":
                self.mhc = mhc_tensor.transpose(1, 2)
                self.pep = pep_tensor.transpose(1, 2)
            elif self.mode == "rnn":
                # self.mhc = mhc_tensor.transpose(0, 1)
                # self.pep = pep_tensor.transpose(0, 1)
                self.mhc = mhc_tensor
                self.pep = pep_tensor
                self._batch_dim = 1
            elif self.mode == "emb":
                # .squeeze(1).long()
                pass
        else:
            self.mhc = mhc_array
            self.pep = pep_array
            self.len = pep_lengths
            self.aff = affinities
            self.mode = mode      


    # def set_mode(self, mode):
    #     assert mode in ["clf", "reg"]
    #     if mode == "reg":
    #         self.aff = self.aff_backup.clone()
    #     elif mode == "clf":
    #         self.aff = self.aff_backup.clone()
    #         self.aff[self.aff >= BIND_THR] = 1
    #         self.aff[self.aff < BIND_THR] = 0


    def __getitem__(self, index):
        return self.mhc[index], self.pep[index], self.len[index], self.aff[index]


    def __len__(self):
        return self.mhc.size(0)


    def len_dim(self):
        if self.mode == "cnn":
            return 2
        else:
            return 1


    def aa_channels(self):
        if self.mode == "cnn":
            return self.pep.size(1)
        else:
            return self.pep.size(2)


    def mhc_max_len(self):
        return self.mhc.size(self.len_dim())


    def pep_max_len(self):
        return self.pep.size(self.len_dim())


    def binders(self):
        binders_vec = self.aff >= BIND_THR
        return MhcDataset(self.mhc[binders_vec], self.pep[binders_vec], 
                          self.len[binders_vec], self.aff[binders_vec], 
                          self.mode, True)


class MhcSynthDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, emb_path, size, mhc_pseudosequences, min_len, max_len, mode=["cnn", "rnn"][0]):
        super(MhcSynthDataset, self).__init__()

        assert mode in ["cnn", "rnn"]

        self.mode = mode

        self.size = size
        self.pseudo = list(mhc_pseudosequences.values())
        self.min_len = min_len
        self.max_len = max_len

        self.aa_model = np.load(emb_path)


    def generate_peptide(self):
        # TODO: sample indices of letter-vectors, not letters and their vectors afterwards
        # inds = np.random.choice(np.arange(0, len(self.w2v_model), 1), np.random.choice(range(self.min_len, self.max_len)))
        # self.w2v_model[inds]

        pep = "".join(np.random.choice(ALPHABET, np.random.choice(range(self.min_len, self.max_len+1))))
        seq_data = np.zeros((self.max_len, len(self.aa_model['A'])))
        seq_data[:len(pep), :] = np.array([self.aa_model[char] for char in pep])
        return seq_data, len(pep)


    def __getitem__(self, index):
        # TODO: optimize peptide generation
        pep_emb, pep_len = self.generate_peptide()
        pep_emb = torch.from_numpy(pep_emb).float()

        if self.mode == "cnn":
            return (torch.from_numpy(self.pseudo[np.random.choice(len(self.pseudo))]).transpose(0, 1), 
                    pep_emb.transpose(0, 1),  
                    pep_len,
                    np.random.uniform(0.0, 0.1))
        else:
            return (torch.from_numpy(self.pseudo[np.random.choice(len(self.pseudo))]), 
                    pep_emb, 
                    pep_len,
                    np.random.uniform(0.0, 0.1))


    def __len__(self):
        return self.size












