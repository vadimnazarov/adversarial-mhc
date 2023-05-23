import pandas as pd
import numpy as np
import re
import argparse
import gensim
from numpy.linalg import norm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", "--i",  type=str, default="./",
                    help="Path to bdata files")
parser.add_argument("--output_path", "--o", type=str,
                    help="Path where to store .npz data")
parser.add_argument("--w2v",  type=str, default="../w2v_models/up9mers_size_20_window_3.pkl",
                    help="Path to Word2Vec model")
parser.add_argument("--embedding_dim", "--dim", type=int, default=20,
                    help="Embedding dimension")
parser.add_argument("--nmers", type=str, default=None,
                    help="Whether to use nmers")

args = parser.parse_args()
output_path = args.output_path
input_path = args.input_path
emb_dim = args.embedding_dim
w2v_model = gensim.models.Word2Vec.load(args.w2v)
only9mers = True if args.nmers else False

unify_alleles = lambda x: re.sub('[*|:|-]', '', x)
log_meas = lambda x: 1 - np.log(x) / np.log(50000)
drop_alleles = ["HLAA1", "HLAA11", "HLAA2", "HLAA3", "HLAA3/11", "HLAA26", "HLAA24",
                "HLAB44", "HLAB51", "HLAB7", "HLAB27", "HLAB8", "HLACw1", "HLACw4", "HLAB60"]

def clean_mhc_df(mhc_df_path, only9mers):
    mhc_df = pd.read_csv(mhc_df_path, sep="\t")
    mhc_df.meas = log_meas(np.clip(mhc_df.meas, np.min(mhc_df.meas), 50000))
    mhc_df.mhc = mhc_df.mhc.apply(unify_alleles)
    mhc_df = mhc_df[~mhc_df.mhc.isin(drop_alleles) & mhc_df.species.isin(["human"])]
    if only9mers:
        mhc_df = mhc_df.loc[mhc_df.peptide_length == 9, :]
    return mhc_df

def embed_pseudo(pseudo_df, emb_dim):
    pseudo_dict = {}
    for i, seq in enumerate(pseudo_df.pseudo):
        acid_embs = np.array([w2v_model.wv[char] / norm(w2v_model.wv[char]) for char in seq])
        pseudo_dict[pseudo_df.mhc[i]] = np.array(acid_embs)
    return pseudo_dict

def get_embeddings(mhc_df, pseudo_dict, only9mers):
    mhc_data, pep_data = [], []
    max_pep_len = np.max(mhc_df.peptide_length)
    for mhc, seq in zip(mhc_df.mhc, mhc_df.sequence):
        pep_embs = np.array([w2v_model.wv[char] / norm(w2v_model.wv[char]) for char in seq])
        mhc_embs = pseudo_dict[mhc]
        pep_data.append(pep_embs)
        mhc_data.append(mhc_embs)
    if not only9mers:
        for ind, i in enumerate(pep_data):
            pep_data[ind] = np.pad(i, ((0, max_pep_len - i.shape[0]), (0,0)), "constant")
    return np.array(mhc_data), np.array(pep_data), mhc_df.meas

print("Loading MHC pseudosequence...")
mhc_df = pd.read_csv(input_path + "mhc_seq_imghtla.csv")
pseudo_mhc = embed_pseudo(mhc_df, emb_dim)
print("Nuumber of unique pseudosequences: {}".format(len(pseudo_mhc)))

print("Loading train data...")
train_df = clean_mhc_df(input_path + "bdata.2009.tsv", only9mers)
mhc_train, pep_train, labels_train = get_embeddings(train_df, pseudo_mhc, only9mers)
print("\nShape of training data.\nMHC embeddings: {}\nPep embeddings: {}\nLabels: {}".format(mhc_train.shape,
                                                                                             pep_train.shape,
                                                                                             labels_train.shape))

print("Loading test data..")
test_df = clean_mhc_df(input_path + "blind.tsv", True)
mhc_test, pep_test, labels_test = get_embeddings(test_df, pseudo_mhc, True)
print("\nShape of test data.\nMHC embeddings: {}\nPep embeddings: {}\nLabels: {}".format(mhc_test.shape,
                                                                                             pep_test.shape,
                                                                                             labels_test.shape))


print("Saving train and test in .npz to {}".format(output_path))
if not os.path.exists(output_path):
    os.makedirs(output_path)
np.savez_compressed(output_path+"mhc_train", mhc_train)
np.savez_compressed(output_path+"pep_train", pep_train)
np.savez_compressed(output_path+"labels_train", labels_train)
np.savez_compressed(output_path+"mhc_test", mhc_test)
np.savez_compressed(output_path+"pep_test", pep_train)
np.savez_compressed(output_path+"labels_test", labels_test)
