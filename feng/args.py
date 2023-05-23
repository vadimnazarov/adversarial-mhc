import argparse
import json


def make_io_args(parser):
    parser.add_argument("--input", "-i",  type=str, 
                        default="./data/",
                        help="Path to the folder with input files")
    parser.add_argument("--output", "-o", type=str,
                        default="./",
                        help="Path to the output folder")
    parser.add_argument("--embedding",  "--embeddings", "--emb", type=str, 
                        default="./aa_models/w2v_9mers_3wind_20dim_norm.pkl",
                        help="Path to amino acid embeddings")
    # parser.add_argument("--pep_len", "--len", type=str,
    #                     default="8-11",
    #                     help="Min and max peptide length in format '<min>-<max>' (default: '8-11')")
    # parser.add_argument("--model", "-m", type=str,
    #                     default="",
    #                     help="Path to the PyTorch model.")


def make_train_args(parser):
    parser.add_argument("--batch_size", "-b", type=int, 
                        default=64,
                        help="size batch")
    parser.add_argument("--epochs", "-e", type=int,
                        default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", "--lr", type=float,
                        default=-1,
                        help="Learning rate")
    parser.add_argument("--sampling", "-s", type=str,
                        default="bal",
                        help="'brute', 'bal' or 'wei'")
    # parser.add_argument("--train_mode", "--tm", type=str,
    #                     default="reg",
    #                     help="Regression ('reg') or classification ('clf')")
    parser.add_argument("--num_workers", "--nw", type=int,
                        default=0,
                        help="Number of workers for DataLoader")
    parser.add_argument("--synth", action="store_true",
                        help="If specified than generate random peptides for batches")
    parser.add_argument("--drop_inp", "--di", type=float,
                        default=.2,
                        help="Dropout for the input data")


def make_resnet_args(parser):
    parser.add_argument("--pep_blocks", "--pep", type=int,
                        default=2,
                        help="Number of blocks for peptide branch")
    parser.add_argument("--mhc_blocks", "--mhc", type=int,
                        default=2,
                        help="Number of blocks for MHC branch")
    parser.add_argument("--drop_cnn", "--dn", type=float,
                        default=.2,
                        help="Dropout for convolutions")


def make_rnn_args(parser):
    parser.add_argument("--hidden_dim", "--hd", type=int,
                        default=32,
                        help="Size of a hidden dimension for RNN")
    parser.add_argument("--layers", "--nl", type=int,
                        default=2,
                        help="Number of layers of RNN")
    parser.add_argument("--drop_rnn", "--dn", type=float,
                        default=.2,
                        help="Dropout for convolutions")


def make_dense_args(parser):
    parser.add_argument("--linear_dim", "--ld", type=str,
                        default="32-32",
                        help="Dimensions and number of Dense layers in a form <#neurons>-<#neurons>-... (default: '32-32'). ")
    parser.add_argument("--drop_lin", "--dl", type=float,
                        default=.2,
                        help="Dropout for linear layers")


def process_io_args(args):
    io_args = {"input", "output", "embedding", "lens"}
    return io_args


def process_train_args(args):
    train_args = {"input", "output", "embedding", "lens"}
    return train_args


def process_nn_args(args):
    nn_args = {
        "mhc_len": train_dataset.mhc_max_len(),
        "pep_len": train_dataset.pep_max_len(),
        "mhc_blocks": args.pep_blocks,
        "pep_blocks": args.mhc_blocks,
        "aa_channels": 20,
        "kernel": 3,
        "hidden": args.hidden_dim,
        "layers": args.layers,
        "dense": list(map(int, args.linear_dim.split("-"))),
        "drop_inp": args.drop_inp,
        "drop_lin": args.drop_lin,
        "drop_nn": args.drop_nn,
        "nn_mode": args.nn_mode
    }
    return nn_args


def join_args_to_json(io_args, train_args, nn_args, filepath):
    full_args = {}
    full_args["io"] = io_args
    full_args["fit"] = train_args
    full_args["nn"] = nn_args
    with open(filepath, "w") as outf:
        outf.write(json.dumps(full_args, sort_keys=True, indent=4, separators=(',', ': ')))