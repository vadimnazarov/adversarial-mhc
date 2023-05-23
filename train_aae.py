import argparse
from torch import optim

from feng.args import *
from feng.data import *
from feng.trainer import *
from feng.aae import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i",  type=str, 
                        default="./data/",
                        help="Path to the folder with input files")
    parser.add_argument("--pseudo", "-p",  type=str, 
                        default="./data/pseudoseqs_freq.csv",
                        help="Path to the file with pseudo sequences")
    parser.add_argument("--output", "-o", type=str,
                        default="./",
                        help="Path to the output folder")
    parser.add_argument("--embedding",  "--embeddings", "--emb", type=str, 
                        default="./aa_models/w2v_9mers_3wind_20dim_norm.pkl",
                        help="Path to amino acid embeddings")
    parser.add_argument("--comment", type=str,
                        default="",
                        help="Additional comment")
    # parser.add_argument("--pep_len", "--len", type=str,
    #                     default="8-11",
    #                     help="Min and max peptide length in format '<min>-<max>' (default: '8-11')")
    parser.add_argument("--abelin", action="store_true",
                        help="Abelin testing if passed")

    # parser.add_argument("--pep_blocks", "--pep", type=int,
    #                     default=2,
    #                     help="Number of blocks for peptide branch")
    # parser.add_argument("--mhc_blocks", "--mhc", type=int,
    #                     default=2,
    #                     help="Number of blocks for MHC branch")

    parser.add_argument("--batch_size", "-b", type=int, 
                        default=64,
                        help="size batch")
    parser.add_argument("--epochs", "-e", type=int,
                        default=10,
                        help="Number of epochs")
    parser.add_argument("--learning_rate", "--lr", type=float,
                        default=0.002,
                        help="Learning rate")
    parser.add_argument("--sampling", "-s", type=str,
                        default="bal",
                        help="'brute', 'bal' or 'wei'")
    parser.add_argument("--nn_mode", "--nn", type=str,
                        default="cnn",
                        help="'cnn' or 'rnn'")
    parser.add_argument("--num_workers", "--nw", type=int,
                        default=0,
                        help="Number of workers for DataLoader")
    parser.add_argument("--synth", action="store_true",
                        help="If specified than generate random peptides for batches")
    parser.add_argument("--chaos", action="store_true",
                        help="How many epochs for chaotic pretraining")

    parser.add_argument("--linear_dim", "--ld", type=str,
                        default="64-64",
                        help="Dimensions and number of Dense layers in a form <#neurons>-<#neurons>-... (default: '64-64'). ")
    parser.add_argument("--latent_dim", type=int,
                        default=32,
                        help="Latent dimension")

    parser.add_argument("--drop_lin", "--dl", type=float,
                        default=.2,
                        help="Dropout for linear layers")

    parser.add_argument("--labels", action="store_true",
                        help="Pass labels to AAE decoder. Not compatible with '--semi'")
    parser.add_argument("--semi", action="store_true",
                        help="Semi-supervised. Not compatible with '--labels'")
    parser.add_argument("--wass", action="store_true",
                        help="Train using Wasserstein metrics")
    parser.add_argument("--gp", action="store_true",
                        help="Train using gradient penalty")
    args = parser.parse_args()

    aae_mode = ""
    assert (not args.labels) or (not args.semi)
    if args.gp:
        assert args.wass
    if args.labels:
        aae_mode = "labels"
    elif args.semi:
        aae_mode = "semi"

    # make_io_args(parser)
    # make_train_args(parser)
    # make_resnet_args(parser)
    # make_rnn_args(parser)
    # make_dense_args(parser)

    # io_args = process_io_args(args)
    # load data, etc.
    # train_args = process_train_args(args)
    # make trainer, optimizer, etc
    # nn_args = process_nn_args(args)
    # start train, test, etc.
    
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    MIN_LEN = 8
    MAX_LEN = 11
    # INPUT_TRAIN = "/aae_train.csv.gz"
    INPUT_NOLABEL = ""
    # INPUT_TEST = "/aae_test.csv.gz"
    INPUT_TRAIN = "/aae_train_high.csv.gz"
    INPUT_NOLABEL = "/aae_train_low.csv.gz"
    INPUT_TEST = "/aae_test_v2.csv.gz"

    pseudo_sequences = load_pseudo(args.pseudo, args.embedding)
    print()
    train_dataset = load_iedb("cnn", args.input + INPUT_TRAIN, args.embedding, pseudo_sequences, min_len=MIN_LEN, max_len=MAX_LEN, pad_char="X")
    
    nolabel_dataset = INPUT_NOLABEL
    if nolabel_dataset:
        nolabel_dataset = load_iedb("cnn", args.input + INPUT_NOLABEL, args.embedding, pseudo_sequences, min_len=MIN_LEN, max_len=MAX_LEN, pad_char="X")

    synth_dataset = None
    if args.synth:
        synth_dataset = MhcSynthDataset(args.embedding, len(train_dataset), pseudo_sequences, min(train_dataset.len), max(train_dataset.len), args.nn_mode)

    test_dataset = None
    abelin_dataset = None

    print()
    if args.abelin:
        abelin_dataset = load_abelin("cnn", args.input + "/abelin.csv.gz", args.embedding, pseudo_sequences, min_len=MIN_LEN, max_len=MAX_LEN, pad_char="X")
    print()
    test_dataset = load_iedb("cnn", args.input + INPUT_TEST, args.embedding, pseudo_sequences, min_len=MIN_LEN, max_len=MAX_LEN, pad_char="X")
    print()

    nn_args = {
        "mhc_len": train_dataset.mhc_max_len(),
        "pep_len": train_dataset.pep_max_len(),
        # "mhc_blocks": args.mhc_blocks,
        # "pep_blocks": args.pep_blocks,
        "aa_channels": train_dataset.aa_channels(),
        # "kernel": 3,
        # "hidden": args.hidden_dim,
        # "layers": args.layers,
        "dense": list(map(int, args.linear_dim.split("-"))),
        # "drop_inp": args.drop_inp,
        "drop_lin": args.drop_lin,
        # "drop_nn": args.drop_nn,
        "nn_mode": args.nn_mode
    }

    if nn_args["nn_mode"] == "aae":
        model = make_aae(args.latent_dim, nn_args["pep_len"], nn_args["aa_channels"], nn_args["dense"], nn_args["drop_lin"], aae_mode, device, grad_p=args.gp)
    else:
        print("Wrong NN architecture:", nn_args["nn_mode"])
        0/0


    print(model.Q)
    print(model.P)
    print(model.D)
    if model.D_cat:
        print(model.D_cat)

#    make_optim = lambda params, lr: optim.Adam(params, lr=lr)
    make_optim = lambda params, lr: optim.RMSprop(params, lr=lr, centered=True)
    
    make_optimizers = lambda lr: {"encoder": make_optim(model.Q.parameters(), lr), 
                                  "decoder": make_optim(model.P.parameters(), lr),
                                  "discriminator": make_optim(model.D.parameters(), lr=lr/2),
                                  "generator": make_optim(model.Q.parameters(), lr=lr/2),
                                  "classifier": make_optim(model.Q.parameters(), lr=lr) if model.D_cat else None,
                                  "discriminator_cat": make_optim(model.D_cat.parameters(), lr=lr/2) if model.D_cat else None,
                                  "generator_cat": make_optim(model.Q.parameters(), lr=lr/2) if model.D_cat else None
                                  }

    # previos lr=0.001
    lr=args.learning_rate
    optimizers = make_optimizers(lr)

    pred_mode = "reg"

    trainer = AAETrainer(nn_args["nn_mode"], model, train_dataset, synth_dataset, pred_mode, nolabel_dataset=nolabel_dataset, wasserstein=args.wass, device=device, use_gp=args.gp)
    trainer.train(args.epochs, None, optimizers, args.batch_size, sampling=args.sampling, 
                  num_workers=args.num_workers, test_dataset=test_dataset, aae_mode=aae_mode)

    lr /= 10
    optimizers = make_optimizers(lr)
    trainer.train(args.epochs, None, optimizers, args.batch_size, sampling=args.sampling, 
                  num_workers=args.num_workers, test_dataset=test_dataset, start_epoch=args.epochs+1, aae_mode=aae_mode)

    out_filename = "_".join([args.sampling, 
                            args.embedding[args.embedding.rfind("/")+1 : -4], 
                            "synth" if args.synth else "nosyn",
                            "e" + str(args.epochs), 
                            "b" + str(args.batch_size),
                            "lat" + str(args.latent_dim),
                            "lin"+args.linear_dim])

    out_filename += "_aae"
    if args.comment: 
        out_filename += "_" + args.comment

    if abelin_dataset:
        ppv_scores = evaluate_abelin(trainer, abelin_dataset, num_workers=0, comment=out_filename)
    
    with open(out_filename +  ".txt", "w") as outf:
        outf.write(json.dumps(trainer.info, sort_keys=True, indent=4, separators=(',', ': ')))

    # for model_type, sub_model in [("P", model.P), ("Q", model.Q)]:
    #     with open("model_" + model_type + "_" + out_filename +  ".pt", "wb") as outf:
    #         torch.save(sub_model.state_dict(), outf)

