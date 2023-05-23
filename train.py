import argparse
from torch import optim

from feng.args import *
from feng.data import *
from feng.trainer import *
from feng.models import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i",  type=str, 
                        default="./data/",
                        help="Path to the folder with input files")
    parser.add_argument("--pseudo", "-p",  type=str, 
                        default="./data/pseudoseqs.csv.gz",
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
    parser.add_argument("--optim", type=str,
                        default="rmsprop",
                        help="rmsprop or adam")


    parser.add_argument("--pep_blocks", "--pep", type=int,
                        default=2,
                        help="Number of blocks for peptide branch")
    parser.add_argument("--mhc_blocks", "--mhc", type=int,
                        default=2,
                        help="Number of blocks for MHC branch")
    parser.add_argument("--filters", "-f", type=int,
                        default=32,
                        help="Number of filters")

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
    # parser.add_argument("--train_mode", "--tm", type=str,
    #                     default="reg",
    #                     help="Regression ('reg') or classification ('clf')")
    parser.add_argument("--num_workers", "--nw", type=int,
                        default=0,
                        help="Number of workers for DataLoader")
    parser.add_argument("--synth", action="store_true",
                        help="If specified than generate random peptides for batches")
    parser.add_argument("--chaos", action="store_true",
                        help="How many epochs for chaotic pretraining")

    parser.add_argument("--hidden_dim", "--hd", type=int,
                        default=32,
                        help="Size of a hidden dimension for RNN")
    parser.add_argument("--layers", "--nl", type=int,
                        default=2,
                        help="Number of layers of RNN")
    parser.add_argument("--linear_dim", "--ld", type=str,
                        default="32-32",
                        help="Dimensions and number of Dense layers in a form <#neurons>-<#neurons>-... (default: '32-32'). ")

    parser.add_argument("--drop_inp", "--di", type=float,
                        default=.2,
                        help="Dropout for the input data")
    parser.add_argument("--drop_nn", "--dn", type=float,
                        default=.2,
                        help="Dropout for convolutions in CNN / RNN cells")
    parser.add_argument("--drop_lin", "--dl", type=float,
                        default=.2,
                        help="Dropout for linear layers")

    parser.add_argument("--aa_channels", type=int,
                        default=0,
                        help="Dropout for linear layers")
    args = parser.parse_args()

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

    # TRAIN_FILE = "/curated.csv.gz"
    # VAL_FILE = "/jci.csv.gz"

    TRAIN_FILE = "/classic_train.csv.gz"
    VAL_FILE = "/classic_test.csv.gz"

    pseudo_sequences = load_pseudo(args.pseudo, args.embedding)
    print()
    train_dataset = load_iedb(args.nn_mode, args.input + TRAIN_FILE, args.embedding, pseudo_sequences, min_len=8, max_len=11, pad_char="X")

    synth_dataset = None
    if args.synth:
        synth_dataset = MhcSynthDataset(args.embedding, len(train_dataset), pseudo_sequences, min(train_dataset.len), max(train_dataset.len), args.nn_mode)

    test_dataset = None
    abelin_dataset = None

    print()
    if args.abelin:
        abelin_dataset = load_abelin(args.nn_mode, args.input, args.embedding, pseudo_sequences, min_len=8, max_len=11, pad_char="X")
    print()
    test_dataset = load_iedb(args.nn_mode, args.input + VAL_FILE, args.embedding, pseudo_sequences, min_len=8, max_len=11, pad_char="X")
    print()

    nn_args = {
        "mhc_len": train_dataset.mhc_max_len(),
        "pep_len": train_dataset.pep_max_len(),
        "mhc_blocks": args.mhc_blocks,
        "pep_blocks": args.pep_blocks,
        "aa_channels": args.aa_channels if args.aa_channels else train_dataset.aa_channels(),
        "kernel": 5,
        "hidden": args.hidden_dim,
        "layers": args.layers,
        "dense": list(map(int, args.linear_dim.split("-"))),
        "drop_inp": args.drop_inp,
        "drop_lin": args.drop_lin,
        "drop_nn": args.drop_nn,
        "nn_mode": args.nn_mode
    }

    # for s in ["bal", "wei"]:
    #     args.sampling = s

    if nn_args["nn_mode"] == "rnn":
        model = AttnRNN(nn_args["pep_len"], nn_args["hidden"], nn_args["layers"], nn_args["aa_channels"], nn_args["dense"], nn_args["drop_lin"], nn_args["drop_nn"], nn_args["drop_inp"])
    elif nn_args["nn_mode"] == "cnn":
        print("AA channels", nn_args["aa_channels"])
        model = ResNet(args.filters, nn_args["mhc_len"], nn_args["pep_len"], args.mhc_blocks, args.pep_blocks, nn_args["aa_channels"], nn_args["kernel"], nn_args["dense"], nn_args["drop_lin"], nn_args["drop_nn"], nn_args["drop_inp"])
    else:
        print("Wrong NN architecture:", nn_args["nn_mode"])
        0/0

    if args.chaos:
        print("Chaotic pre-training")
        criterion, pred_mode = F.binary_cross_entropy, "clf"
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, centered=True)

        train_dataset_binders = train_dataset.binders()
        print("Number of binders:", len(train_dataset_binders))
        chaos_dataset = MhcSynthDataset(args.embedding, len(train_dataset_binders), pseudo_sequences, 8, 11, nn_args["nn_mode"])

        trainer = Trainer(nn_args["nn_mode"], model, train_dataset_binders, synth_dataset=chaos_dataset, pred_mode=pred_mode)
        trainer.train(args.chaos, criterion, optimizer, args.batch_size, sampling="wei", num_workers=args.num_workers, test_dataset=test_dataset)

    if args.optim == "rmsprop":
        make_optimizer = lambda model: optim.RMSprop(model.parameters(), lr=args.learning_rate, centered=True)
    elif args.optim == "adam": 
        make_optimizer = lambda model: optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        print("Wrong optimizer name:", args.optim)
        0/0 
    # make_optimizer = lambda model: optim.SparseAdam(model.parameters(), lr=args.learning_rate)

    ##################
    ##################
    trainer_list = {}
    for i in range(50):
        if i % 2:
            criterion, pred_mode = F.mse_loss, "reg"
        else:
            criterion, pred_mode = F.binary_cross_entropy, "clf"
        optimizer = make_optimizer(model)
        trainer = Trainer(nn_args["nn_mode"], model, train_dataset, synth_dataset=synth_dataset, pred_mode=pred_mode)
        trainer.train(args.epochs, criterion, optimizer, args.batch_size, sampling=args.sampling, num_workers=args.num_workers, test_dataset=test_dataset, start_epoch=args.epochs*i+1)
        trainer_list[pred_mode] = trainer
    ##################
    ##################

    # criterion, pred_mode = F.mse_loss, "reg"
    # criterion, pred_mode = F.binary_cross_entropy, "clf"
    # trainer = Trainer(nn_args["nn_mode"], model, train_dataset, synth_dataset=synth_dataset, pred_mode=pred_mode)

    # optimizer = make_optimizer(model)
    # trainer.train(args.epochs, criterion, optimizer, args.batch_size, sampling=args.sampling, num_workers=args.num_workers, test_dataset=test_dataset)

    # optimizer = make_optimizer(model)
    # trainer.train(args.epochs, criterion, optimizer, args.batch_size*4, sampling=args.sampling, num_workers=args.num_workers, test_dataset=test_dataset, start_epoch=args.epochs+1)

    # optimizer = make_optimizer(model)
    # trainer.train(args.epochs, criterion, optimizer, args.batch_size*16, sampling=args.sampling, num_workers=args.num_workers, test_dataset=test_dataset, start_epoch=args.epochs*2+1)

    for key, trainer in trainer_list.items():
        print(key)
        out_filename = "_".join([args.sampling, 
                                args.embedding[args.embedding.rfind("/")+1 : -4], 
                                "synth" if args.synth else "nosyn",
                                "e" + str(args.epochs), 
                                "b" + str(args.batch_size), 
                                "mhc"+str(args.mhc_blocks), 
                                "pep"+str(args.pep_blocks),
                                "lin"+args.linear_dim])

        out_filename += "_filters" + str(args.filters)
        if args.comment: 
            out_filename += "_" + args.comment

        if abelin_dataset:
            ppv_scores = evaluate_abelin(trainer, abelin_dataset, num_workers=0, comment=out_filename)
    
    # with open(out_filename +  ".txt", "w") as outf:
    #     outf.write(json.dumps(trainer.info, sort_keys=True, indent=4, separators=(',', ': ')))

    # with open("model_" + out_filename +  ".pt", "wb") as outf:
    #     torch.save(model.state_dict(), outf)

