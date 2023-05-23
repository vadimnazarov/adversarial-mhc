from datetime import datetime
import itertools
import json
import multiprocessing as mp
import random
import numpy as np
from sklearn.metrics import precision_score, f1_score, roc_auc_score
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from .data import *
from .sampler import *


def make_dataloader(dataset, batch_size, num_workers, nn=["cnn", "rnn"][0], sampling=SAMPLING_TYPES[0], pin_memory=USE_CUDA):
    assert sampling in SAMPLING_TYPES

    if sampling == "brute":
        batch_sampler = BruteforceSampler(len(dataset), batch_size)

        #
        # With Synth Dataset:
        #
        # SEGFAULT 11
        # batch_sampler = BruteforceSampler(len(dataset), batch_size)

        # SEGFAULT 11
        # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        # SEGFAULT 11
        # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=pin_memory)

        # First run: No crash for 50 iterations
        # Second run: No problems
        # Third run on the last iteration:
        # Exception ignored in: <function WeakValueDictionary.__init__.<locals>.remove at 0x1072f00d0>
        # Traceback (most recent call last):
        #   File "/Users/vdn/anaconda/lib/python3.5/weakref.py", line 117, in remove
        # TypeError: 'NoneType' object is not callable
        # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)

        #
        # Without Synth Dataset:
        #
        # SEGFAULT 11
        # batch_sampler = BruteforceSampler(len(dataset), batch_size)

        # SEGFAULT 11
        # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

        # First run: No crush for 50 iterations
        # Second run: No crush for 50 iterations
        # Third run: No crush for 50 iterations
        # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=pin_memory)

        # On the last iteration [THIS IS A PYTHON BUG, NOT PYTORCH]
        # Exception ignored in: <function WeakValueDictionary.__init__.<locals>.remove at 0x1048cc0d0>
        # Traceback (most recent call last):
        #   File "/Users/vdn/anaconda/lib/python3.5/weakref.py", line 117, in remove
        # TypeError: 'NoneType' object is not callable
        # Second run was OK.
        # Third run: was OK.
        # return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)


    elif sampling == "bal": 
        batch_sampler = BindingBalancedSampler(dataset.aff, BIND_THR, batch_size)
    elif sampling == "wei":
        _, uniq_mhc = np.unique(dataset.mhc, return_inverse=True, axis=0)
        mhc_bind = np.hstack([uniq_mhc.reshape((-1, 1)), np.where(dataset.aff > BIND_THR, 1, 0).reshape((-1, 1))])
        _, inds, counts = np.unique(mhc_bind, return_counts=True, return_inverse=True, axis=0)
        weights = {i: 1 / (len(counts) * counts[i]) for i in inds}
        batch_sampler = WeightedSampler([weights[i] for i in inds], batch_size, alpha=1.5)

    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory)


class Trainer:

    def __init__(self, nn_mode, model, dataset, synth_dataset=None, pred_mode="reg"):
        nn_mode = nn_mode.lower()

        self.model = model
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.synth_dataset = synth_dataset
        self.nn_mode = nn_mode
        self.train_scores = []
        self.test_scores = []
        self.info = {}
        self.info["model"] = model.name()
        self.info["training start date"] = str(datetime.now())
        # TODO:
        # self.info["phase"]
        # self.info["training end date"] = str(datetime.now())
        self.info["scores"] = {}
        self.info["scores"]["train"] = {}
        self.info["scores"]["val"] = {}

        if self.nn_mode == "cnn":
            self.preproc = lambda x, _: x
        elif self.nn_mode == "rnn":
            # self.preproc = lambda x, lens: pack_padded_sequence(x, list(lens), batch_first=True)
            self.preproc = lambda x, lens: (x, lens)


    # TODO: training phases: each one with their own target, epochs, scores, etc.
    def add_training_phase(phase_name, datasets, target_mode="reg"):
        assert "train" in datasets

        new_phase_key = str(len(self.info["phase"]) + 1) + "-" + phase_name
        self.info["phase"][new_phase_key] = {"scores": {}}

        for key in datasets:
            self.info["phase"][new_phase_key]["scores"][key] = {}

        self.info["phase"][new_phase_key]["start datetime"] = ""
        self.info["phase"][new_phase_key]["end datetime"] = ""
        self.info["phase"][new_phase_key]["target_mode"] = target_mode
        # batch size
        # optimizer and its params
        # sampling
        # start epoch
        # n epochs
        # criterion


    def train(self, n_epochs, criterion, optimizer, batch_size, test_dataset=None, sampling="brute", num_workers=mp.cpu_count(), start_epoch=1):

        def _model_scores(pred_df, new_scores):
            # print(sum(pred_df[:,1] > 1))
            if self.pred_mode == "reg":
                f1 = f1_score(np.where(pred_df[:,0] >= BIND_THR, 1, 0), np.where(pred_df[:,1] >= BIND_THR, 1, 0))
                auc = roc_auc_score(np.where(pred_df[:,0] >= BIND_THR, 1, 0), np.where(pred_df[:,1] >= BIND_THR, 1, 0))
            else:
                f1 = f1_score(np.where(pred_df[:,0] >= .5, 1, 0), np.where(pred_df[:,1] >= .5, 1, 0))
                auc = roc_auc_score(np.where(pred_df[:,0] >= .5, 1, 0), np.where(pred_df[:,1] >= .5, 1, 0))

            # mean = np.mean(new_scores)
            n_of_binders = np.where(pred_df[:, 0] > .9)[0].shape[0]
            pred_df = pred_df[pred_df[:,1].argsort()]
            mean = pred_df[-n_of_binders:, 0].sum() / n_of_binders

            p05 = np.percentile(new_scores, 5)
            p95 = np.percentile(new_scores, 95)

            # TODO: namedtuple?
            return f1, auc, mean, p05, p95

        def _add_scores(mode, values):
            # TODO: Check for different training modes: reg or clf
            for key, value in zip(["f1", "auc", "mse", "mse_p05", "mse_p95"], values):
                if key not in self.info["scores"][mode]:
                    self.info["scores"][mode][key] = []
                self.info["scores"][mode][key].append(value)


        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        if self.synth_dataset:
            self.batch_size = self.batch_size // 2
            self.synth_dataloader = make_dataloader(self.synth_dataset, self.batch_size, num_workers, self.nn_mode, "brute")
        else:
            self.synth_dataloader = itertools.repeat(None)

        self.dataloader = make_dataloader(self.dataset, self.batch_size, num_workers, self.nn_mode, sampling)

        for i_epoch in range(start_epoch, n_epochs+start_epoch):
            print("Epoch:", i_epoch, "/", n_epochs+start_epoch-1)
            print("Model:", self.model.name())

            (new_train_scores, pred_df, n_batches), train_seconds = compute_time(self.step)

            self.train_scores.append(new_train_scores)
            scores = _model_scores(pred_df, new_train_scores)

            print("Results:")
            print("{:7}{:>7}{:>7}{:>9}{:>9}{:>9}".format("", "F1", "AUC", "MSE(avg)", "MSE(.05)", "MSE(.95)"))
            print("{:7}{:>7.3}{:>7.3}{:>9.4}{:>9.4}{:>9.4}".format("train", *scores))
            _add_scores("train", scores)

            if test_dataset:
                (new_test_scores, test_pred_df), test_seconds = compute_time(self.evaluate, test_dataset)
                self.test_scores.append(new_test_scores)
                scores = _model_scores(test_pred_df, new_test_scores)
                _add_scores("val", scores)

                print("{:7}{:>7.3}{:>7.3}{:>9.4}{:>9.4}{:>9.4}".format("val", *scores))

            print("Time:")
            print("  training sec/epoch: ", round(train_seconds, 3))
            print("  training sec/batch: ", round(train_seconds / n_batches, 3), "(" + str(n_batches), "batches)")
            if test_dataset:
                print("  validation sec/test:", round(test_seconds, 3))

            print()


    def step(self):
        loss_list = []
        y_true = []
        preds = []

        for n_batches, (batch, synth_batch) in enumerate(zip(self.dataloader, self.synth_dataloader)):
            # NOTE: batch == mhc, pep, pep_lens, affinities
            weights = torch.FloatTensor([1 for _ in range(self.batch_size)])
            if synth_batch:
                # TODO: np.argsort for lengths in case of RNN
                batch[0] = torch.cat([batch[0], synth_batch[0]], 0)
                batch[1] = torch.cat([batch[1], synth_batch[1]], 0)
                batch[2] = batch[2].long() + synth_batch[2]
                batch[3] = torch.cat([batch[3], synth_batch[3].float()], 0)
                weights = torch.cat([weights, torch.FloatTensor([0.7 for _ in range(self.batch_size)])])
            # TODO: CUDA support
            batch[3] = batch[3].reshape((-1, 1))
            if self.pred_mode == "clf":
                batch[3][batch[3] >= BIND_THR] = 1
                batch[3][batch[3] < BIND_THR] = 0

            self.optimizer.zero_grad()

            y_pred = self.model(self.preproc(batch[0], [batch[0].size(1) for _ in range(self.batch_size)]), 
                                self.preproc(batch[1], batch[2]),
                                self.pred_mode)

            if n_batches:
                y_true = np.vstack([y_true, batch[3]])
                preds = np.vstack([preds, y_pred.detach().numpy()])
            else:
                y_true = batch[3]
                preds = y_pred.detach().numpy()

            loss = self.criterion(y_pred, batch[3], reduce=False) * weights
            loss = loss.mean()

            loss_list.append(loss.item())

            loss.backward()
            self.optimizer.step()

        return loss_list, np.hstack([y_true, preds]), n_batches+1


    def evaluate(self, test_dataset, batch_size=1024, num_workers=0):
        self.model.eval()

        loss_list = []
        y_true = []
        preds = []

        dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
            shuffle=False, pin_memory=USE_CUDA, drop_last=False)

        for i, batch in enumerate(dataloader):
            y_pred = self.model(self.preproc(batch[0], [batch[0].size(1) for _ in range(batch[0].size(0))]), 
                                self.preproc(batch[1], batch[2]), 
                                self.pred_mode)
            batch[-1] = batch[-1].reshape((-1, 1))
            if self.pred_mode == "clf":
                batch[3][batch[3] >= BIND_THR] = 1
                batch[3][batch[3] < BIND_THR] = 0

            loss = self.criterion(y_pred, batch[-1])
            loss_list.append(loss.item())

            if i:
                y_true = np.vstack([y_true, batch[-1]])
                preds = np.vstack([preds, y_pred.detach().numpy()])
            else:
                y_true = batch[-1]
                preds = y_pred.detach().numpy()

        self.model.train()
        return loss_list, np.hstack([y_true, preds])


def evaluate_abelin(trainer, abelin_dataset, batch_size=1024, num_workers=mp.cpu_count(), comment=""):

    def _evaluate_allele(pred_df, key, allele):
        n_of_binders = np.where(pred_df[:, 0] > .9)[0].shape[0]
        pred_df[:, 1] = unlog_aff(pred_df[:, 1])
        pred_df = pred_df[pred_df[:,1].argsort()]
        prec_score = pred_df[:n_of_binders, 0].sum() / n_of_binders
        trainer.info["scores"]["abelin"][key][allele] = prec_score
        return prec_score

    ppv_scores = {allele:-1 for allele in abelin_dataset}
    trainer.info["scores"]["abelin"] = {}
    trainer.info["scores"]["abelin"]["all"] = {}

    preds_to_save = {}

    
    for allele, dataset in sorted(abelin_dataset.items(), key=lambda x: x[0]):
        _, pred_df = trainer.evaluate(dataset, batch_size, num_workers)

        ppv_scores[allele] = _evaluate_allele(pred_df.copy(), "all", allele)
        preds_to_save[allele] = pred_df.copy()

        for uniq_len in np.unique(dataset.len):
            if str(uniq_len) not in trainer.info["scores"]["abelin"]:
                trainer.info["scores"]["abelin"][str(uniq_len)] = {}

            pred_sub_df = pred_df[dataset.len == uniq_len]

            _evaluate_allele(pred_sub_df.copy(), str(uniq_len), allele)

        print("{:10}{:>7}{:>6}{:>7.4}".format(allele, len(dataset), np.where(dataset.aff.numpy() > .9)[0].shape[0], ppv_scores[allele]))

    np.savez_compressed("abelin_" + comment, **preds_to_save)

    return ppv_scores
