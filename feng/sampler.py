import itertools
import random
import numpy as np
from torch.utils.data.sampler import Sampler

from .data import *


SAMPLING_TYPES = ["brute", "bal", "wei"]


#
# Sampler can rearrange data so it will be outputted in order suitable for immediate pack_padded_sequence
#

class BruteforceSampler(Sampler):
    def __init__(self, n_samples, batch_size):
        self.n_samples = n_samples
        self.batch_size = batch_size


    def __iter__(self):
        # Dropping the last batch because screw it

        indices = np.random.permutation(self.n_samples)
        # indices = torch.randperm(self.n_samples)
        for i in range(self.n_samples // self.batch_size):
            yield list(np.sort(indices[i*self.batch_size : (i+1)*self.batch_size]))
            # yield list(torch.sort(indices[i*self.batch_size : (i+1)*self.batch_size])[0])


    def __len__(self):
        return self.n_samples


class BindingBalancedSampler(Sampler):
    def __init__(self, affinities, bind_thr, batch_size):
        self.n_samples = affinities.size(0)
        self.batch_size = batch_size
        indices = np.arange(0, affinities.size(0), 1)
        self.binders    = indices[affinities.numpy() >= bind_thr]
        self.nonbinders = indices[affinities.numpy() < bind_thr]


    def __iter__(self):
        half_bs = self.batch_size // 2

        indices_binders = np.random.permutation(self.binders)
        indices_nonbinders = np.random.permutation(self.nonbinders)
        if len(indices_binders) > len(indices_nonbinders):
            indices_nonbinders = np.hstack([indices_nonbinders, np.random.choice(indices_nonbinders, len(indices_binders) - len(indices_nonbinders))])
        elif len(indices_binders) < len(indices_nonbinders):
            indices_binders = np.hstack([indices_binders, np.random.choice(indices_binders, len(indices_nonbinders) - len(indices_binders))])

        for i in range(len(indices_binders) // half_bs):
            yield list(np.sort(np.hstack([indices_binders[i*half_bs : (i+1)*half_bs], indices_nonbinders[i*half_bs : (i+1)*half_bs]])))


    def __len__(self):
        return self.n_samples


# multinomial distribution (`torch.multinomial`) is super slow.
# using the alternative from:
# https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
# https://github.com/parthaca/examples/blob/master/word_language_model/alias_multinomial.py
class WeightedSampler(Sampler):

    def __init__(self, weights, batch_size, alpha=1):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = len(weights)
        self.batch_size = batch_size

        K = len(weights)
        self.q = torch.zeros(K)
        self.J = torch.LongTensor([0]*K)
        if USE_CUDA:
            self.q = self.q.cuda()
            self.J = self.J.cuda()
    
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger  = []
        for kk, prob in enumerate(weights):
            self.q[kk] = K*prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
    
        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
    
            self.J[small] = large
            self.q[large] = (self.q[large] - 1.0) + self.q[small]
    
            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.q.clamp_(0, 1)
        self.J.clamp_(0, K-1)

        self.num_samples = int(alpha * self.num_samples)


    def __iter__(self):
        for i in range(self.num_samples // self.batch_size):
            K  = self.J.size(0)
            r = torch.LongTensor(np.random.randint(0, K, size=self.batch_size))
            if USE_CUDA:
                r = r.cuda()
            q = self.q.index_select(0, r)
            j = self.J.index_select(0, r)
            b = torch.bernoulli(q)
            oq = r.mul(b.long())
            oj = j.mul((1-b).long())
            yield list(torch.sort(oq + oj)[0])


    def __len__(self):
        return self.num_samples


