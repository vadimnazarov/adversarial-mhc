import torch
from torch import nn
import torch.autograd as autograd
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics import f1_score

from feng.mhcmodel import *
from feng.trainer import *


# Wasserstein loss:
# https://github.com/maitek/waae-pytorch/blob/master/WAAE.py
# WAE with gradient penalty
# https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
# Denoising AAE
# https://github.com/ToniCreswell/pyTorch_DAAE
# GAN hacks:
# https://github.com/soumith/ganhacks
# Cool blogpost
# https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490


ALPHABET = np.array(list("ACDEFGHIKLMNPQRSTVWYX"))
to_seq = lambda x: "".join(ALPHABET[x.view((len(ALPHABET), 11)).argmax(dim=0)])
EPS = 1e-15
LAMBDA = 10 # for gradient penalty


def gradient_penalty(model_D, batch_real, batch_fake, device):
    alpha = torch.rand(batch_real.size(0), 1).to(device)
    alpha = alpha.expand(batch_real.size())
    
    interpolates = alpha * batch_real + ((1 - alpha) * batch_fake)
    
    z_inter = model_D(interpolates)
    
    gradients = autograd.grad(outputs=z_inter, 
                              inputs=interpolates,
                              grad_outputs=torch.ones(z_inter.size(), device=device),
                              create_graph=True, 
                              retain_graph=True, 
                              only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class AAETrainer(Trainer):

    def __init__(self, nn_mode, model, dataset, synth_dataset=None, pred_mode="reg", nolabel_dataset=None, wasserstein=False, device="cpu", use_gp=False):
        super(AAETrainer, self).__init__(nn_mode, model, dataset, synth_dataset, pred_mode)

        self.use_gp = use_gp
        self.preproc = lambda x, _: x
        self.nolabel_dataset = nolabel_dataset
        if nolabel_dataset:
            self.info["scores"]["nolabel"] = {}
        self.device=device
        
        self.wass = wasserstein


    def train(self, n_epochs, criterion, optimizer, batch_size, test_dataset=None, sampling="brute", num_workers=mp.cpu_count(), start_epoch=1, aae_mode=""):

        def _model_scores(new_scores, aae_mode):
            res = []
            for ll in self.loss_labels:
                if ll not in ["f1", "ppv"]:
                    res.append(np.mean(new_scores[ll]))
                elif aae_mode == "semi":
                    if ll == "f1":
                        df = torch.cat(new_scores["f1"], 0)

                        # F1
                        # clf_score = f1_score(np.where(df[:,0] >= .5, 1, 0), np.where(df[:,1] >= .5, 1, 0))

                        # PPV
                        n_of_binders = np.where(df[:, 0] > .5)[0].shape[0]
                        df = df[torch.sort(df[:,1])[1]]
                        clf_score = df[-n_of_binders:, 0].sum() / n_of_binders

                        clf_score = clf_score.item()
                        new_scores["f1"] = clf_score
                        res.append(clf_score)
                    else:
                        print("Unknown loss:", ll)
            return res

        def _add_scores(mode, values):
            for key, value in zip(self.loss_labels, values):
                if key not in self.info["scores"][mode]:
                    self.info["scores"][mode][key] = []
                self.info["scores"][mode][key].append(value)
                

        if aae_mode == "semi":
            self.loss_labels = ["recon", "reg", "gen", "reg_c", "gen_c", "ent", "f1"]
        else:
            self.loss_labels = ["recon", "reg", "gen"]

        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        
        if self.nolabel_dataset:
            self.nolabel_dataloader = DataLoader(self.nolabel_dataset, batch_size=batch_size, num_workers=num_workers, 
                                              shuffle=True, pin_memory=USE_CUDA, drop_last=False)
        
        if self.synth_dataset:
            self.batch_size = self.batch_size // 2
            self.synth_dataloader = make_dataloader(self.synth_dataset, self.batch_size, num_workers, self.nn_mode, "brute")
        else:
            self.synth_dataloader = itertools.repeat(None)
            
        if test_dataset:
            self.test_dataloader = DataLoader(test_dataset, batch_size=1024, num_workers=num_workers, 
                                              shuffle=False, pin_memory=USE_CUDA, drop_last=False)

        self.dataloader = make_dataloader(self.dataset, self.batch_size, num_workers, self.nn_mode, sampling)

        for i_epoch in range(start_epoch, n_epochs+start_epoch):
            # print("Epoch:", i_epoch, "/", n_epochs+start_epoch-1)
            # print("Model:", self.model.name())

            #########
            # TRAIN #
            #########
            (new_train_scores, n_batches), train_seconds = compute_time(self.step, dataloader=self.dataloader, aae_mode=aae_mode)

            scores = _model_scores(new_train_scores, aae_mode)

            # print("Results:")
            print("Epoch:", i_epoch, "/", n_epochs+start_epoch-1, end="\t")
            print(("{:7}" + "{:>7}"*len(self.loss_labels)).format("", *self.loss_labels))
            print("Epoch:", i_epoch, "/", n_epochs+start_epoch-1, end="\t")
            print(("{:7}" + "{:>7.3}"*len(self.loss_labels)).format("train", *scores))
            _add_scores("train", scores)
            
            ##############
            # UNLABELLED #
            #############
            nolabel_seconds = 0
            if self.nolabel_dataset:
                (new_nolabel_scores, n_batches), nolabel_seconds = compute_time(self.step, dataloader = self.nolabel_dataloader, aae_mode="nolabel")
                scores = _model_scores(new_nolabel_scores, "nolabel")

                print("Epoch:", i_epoch, "/", n_epochs+start_epoch-1, end="\t")
                print(("{:7}" + "{:>7.3}"*min(len(self.loss_labels), 5)).format("nolabel", *scores[:5]))
                _add_scores("nolabel", scores)

            ###########
            # TESTING #
            ###########
            test_seconds = 0
            if test_dataset:
                if (i_epoch % 3 == 1) or (i_epoch == start_epoch):
                    new_test_scores, test_seconds = compute_time(self.evaluate, test_dataset, aae_mode=aae_mode)
                #self.test_scores.append(new_test_scores)
                    test_scores = _model_scores(new_test_scores, aae_mode)
                _add_scores("val", test_scores)

                print("Epoch:", i_epoch, "/", n_epochs+start_epoch-1, end="\t")
                print(("{:7}" + "{:>7.3}"*len(self.loss_labels)).format("val", *test_scores))

            print("Time:", round(train_seconds, 3), round(nolabel_seconds, 3), round(test_seconds, 3))
            # print("  training sec/epoch: ", round(train_seconds, 3))
            # print("  training sec/batch: ", round(train_seconds / n_batches, 3), "(" + str(n_batches), "batches)")
            # if test_dataset:
                # print("  validation sec/test:", round(test_seconds, 3))

            print()
        

    def step(self, dataloader, aae_mode=""):
        loss_list = {ll:[] for ll in self.loss_labels}

        for n_batches, batch in enumerate(dataloader):
            batch_x = batch[1].view((batch[1].size(0), -1)).to(self.device)

            ##################
            # RECONSTRUCTION #
            ##################
            
            self.optimizer["encoder"].zero_grad()
            self.optimizer["decoder"].zero_grad()
            self.optimizer["discriminator"].zero_grad()
            self.optimizer["generator"].zero_grad()
            if aae_mode in ["semi", "nolabel"]:
                self.optimizer["discriminator_cat"].zero_grad()
                self.optimizer["generator_cat"].zero_grad()
                self.optimizer["classifier"].zero_grad()

            z_vector = self.model.Q(batch_x, aae_mode)

            if aae_mode in ["semi"]:
                y, z_vector, label_vec = torch.cat(z_vector, 1), z_vector[0], z_vector[1]
                true_label_vec = torch.zeros((batch_x.size(0), ), device=self.device)
                true_label_vec[batch[3] >= BIND_THR] = 1
            elif aae_mode == "labels":
                label_vec = torch.zeros((batch_x.size(0), 2), device=self.device)
                label_vec[batch[3] < BIND_THR, 0] = 1
                label_vec[batch[3] >= BIND_THR, 1] = 1
                y = torch.cat([z_vector, label_vec], 1)
            elif aae_mode == "nolabel":
                y, z_vector, label_vec = torch.cat(z_vector, 1), z_vector[0], z_vector[1]
            else:
                y = z_vector

            z_recon = self.model.P(y)

            recon_loss = F.binary_cross_entropy(z_recon.view((z_recon.size(0), -1)), batch_x)
            recon_loss.backward(retain_graph=True)
            loss_list["recon"].append(recon_loss.item())

            self.optimizer["encoder"].step()
            self.optimizer["decoder"].step()

            ##################
            # REGULARIZATION #
            ##################
            if not self.wass:
                ###############
                # CLASSIC REG #
                ###############
                z_real = torch.zeros(batch_x.size(0), self.model.lat, device=self.device).normal_(0, 5)

                self.model.Q.eval()
                z_vector = self.model.Q(batch_x, aae_mode)
                if aae_mode in ["semi", "nolabel"]:
                    z_vector, reg_pred_label = z_vector
                self.model.Q.train()
                d_fake = self.model.D(z_vector)
                d_real = self.model.D(z_real)
            
                reg_loss = F.binary_cross_entropy(torch.cat([d_real, d_fake], 0), 
                                                  torch.cat([torch.ones(d_real.shape, device=self.device).uniform_(0.7, 1.), 
                                                             torch.zeros(d_fake.shape, device=self.device).uniform_(0.0, 0.3)], 0))
                reg_loss.backward(retain_graph=True)
                loss_list["reg"].append(reg_loss.item())
                self.optimizer["discriminator"].step()
            else:
                ###################
                # WASSERSTEIN REG #
                ###################
                loss_arr = []
                for _ in range(5):
                    z_real = torch.zeros(batch_x.size(0), self.model.lat, device=self.device).normal_(0, 5)

                    self.model.Q.eval()
                    z_vector = self.model.Q(batch_x, aae_mode)
                    z_vector, reg_pred_label = z_vector
                    self.model.Q.train()
                    
                    d_fake = self.model.D(z_vector)
                    d_real = self.model.D(z_real)
                    
                    if not self.use_gp:
                        reg_loss = -d_real.mean() + d_fake.mean()
                    else:
                        # Gradient penalty
                        reg_loss = -d_real.mean() + d_fake.mean() + gradient_penalty(self.model.D, z_real, z_vector, self.device)

                    reg_loss.backward(retain_graph=True)    
                    loss_arr.append(reg_loss.item())    
                    self.optimizer["discriminator"].step()
                    
                    if not self.use_gp:
                        for p in self.model.D.parameters():
                            p.data.clamp_(-0.01, 0.01)
                loss_list["reg"].append(np.mean(loss_arr))
        

            #######################
            # SEMI-SUPERVISED REG #
            #######################
            
            if aae_mode in ["semi", "nolabel"]:                
                if not self.wass:
                    ###############
                    # CLASSIC REG #
                    ###############
                    real_label_vec = torch.multinomial(torch.tensor([.5, .5]), batch_x.size(0)*2, replacement=True).view((batch_x.size(0), 2)).float().to(self.device)
                    fake_label_vec = reg_pred_label.float()

                    d_cat_real = self.model.D_cat(real_label_vec)
                    d_cat_fake = self.model.D_cat(fake_label_vec)
                
                    reg_c_loss = F.binary_cross_entropy(torch.cat([d_cat_real, d_cat_fake], 0), 
                                                    torch.cat([torch.ones(d_cat_real.shape, device=self.device).uniform_(0.7, 1.), 
                                                               torch.zeros(d_cat_fake.shape, device=self.device).uniform_(0.0, 0.3)], 0))
                    reg_c_loss.backward(retain_graph=True)
                    loss_list["reg_c"].append(reg_c_loss.item())
                    self.optimizer["discriminator_cat"].step()
                    
                else:
                    ###################
                    # WASSERSTEIN REG #
                    ###################
                    loss_arr = []
                    for _ in range(5):
                        real_label_vec = torch.multinomial(torch.tensor([.5, .5]), batch_x.size(0)*2, replacement=True).view((batch_x.size(0), 2)).float().to(self.device)
                        fake_label_vec = reg_pred_label.float()

                        d_cat_real = self.model.D_cat(real_label_vec)
                        d_cat_fake = self.model.D_cat(fake_label_vec)
                        
                        if not self.use_gp:
                            reg_c_loss = -d_cat_real.mean() + d_cat_fake.mean()
                        else:
                            # Gradient penalty
                            reg_c_loss = -d_cat_real.mean() + d_cat_fake.mean() + gradient_penalty(self.model.D_cat, real_label_vec, fake_label_vec, self.device)
                        reg_c_loss.backward(retain_graph=True)
                        loss_arr.append(reg_c_loss.item())
                        self.optimizer["discriminator_cat"].step()
                        if not self.use_gp:
                            for p in self.model.D_cat.parameters():
                                p.data.clamp_(-0.01, 0.01)
                    loss_list["reg_c"].append(np.mean(loss_arr))

                
            #############
            # GENERATOR #
            #############

            z_fake = self.model.Q(batch_x)
            d_fake = self.model.D(z_fake)

            if not self.wass:
                gen_loss = F.binary_cross_entropy(d_fake, torch.ones(d_fake.shape, device=self.device).uniform_(0.7, 1.))
            else:
                gen_loss = -d_fake.mean()
                
            gen_loss.backward()
            loss_list["gen"].append(gen_loss.item())

            self.optimizer["generator"].step()

            #######################
            # SEMI-SUPERVISED GEN #
            #######################
            
            if aae_mode in ["semi", "nolabel"]:
                if not self.wass:
                    gen_c_loss = F.binary_cross_entropy(fake_label_vec, torch.ones(fake_label_vec.shape, device=self.device).uniform_(0.7, 1.))
                else:
                    gen_c_loss = -fake_label_vec.mean()
                
                gen_c_loss.backward(retain_graph=True)
                loss_list["gen_c"].append(gen_c_loss.item())
                self.optimizer["generator_cat"].step()

                
            #######################
            # SEMI-SUPERVISED CLF #
            #######################
            
            if aae_mode == "semi":
                clf_loss = F.binary_cross_entropy(reg_pred_label[:,1], true_label_vec)
                clf_loss.backward()
                
                loss_list["ent"].append(clf_loss.item())
                loss_list["f1"].append(torch.cat([true_label_vec.view((-1,1)), reg_pred_label[:,1].view((-1,1))], dim=1))
                self.optimizer["classifier"].step()

        return loss_list, n_batches+1


    def evaluate(self, test_dataset, batch_size=1024, num_workers=0, aae_mode=""):
        self.model.Q.eval()
        self.model.P.eval()
        self.model.D.eval()
   #     self.model.Q.cpu()
  #      self.model.P.cpu()
 #       self.model.D.cpu()
        if self.model.D_cat:
            self.model.D_cat.eval()
#            self.model.D_cat.cpu()

        loss_list = {ll:[] for ll in self.loss_labels}

        for i, batch in enumerate(self.test_dataloader):
            batch_x = batch[1].view((batch[1].size(0), -1)).to(self.device)

            z_vector = self.model.Q(batch_x, aae_mode)

            if aae_mode == "semi":
                y, z_vector, label_vec = torch.cat(z_vector, 1), z_vector[0], z_vector[1]
                true_label_vec = torch.zeros((batch_x.size(0), ), device=self.device)
                true_label_vec[batch[3] >= BIND_THR] = 1
            elif aae_mode == "labels":
                label_vec = torch.zeros((batch_x.size(0), 2), device=self.device)
                label_vec[batch[3] < BIND_THR, 0] = 1
                label_vec[batch[3] >= BIND_THR, 1] = 1
                y = torch.cat([z_vector, label_vec], 1)
            else:
                y = z_vector

            z_recon = self.model.P(y)
            recon_loss = F.binary_cross_entropy(z_recon.view((z_recon.size(0), -1)), batch_x)
            loss_list["recon"].append(recon_loss.item())

            z_real = torch.zeros(self.batch_size, self.model.lat, device=self.device).normal_(0, 5)

            z_vector = self.model.Q(batch_x, aae_mode)
            if aae_mode == "semi":
                z_vector, reg_pred_label = z_vector
            d_fake = self.model.D(z_vector)
            d_real = self.model.D(z_real)

            reg_loss = F.binary_cross_entropy(torch.cat([d_real, d_fake], 0), 
                                              torch.cat([torch.ones(d_real.shape, device=self.device), 
                                                         torch.zeros(d_fake.shape, device=self.device)], 0))
            loss_list["reg"].append(reg_loss.item())

            if aae_mode == "semi":
                real_label_vec = torch.multinomial(torch.tensor([.5, .5]), self.batch_size*2, replacement=True).view((self.batch_size, 2)).float().to(self.device)
                fake_label_vec = reg_pred_label.float()
                d_cat_real = self.model.D_cat(real_label_vec)
                d_cat_fake = self.model.D_cat(fake_label_vec)
                reg_c_loss = F.binary_cross_entropy(torch.cat([d_cat_real, d_cat_fake], 0), 
                                                    torch.cat([torch.ones(d_cat_real.shape, device=self.device), 
                                                               torch.zeros(d_cat_fake.shape, device=self.device)], 0))
                loss_list["reg_c"].append(reg_c_loss.item())

            z_fake = self.model.Q(batch_x)
            d_fake = self.model.D(z_fake)

            gen_loss = F.binary_cross_entropy(d_fake, torch.ones(d_fake.shape, device=self.device))
            loss_list["gen"].append(gen_loss.item())

            if aae_mode == "semi":
                gen_c_loss = F.binary_cross_entropy(fake_label_vec, torch.ones(fake_label_vec.shape, device=self.device))
                loss_list["gen_c"].append(gen_c_loss.item())

                clf_loss = F.binary_cross_entropy(reg_pred_label[:,1], true_label_vec)
                loss_list["ent"].append(clf_loss.item())
                loss_list["f1"].append(torch.cat([true_label_vec.view((-1,1)), reg_pred_label[:,1].view((-1,1))], dim=1))

        self.model.Q.train()
        self.model.P.train()
        self.model.D.train()
  #      self.model.Q.to(self.device)
 #       self.model.P.to(self.device)
#        self.model.D.to(self.device)
        if self.model.D_cat:
            self.model.D_cat.train()
           # self.model.D_cat.to(self.device)
        return loss_list



def make_aae(latent_dim, pep_dim, aa_channels, lin_sizes, dropout, aae_mode="", device="cpu", grad_p=False):
    Q = AAEencoder(latent_dim, pep_dim, aa_channels, lin_sizes, dropout).to(device)
    decoder_latent_dim = latent_dim
    P = AAEdecoder(latent_dim+(aae_mode != "")*2, pep_dim, aa_channels, lin_sizes, dropout).to(device)
    D = AAEdiscriminator(latent_dim, lin_sizes, dropout, not grad_p).to(device)
    D_cat = None
    if aae_mode == "semi":
        D_cat = AAEdiscriminator(2, [128,128], dropout, not grad_p).to(device)
    return AAE(latent_dim, Q, P, D, D_cat)



def make_dense_layers(prev_size, sizes, dropout=None, bn=True):
    def add_block(layers, prev_size, next_size, dropout):
        layers.append(nn.Linear(prev_size, next_size))
        init.kaiming_uniform_(layers[-1].weight)
        if bn:
            layers.append(nn.BatchNorm1d(next_size))
        layers.append(nn.RReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))

    layers = []
    add_block(layers, prev_size, sizes[0], dropout)
    for i in range(1, len(sizes)):
        add_block(layers, sizes[i-1], sizes[i], dropout)
    return nn.Sequential(*layers)


class AAE():

    def __init__(self, lat_dim, Q, P, D, D_cat=None):
        self.Q = Q
        self.P = P
        self.D = D
        self.D_cat = D_cat
        self.lat = lat_dim

    def name(self):
        return "AAE"


class AAEencoder(MhcModel):

    def __init__(self, latent_dim, pep_dim, aa_channels, lin_sizes, dropout=None, comment="mhc_specific_aae_e"):
        super(AAEencoder, self).__init__("aae_encoder", comment)

        self.pep_dim = pep_dim
        self.aa_channels = aa_channels

        self.dropout = dropout

        self.input_dim = pep_dim*aa_channels
        self.latent_dim = latent_dim
        self.sizes = lin_sizes

        self.model = make_dense_layers(self.input_dim, self.sizes, self.dropout)

        self.final_latent = nn.Linear(self.sizes[-1], self.latent_dim)
        init.kaiming_uniform_(self.final_latent.weight)

        self.final_label = nn.Linear(self.sizes[-1], 2)
        init.kaiming_uniform_(self.final_label.weight)


    def forward(self, input, aae_mode=""):
        x = self.model(input)
        if aae_mode in ["semi", "nolabel"]:
            return self.final_latent(x), F.softmax(torch.sigmoid(self.final_label(x)), dim=1)
        else:
            return self.final_latent(x)


class AAEdecoder(MhcModel):

    def __init__(self, latent_dim, pep_dim, aa_channels, lin_sizes, dropout=None, comment="mhc_specific_aae_d"):
        super(AAEdecoder, self).__init__("aae_decoder", comment)

        self.pep_dim = pep_dim
        self.aa_channels = aa_channels

        self.dropout = dropout

        self.input_dim = pep_dim*aa_channels
        self.latent_dim = latent_dim
        self.sizes = lin_sizes

        self.model = make_dense_layers(latent_dim, self.sizes, self.dropout)
        self.final = nn.Linear(self.sizes[-1], self.input_dim)
        init.kaiming_uniform_(self.final.weight)


    def forward(self, input):
        x = self.model(input)
        x = self.final(x)
        return F.softmax(x.view((-1, self.aa_channels, self.pep_dim)), dim=1)


class AAEdiscriminator(MhcModel):

    def __init__(self, latent_dim, lin_sizes, dropout=None, bn=True, comment="mhc_specific_aae_de"):
        super(AAEdiscriminator, self).__init__("aae_discriminator", comment)

        self.dropout = dropout

        self.latent_dim = latent_dim
        self.sizes = lin_sizes

        self.model = make_dense_layers(self.latent_dim, self.sizes, self.dropout, bn)
        self.final = nn.Linear(self.sizes[-1], 1)
        init.kaiming_uniform_(self.final.weight)


    def forward(self, input):
        x = self.model(input)
        x = self.final(x)
        return torch.sigmoid(x)
