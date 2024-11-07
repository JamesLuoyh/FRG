import torch
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli
from torch.nn import Module, Linear, ReLU, Dropout, BCELoss, CrossEntropyLoss, Sigmoid, Sequential, Parameter
from torch.nn.functional import softplus
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.nn import init
import pandas as pd
import numpy as np
import experiments.utils as utils
import os
import time
import torch.nn.functional as F
class PytorchFCRLBaseline(SupervisedPytorchBaseModel):
    """
    Implementation of the LMIFR
    """
    def __init__(self,device, **kwargs):
      """ 

      :param device: The torch device, e.g., 
        "cuda" (NVIDIA GPU), "cpu" for CPU only,
        "mps" (Mac M1 GPU)
      """
      super().__init__(device, **kwargs)

    # def create_model(self,**kwargs):
    def create_model(self,
            x_dim,
            s_dim,
            y_dim,
            z1_enc_dim,
            z2_enc_dim,
            z1_dec_dim,
            x_dec_dim,
            z_dim,
            dropout_rate,
            lr,
            downstream_bs,
            use_validation=False,
            activation=ReLU(),
            s_num=2,
            nce_size=50
        ):
        self.vfae = ContrastiveVariationalAutoEncoder(x_dim,
            s_dim,
            y_dim,
            z1_enc_dim,
            z2_enc_dim,
            z1_dec_dim,
            x_dec_dim,
            z_dim,
            dropout_rate,
            lr,
            activation=activation,
            s_num=s_num,
            nce_size=nce_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=1)
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.downstream_bs = downstream_bs
        self.adv_loss = BCELoss()
        self.use_validation = use_validation
        self.z_dim = z_dim
        return self.vfae

    # set a prior distribution for the sensitive attribute for VAE case
    def set_pu(self, pu):
        pu_dist = Bernoulli(probs=torch.tensor(pu).to(self.device))
        self.vfae.set_pu(pu_dist)
        return

    def get_representations(self, X):
        return self.vfae.get_representations(X)


    def train(self, X_train, Y_train, batch_size, num_epochs,data_frac, n_valid, X_test):
        print("Training model...")
        loss_list = []
        accuracy_list = []
        iter_list = []
        # if self.use_validation:
        X_valid = X_train[-n_valid:]
        
        S_valid = X_train[-n_valid:, self.x_dim:self.x_dim+self.s_dim]
        x_train_tensor = torch.from_numpy(X_train[:-n_valid])
        x_valid_tensor = torch.from_numpy(X_train[-n_valid:])
        s_valid_tensor = torch.from_numpy(S_valid)
        if type(Y_train) == list:
            Y_train = Y_train[0]
        Y_valid = Y_train[-n_valid:]
        y_train_label = torch.from_numpy(Y_train[:-n_valid])
        y_valid_label = torch.from_numpy(Y_train[-n_valid:])
        # else:    
        #     x_train_tensor = torch.from_numpy(X_train)
        #     y_train_label = torch.from_numpy(Y_train)
        #     x_valid_tensor = x_train_tensor
        #     y_valid_label = y_train_label
        train = torch.utils.data.TensorDataset(x_train_tensor, y_train_label)
        if batch_size == 0:
            batch_size = len(x_train_tensor)
        
        # betas = [1e-4]#,1e-3]#,1e-2,1e-1,1.0]#]#1e-1,]#1e-3,1e-2,
        # lrs = [1e-3]#,1e-3]#, 1e-4]#1e-4]


        # Income

        # above 0.32
        # 0.0001,500,0.3,0.001
        betas = [1e-4]#1e-4,1e-3,1e-2,1e-1,1.0]#]
        lrs = [1e-3]#,1e-4]#, 1e-4]#1e-4] 
        num_epochs = 500 
        # below 0.32
        # betas = [1.0]#1e-4,1e-3,1e-2,1e-1,1.0]#]
        # lrs = [1e-3]#,1e-4]#, 1e-4]#1e-4] 
        # num_epochs = 500  
        # HEALTH
        # betas = [1e-4]#,1e-3,1e-2,1e-1,1.0]#]
        # lrs = [1e-3]#,1e-4]#, 1e-4]#1e-4] 
        # num_epochs = 500  
        ###########
        # Adult 0.04
        # betas = [1e-2]
        # lrs = [1e-3]

        # Adult 0.08
        # betas = [1e-2]
        # lrs = [1e-3]
        
        # # Adult 0.12
        # betas = [1e-2]
        # lrs = [1e-4]

        # Adult 0.16
        # betas = [1e-3]
        # lrs = [1e-3]

        # num_epochs = 500
        print(
            f"Running gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
        )
        if self.use_validation:
            repeats = 2
        else:
            repeats = 1
        for beta in betas:
            for lr in lrs:
                param_search_id = int(time.time())
                for repeat in range(repeats): # repeat the parameter search for 3 times
                    trainloader = torch.utils.data.DataLoader(
                        train, batch_size=batch_size, shuffle=True
                    )
                    self.vfae.reset_params(self.device)
                    itot = 0
                    self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=lr)
                    self.vfae.beta = beta
                    for epoch in range(num_epochs):
                        for i, (features, labels) in enumerate(trainloader):
                            features = features.float().to(self.device)
                            labels = labels.to(self.device)

                            # Clear gradients w.r.t. parameters
                            self.optimizer.zero_grad()
                            self.vfae.train()
                            self.pytorch_model.train()
                            # Forward pass to get output/logits
                            vae_loss, mi_sz, y_prob = self.pytorch_model(features)
                            vae_loss.backward()

                            # # Updating parameters
                            self.optimizer.step()
                            if i % 100 == 0:
                                it = f"{i+1}/{len(trainloader)}"
                                print(f"Epoch, it, itot, loss, mi: {epoch},{it},{itot},{vae_loss}, {mi_sz.mean()}")
                            itot += 1
                    # evaluate validation data
                    self.vfae.eval()
                    self.pytorch_model.eval()
                    if self.use_validation:
                        kwargs = {
                            'downstream_lr'     : 1e-4,
                            'y_dim'             : 1,
                            'downstream_epochs' : 5,
                            'downstream_bs'     : self.downstream_bs,
                            's_dim'             : self.s_dim,
                            'z_dim'             : self.z_dim,
                            'hidden_dim'        : self.z_dim,
                            'device'            : self.device,
                            'X'                 : x_valid_tensor.cpu().numpy(),
                        }

                        x_valid_tensor = x_valid_tensor.float().to(self.device)

                        
                        # Train downstream model
                        y_pred = utils.unsupervised_downstream_predictions(self, self.get_model_params(), X_train[:-n_valid], Y_train[:-n_valid], X_valid, **kwargs)
                        x_valid_tensor = x_valid_tensor.float().to(self.device)
                        s_valid_tensor = s_valid_tensor.float().to(self.device)
                        y_valid_label = y_valid_label.float().to(self.device)
                        vae_loss, mi_sz, y_prob = self.pytorch_model(x_valid_tensor)
                        y_pred_all = vae_loss, mi_sz, y_pred
                        delta_DP = utils.demographic_parity(y_pred_all, None, **kwargs)
                        y_hat = (y_pred > 0.5).astype(np.float32)
                        auc = roc_auc_score(Y_valid, y_hat)
                        f1 = f1_score(Y_valid, y_hat)
                        acc = accuracy_score(Y_valid, y_hat)

                        # y_pred_all = vae_loss, mi_sz, y_prob.detach().cpu().numpy()
                        # delta_DP = utils.demographic_parity(y_pred_all, None, **kwargs)
                        # auc = roc_auc_score(y_valid_label.numpy(), y_prob.detach().cpu().numpy())
                        result_log = f'/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/fcrl_income.csv'
                        if not os.path.isfile(result_log):
                            with open(result_log, "w") as myfile:
                                myfile.write("param_search_id,auc,acc,f1,delta_dp,mi,beta,epoch,dropout,lr")
                        df = pd.read_csv(result_log)
                        row = {'param_search_id':param_search_id, 'auc': auc, 'acc': acc, 'f1': f1, 'delta_dp': delta_DP, 'mi': mi_sz.mean().item(), 'beta':beta, 'epoch': num_epochs, 'dropout':self.vfae.dropout.p, 'lr': lr}
                        # print(row)
                        df.loc[len(df)] = row
                        df.to_csv(result_log, index=False)


class ContrastiveVariationalAutoEncoder(Module):
    """
    Implementation of the Variational Fair AutoEncoder
    """

    def __init__(self,
                 x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 lr,
                 activation=ReLU(),
                 s_num=2,
                 nce_size=50):
        super().__init__()
        self.y_out_dim = y_dim #2 if y_dim == 1 else y_dim
        self.encoder_z1 = VariationalMLP(x_dim + s_dim, z1_enc_dim, z_dim, activation)
        self.encoder_z2 = VariationalMLP(z_dim + y_dim, z2_enc_dim, z_dim, activation)

        self.decoder_z1 = VariationalMLP(z_dim + y_dim, z1_dec_dim, z_dim, activation)
        self.decoder_y = DecoderMLP(z_dim, x_dec_dim, self.y_out_dim, activation)
        self.decoder_x = DecoderMLP(z_dim + s_dim, x_dec_dim, x_dim, activation)

        self.dropout = Dropout(dropout_rate)
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.s_num = s_num
        self.z1_enc_dim = z1_enc_dim
        self.z2_enc_dim = z2_enc_dim
        self.z1_dec_dim = z1_dec_dim
        self.x_dec_dim = x_dec_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.loss = VFAELoss()
        self.lambda_ = 2
        self.beta = lr
        self.nce_size = nce_size
        self._nce_estimator = CPC([x_dim], z_dim, s_num, nce_size)
    #     self.to(device)

    # def to(self, device):
    #     self.device = device
        # return super().to(device=device)
    
    def reset_params(self,device):
        self.encoder_z1 = VariationalMLP(self.x_dim + self.s_dim, self.z1_enc_dim, self.z_dim, self.activation)
        self.encoder_z2 = VariationalMLP(self.z_dim + self.y_dim, self.z2_enc_dim, self.z_dim, self.activation)

        self.decoder_z1 = VariationalMLP(self.z_dim + self.y_dim, self.z1_dec_dim, self.z_dim, self.activation)
        self.decoder_y = DecoderMLP(self.z_dim, self.x_dec_dim, self.y_out_dim, self.activation)
        self.decoder_x = DecoderMLP(self.z_dim + self.s_dim, self.x_dec_dim, self.x_dim, self.activation)
        self.dropout = Dropout(self.dropout_rate)
        self._nce_estimator = CPC([self.x_dim], self.z_dim, self.s_num, self.nce_size)
        self.to(device)

    @staticmethod
    def kl_gaussian(logvar_a, mu_a):
        """
        Average KL divergence between two (multivariate) gaussians based on their mean and standard deviation for a
        batch of input samples. https://arxiv.org/abs/1405.2664

        :param logvar_a: standard deviation a
        :param mu_a: mean a
        :return: kl divergence
        """
        per_example_kl = - logvar_a - 1 + (logvar_a.exp() + (mu_a).square())
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl

    def get_representations(self, inputs):
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)
        return z1_encoded



    def forward(self, inputs):
        """
        :param inputs: dict containing inputs: {'x': x, 's': s, 'y': y} where x is the input feature vector, s the
        sensitive variable and y the target label.
        :return: dict containing all 8 VFAE outputs that are needed for computing the loss term, i.e. :
            - x_decoded: the reconstructed input with shape(x_decoded) = shape(concat(x, s))
            - y_decoded: the predictive posterior output for target label y
            - z1_encoded: the sample from latent variable z1
            - z1_enc_logvar: variance of the z1 encoder distribution
            - z1_enc_mu: mean of the z1 encoder distribution
            - z2_enc_logvar: variance of the z2 encoder distribution
            - z2_enc_mu: mean of the z2 encoder distribution
            - z1_dec_logvar: variance of the z1 decoder distribution
            - z1_dec_mu: mean of the z1 decoder distribution
        """
        # z1_y = torch.cat([z1_encoded, y], dim=1)
        # z2_encoded, z2_enc_logvar, z2_enc_mu = self.encoder_z2(z1_y)

        # # decode
        # z2_y = torch.cat([z2_encoded, y], dim=1)
        # z1_decoded, z1_dec_logvar, z1_dec_mu = self.decoder_z1(z2_y)

        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        x_s = self.dropout(x_s)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)


        z1_s = torch.cat([z1_encoded, s], dim=1)
        x_decoded = self.decoder_x(z1_s)

        y_decoded = self.decoder_y(z1_encoded)
        
        # print(self.mi_sz)
        outputs = {
            # predictive outputs
            'x_decoded': x_decoded,
            'y_decoded': y_decoded,
            'z1_encoded': z1_encoded,

            # outputs for regularization loss terms
            'z1_enc_logvar': z1_enc_logvar,
            'z1_enc_mu': z1_enc_mu,
        }
        # will return the constraint C2 term. log(qu) - log(pu) instead of y_decoded
        self.vae_loss = self.loss(outputs, {'x': x, 's': s, 'y': y})
        
        kl_gaussian = self.kl_gaussian(z1_enc_logvar, z1_enc_mu)
        if self.s_dim > 1:
            s = torch.argmax(s, dim=1)
        nce_estimate = self._nce_estimator(x, s, z1_encoded)

        self.mi_sz = self.beta * kl_gaussian - self.lambda_ * nce_estimate
        self.vae_loss += self.mi_sz.mean()
        self.mi_sz = self.mi_sz.unsqueeze(1)
        self.pred = y_decoded # torch.softmax(y_decoded, dim=-1)[:, 1]
        self.s = s
        self.z = z1_encoded
        self.y_prob = y_decoded.squeeze()

        return self.vae_loss, self.mi_sz, self.y_prob
    
class VariationalMLP(Module):
    """
    Single hidden layer MLP using the reparameterization trick for sampling a latent z.
    """

    def __init__(self, in_features, hidden_dim, z_dim, activation):
        super().__init__()
        self.encoder = Linear(in_features, hidden_dim)
        self.activation = activation

        self.logvar_encoder = Linear(hidden_dim, z_dim)
        self.mu_encoder = Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        """
        :param inputs:
        :return:
            - z - the latent sample
            - logvar - variance of the distribution over z
            - mu - mean of the distribution over z
        """
        x = self.encoder(inputs)
        logvar = (0.5 * self.logvar_encoder(x)).exp()
        mu = self.mu_encoder(x)

        # reparameterization trick: we draw a random z
        epsilon = torch.randn_like(mu)
        z = epsilon * mu + logvar
        return z, logvar, mu


class DecoderMLP(Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = Linear(hidden_dim, latent_dim)
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.sigmoid(self.lin_out(x))


class VFAELoss(Module):
    """
    Loss function for training the Variational Fair Auto Encoder.
    """

    def __init__(self, alpha=1.0, beta=0.0, mmd_dim=0, mmd_gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.bce = BCELoss()
        self.ce = CrossEntropyLoss()
        # self.mmd = FastMMD(mmd_dim, mmd_gamma)

    def forward(self, y_pred, y_true):
        """

        :param y_pred: dict containing the vfae outputs
        :param y_true: dict of ground truth labels for x, s and y
        :return: the loss value as Tensor
        """
        x, s, y = y_true['x'], y_true['s'], y_true['y']
        x_s = torch.cat([x, s], dim=-1)
        device = y.device
        supervised_loss = self.bce(y_pred['y_decoded'], y.to(device))
        reconstruction_loss = F.binary_cross_entropy(y_pred['x_decoded'], x, reduction='sum')
        zeros = torch.zeros_like(y_pred['z1_enc_logvar'])
        kl_loss_z1 = self._kl_gaussian(y_pred['z1_enc_logvar'],
                                       y_pred['z1_enc_mu'],
                                       zeros,
                                       zeros)

        loss = (reconstruction_loss  + kl_loss_z1) / len(y)

        return supervised_loss

    @staticmethod
    def _kl_gaussian(logvar_a, mu_a, logvar_b, mu_b):
        """
        Average KL divergence between two (multivariate) gaussians based on their mean and standard deviation for a
        batch of input samples. https://arxiv.org/abs/1405.2664

        :param logvar_a: standard deviation a
        :param mu_a: mean a
        :param logvar_b: standard deviation b
        :param mu_b: mean b
        :return: kl divergence, mean averaged over batch dimension.
        """
        per_example_kl = logvar_b - logvar_a - 1 + (logvar_a.exp() + (mu_a - mu_b).square()) / logvar_b.exp()
        kl = 0.5 * torch.sum(per_example_kl, dim=1)
        return kl.mean()

    @staticmethod
    def _separate_protected(batch, s):
        """separate batch based on labels indicating protected and non protected .

        :param batch: values to select from based on s.
        :param s: tensor of labels with s=1 meaning protected and s=0 non protected.
        :return:
            - protected - items from batch with protected label
            - non_protected - items from batch with non protected label
        """
        idx_protected = (s == 1).nonzero()[:, 0]
        idx_non_protected = (s == 0).nonzero()[:, 0]
        protected = batch[idx_protected]
        non_protected = batch[idx_non_protected]

        return protected, non_protected

class CPC(Module):
    def __init__(self, input_size, z_size, c_size, hidden_size):
        super().__init__()

        self.f_x = Sequential(
            Linear(input_size[0], hidden_size),
            ReLU(),
            Linear(hidden_size, z_size),
        )
        # just make one transform
        self.f_z = Sequential(Linear(z_size, z_size))
        self.w_s = Parameter(data=torch.randn(c_size, z_size, z_size))
        # self.to(device)

    # def to(self, device):
    #     self.device = device
    #     return super().to(device=device)

    def forward(self, x, c, z):
        N = x.shape[0]
        c = c.long()
        f_x = self.f_x(x)
        f_z = self.f_z(z)
        temp = torch.bmm(
            torch.bmm(
                f_x.unsqueeze(2).transpose(1, 2), self.w_s[c.reshape(-1)]
            ),
            f_z.unsqueeze(2),
        )
        T = softplus(temp.view(-1))

        neg_T = torch.zeros(N, device=x.device)

        for cat in set(c.reshape(-1).tolist()):
            f_z_given_c = f_z[(c == cat).reshape(-1)]
            f_x_given_c = f_x[(c == cat).reshape(-1)]

            # (N,Z) X (Z,Z)
            temp = softplus(
                f_x_given_c @ self.w_s[cat] @ f_z_given_c.transpose(0, 1)
            )
            # columns are different Z's, rows are different x's.
            # mean along dim 0, is the mean over the same Z different X
            # mean along dim 1, is the mean over the same X different Z
            # Change to sum because the contrastive estimation model overfit
            # for the Z corresponding to the X easily. When Z and X does not match
            # the T evaluated is almost 0 and the average is also 0.
            # This will make MI(Z;X|S) very big and the MI(Z;S) becomes negative.
            neg_T[(c == cat).reshape(-1)] = temp.sum(dim=1).view(-1)
            
        return torch.log(T + 1e-16) - torch.log(neg_T + 1e-16)
