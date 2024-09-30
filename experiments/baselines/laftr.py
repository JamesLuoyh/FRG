import torch
from torch.nn import Sequential, Module, Linear, ReLU, Dropout, BCELoss, CrossEntropyLoss, Sigmoid
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import experiments.utils as utils
import os
import time
from torch.nn import init


import torch.nn.functional as F
class PytorchLAFTR(SupervisedPytorchBaseModel):
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
            hidden_dim,
            dropout_rate,
            lr,
            downstream_bs,
            use_validation=False,
            activation=ReLU(),
        ):
        self.vfae = LAFTR(x_dim,
            s_dim,
            y_dim,
            z1_enc_dim,
            z2_enc_dim,
            z1_dec_dim,
            x_dec_dim,
            z_dim,
            hidden_dim,
            dropout_rate,
            activation=ReLU()).to(self.device)
        self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=1e-4)
        alpha_adv = lr
        self.lr = lr
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.downstream_bs = downstream_bs
        self.discriminator = DecoderMLP(z_dim, z_dim, s_dim, activation).to(self.device)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.adv_loss = BCELoss()
        self.use_validation = use_validation
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        return self.vfae

    # set a prior distribution for the sensitive attribute for VAE case
    # def set_pu(self, pu):
    #     pu_dist = Bernoulli(probs=torch.tensor(pu).to(self.device))
    #     self.vfae.set_pu(pu_dist)
    #     return

    def get_representations(self, X):
        return self.vfae.get_representations(X)


    def train(self, X_train, Y_train, batch_size, num_epochs, data_frac, n_valid, X_test):
        print("Training model...")
        loss_list = []
        accuracy_list = []
        iter_list = []
        # if self.use_validation:
        X_valid = X_train[-n_valid:]
        
        S_valid = X_train[-n_valid:, self.x_dim:self.x_dim+self.s_dim]
        x_train_tensor = torch.from_numpy(X_train[:-n_valid])
        x_valid_tensor = torch.from_numpy(X_train[-n_valid:])
        x_valid_tensor = torch.from_numpy(X_valid)
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
        print(
            f"Running gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
        )
        # sample
        
        # 0.001	0.0001	0.0001	0.01	0.0001

        # HEALTH 0.04
        # alphas = [1e-4]#, 1e-3]
        # betas = [1e-4]#,1e-3]#, 1e-3]
        # gammas = [1e-3]#,1e-3]#, 1e-3]
        # lr_advs = [1e-4]
        # lrs = [1e-2]
        # num_epochs_l = [1000]
        # adv_rounds_l = [2]

        # HEALTH 0.08, 0.12, 0.16
        alphas = [1e-3]#, 1e-3]
        betas = [1e-4]#,1e-3]#, 1e-3]
        gammas = [1e-4]#,1e-3]#, 1e-3]
        lr_advs = [1e-4]
        lrs = [1e-2]
        num_epochs_l = [1000]
        adv_rounds_l = [2]
        ###############

        # ADULTS
        # alphas = [1e-4]#,1e-3]#, 1e-3]
        # betas = [1e-4]#],1e-3]#, 1e-3]
        # gammas = [1e-4]#,1e-3]#, 1e-3]
        # lr_advs = [1e-4]#,1e-3]
        # lrs = [1e-4]#,1e-3,1e-2]
        # num_epochs_l = [1000]
        # adv_rounds_l = [5]#,2,5]

        if self.use_validation:
            repeats = 2
        else:
            repeats = 1
        for alpha in alphas:
            for beta in betas:
                for gamma in gammas:
                    for lr in lrs:
                        for lr_adv in lr_advs:
                            for num_epochs in num_epochs_l:
                                for adv_rounds in adv_rounds_l:
                                    param_search_id = int(time.time())
                                    for repeat in range(repeats): # repeat the parameter search for 3 times
                                        trainloader = torch.utils.data.DataLoader(
                                            train, batch_size=batch_size, shuffle=True
                                        )
                                        self.vfae.reset_params(self.device)
                                        itot = 0
                                        self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=lr)
                                        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_adv)
                                        self.vfae.alpha = alpha
                                        self.vfae.beta = beta
                                        self.vfae.gamma = gamma
                                        for epoch in range(num_epochs):
                                            for i, (features, labels) in enumerate(trainloader):
                                                self.discriminator.eval()
                                                features = features.float().to(self.device)
                                                labels = labels.to(self.device)

                                                # Clear gradients w.r.t. parameters
                                                self.optimizer.zero_grad()
                                                self.vfae.train()
                                                self.pytorch_model.train()
                                                # Forward pass to get output/logits
                                                vae_loss, mi_sz, y_prob = self.pytorch_model(features, self.discriminator)

                                                # Getting gradients w.r.t. parameters
                                                if itot % adv_rounds == 0:
                                                    vae_loss.backward()

                                                    # Updating parameters
                                                    self.optimizer.step()
                                                    
                                                # Update the adversary
                                                self.update_adversary(features)
                                                if i % 100 == 0:
                                                    it = f"{i+1}/{len(trainloader)}"
                                                    print(f"Epoch, it, itot, loss: {epoch},{it},{itot},{vae_loss}")
                                                itot += 1
                                        # evaluate validation data
                                        self.discriminator.eval()
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
                                                'hidden_dim'        : self.hidden_dim,
                                                'device'            : self.device,
                                                'X'                 : x_valid_tensor.cpu().numpy(),
                                            }

                                            x_valid_tensor = x_valid_tensor.float().to(self.device)

                                            # vae_loss, mi_sz, y_prob = self.pytorch_model(x_valid_tensor, self.discriminator)
                                            
                                            # Train downstream model
                                            y_pred = utils.unsupervised_downstream_predictions(self, self.get_model_params(), X_train[:-n_valid], Y_train[:-n_valid], X_valid, **kwargs)
                                            x_valid_tensor = x_valid_tensor.float().to(self.device)
                                            s_valid_tensor = s_valid_tensor.float().to(self.device)
                                            y_valid_label = y_valid_label.float().to(self.device)
                                            vae_loss, mi_sz, y_prob = self.pytorch_model(x_valid_tensor, self.discriminator)
                                            y_pred_all = vae_loss, mi_sz, y_pred
                                            delta_DP = utils.demographic_parity(y_pred_all, None, **kwargs)
                                            y_hat = (y_pred > 0.5).astype(np.float32)
                                            auc = roc_auc_score(Y_valid, y_hat)
                                            f1 = f1_score(Y_valid, y_hat)
                                            acc = accuracy_score(Y_valid, y_hat)

                                            # y_pred_all = vae_loss, mi_sz, y_prob.detach().cpu().numpy()
                                            # delta_DP = utils.demographic_parity(y_pred_all, None, **kwargs)
                                            # auc = roc_auc_score(y_valid_label.numpy(), y_prob.detach().cpu().numpy())
                                            result_log = f'/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/laftr_health.csv'
                                            if not os.path.isfile(result_log):
                                                with open(result_log, "w") as myfile:
                                                    myfile.write("param_search_id,auc,delta_dp,alpha,beta,gamma,lr,lr_adv,epoch,adv_rounds,dropout")
                                            df = pd.read_csv(result_log)
                                            row = {'param_search_id':param_search_id, 'auc': auc, 'delta_dp': delta_DP, 'alpha': alpha, 'beta':beta, 'gamma': gamma, 'lr': lr, 'lr_adv':lr_adv,'epoch': num_epochs, 'adv_rounds':adv_rounds, 'dropout':self.vfae.dropout.p}
                                            df.loc[len(df)] = row
                                            df.to_csv(result_log, index=False)

    def update_adversary(self, features):
        self.pytorch_model.eval()
        self.vfae.eval()
        X_torch = features.clone().detach().requires_grad_(True)
        self.pytorch_model(X_torch, self.discriminator)
        self.discriminator.train()
        self.optimizer_d.zero_grad()
        s_decoded = self.discriminator(self.pytorch_model.z)
        discriminator_loss = self.adv_loss(s_decoded, self.pytorch_model.s)
        discriminator_loss.backward()
        self.optimizer_d.step()
        self.discriminator.eval()
        self.pytorch_model.train()

class LAFTR(Module):
    """
    Implementation of the LMIFR.
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
                 hidden_dim,
                 dropout_rate,
                 activation=ReLU()
                 ):
        super().__init__()
        self.y_out_dim = y_dim
        self.encoder_z1 = VariationalMLP(x_dim + s_dim, z1_enc_dim, z_dim, activation)
        self.encoder_z2 = VariationalMLP(z_dim + y_dim, z2_enc_dim, z_dim, activation)

        self.decoder_z1 = VariationalMLP(z_dim + y_dim, z1_dec_dim, z_dim, activation)
        self.decoder_y = DecoderMLP(z_dim, x_dec_dim, self.y_out_dim, activation)
        self.decoder_x = DecoderMLP(z_dim + s_dim, x_dec_dim, x_dim, activation)
        self.activation = activation
        self.dropout = Dropout(dropout_rate)
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z1_enc_dim = z1_enc_dim
        self.z2_enc_dim = z2_enc_dim
        self.z1_dec_dim = z1_dec_dim
        self.x_dec_dim = x_dec_dim
        self.dropout_rate = dropout_rate
        self.loss = VFAELoss()


    def reset_params(self, device):
        self.encoder_z1 = VariationalMLP(self.x_dim + self.s_dim, self.z1_enc_dim, self.z_dim, self.activation)
        self.encoder_z2 = VariationalMLP(self.z_dim + self.y_dim, self.z2_enc_dim, self.z_dim, self.activation)

        self.decoder_z1 = VariationalMLP(self.z_dim + self.y_dim, self.z1_dec_dim, self.z_dim, self.activation)
        self.decoder_y = DecoderMLP(self.z_dim, self.x_dec_dim, self.y_out_dim, self.activation)
        self.decoder_x = DecoderMLP(self.z_dim + self.s_dim, self.x_dec_dim, self.x_dim, self.activation)
        self.dropout = Dropout(self.dropout_rate)
        self.to(device)
    # def set_pu(self, pu):
    #     self.pu = pu
    #     return

    def get_representations(self, inputs):
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)
        return z1_encoded


    def forward(self, inputs, discriminator):
        """
        :param inputs: dict containing inputs: {'x': x, 's': s, 'y': y} where x is the input feature vector, s the
        sensitive variable and y the target label.
        """
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        x_s = torch.cat([x, s], dim=1)
        x_s = self.dropout(x_s)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)


        z1_s = torch.cat([z1_encoded, s], dim=1)
        x_decoded = self.decoder_x(z1_s)

        y_decoded = self.decoder_y(z1_encoded)
        s_decoded = discriminator(z1_encoded)
        
        adv_loss = 1 - torch.abs(s_decoded - s).mean()
        # p_adversarial = Bernoulli(probs=s_decoded)
        # log_p_adv = p_adversarial.log_prob(s)
        # log_p_u = self.pu.log_prob(s)
        # self.mi_sz = log_p_adv - log_p_u
        # self.mi_sz_upper_bound = -0.5 * torch.sum(1 + z1_enc_logvar - z1_enc_mu ** 2 - z1_enc_logvar.exp(), dim = 1)
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
        self.vae_loss += self.gamma * adv_loss
        self.pred = y_decoded
        self.s = s
        self.z = z1_encoded
        self.y_prob = y_decoded.squeeze()
        return self.vae_loss, None, self.y_prob

class VariationalMLP(Module):
    """
    Single hidden layer MLP using the reparameterization trick for sampling a latent z.
    """

    def __init__(self, in_features, hidden_dim, z_dim, activation):
        super().__init__()
        self.activation = activation
        self.encoder = Sequential(
          Linear(in_features, hidden_dim),
          self.activation,
        )

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
        logvar = self.logvar_encoder(x)
        sigma = torch.sqrt(torch.exp(logvar))
        mu = self.mu_encoder(x)

        # reparameterization trick: we draw a random z
        z = sigma * torch.randn_like(mu) + mu
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
    Loss function for training the LMIFR.
    """

    def __init__(self, alpha=1.0, beta=0.0, mmd_dim=0, mmd_gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.bce = BCELoss()
        self.ce = CrossEntropyLoss()

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

        # # becomes kl between z2 and a standard normal when passing zeros

        reconstruction_loss /= len(y)
        # loss *= 0.1 # this is to keep the step size smaller for the primary objective
        loss = self.alpha * supervised_loss + self.beta * reconstruction_loss
        return loss
