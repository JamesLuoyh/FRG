import torch
import torch.nn as nn
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli, Categorical
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import experiments.utils as utils

class PytorchCNNICVAE(SupervisedPytorchBaseModel):
    """
    Implementation of the Variational Fair AutoEncoder. Note that the loss has to be computed separately.
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
                 epsilon,
                 lambda_init=0.5,
                 activation=nn.ReLU(),
                 use_validation=True,
                 ):
        lam = 1
        self.vfae = CNNICVAE(x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 lam,
                 activation=nn.ReLU()).to(self.device)
        self.lr = 1e-3
        self.lr_d = 1e-3
        self.lr_l = 1e-2
        self.lambda_init = 1.0
        self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=self.lr)
        self.use_validation = use_validation
        self.discriminator = DecoderMLP(z_dim, z_dim, s_dim, activation).to(self.device)
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        return self.vfae

    # set a prior distribution for the sensitive attribute for VAE case
    def set_pu(self, pu):
        if len(pu) == 1:
            pu_dist = Bernoulli(probs=torch.tensor(pu).to(self.device))
        else:
            pu_dist = Categorical(probs=torch.tensor(pu).to(self.device))
        self.vfae.set_pu(pu_dist)
        return

    def train(self, X_train, Y_train, batch_size, num_epochs, data_frac):
        print("Training model...")
        loss_list = []
        accuracy_list = []
        iter_list = []
        X_train, S_train, Y_semi_train = X_train        
        
        if self.use_validation:
            X_train_size = int(len(X_train) * 0.8)
            X_valid = X_train[X_train_size:]
            Y_valid = Y_semi_train[X_train_size:]
            S_valid = S_train[X_train_size:]
            x_train_tensor = torch.from_numpy(X_train[:X_train_size])
            y_semi_train_tensor = torch.from_numpy(Y_semi_train[:X_train_size])
            s_train_tensor = torch.from_numpy(S_train[:X_train_size])
            x_valid_tensor = torch.from_numpy(X_valid)
            y_valid_label = torch.from_numpy(Y_valid)
            s_valid_tensor = torch.from_numpy(S_valid)
            
        else:
            x_train_tensor = torch.from_numpy(X_train)
            y_semi_train_tensor = torch.from_numpy(Y_semi_train)
            s_train_tensor = torch.from_numpy(S_train)
            x_valid_tensor = x_train_tensor
            y_valid_label = y_semi_train_tensor
        train = torch.utils.data.TensorDataset(x_train_tensor, s_train_tensor, y_semi_train_tensor, y_semi_train_tensor)
        trainloader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True
        )
        
        print(
            f"Running gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
        )
        itot = 0
        lr_l = [1e-4]#[1e-3,1e-4,1e-5]
        num_epochs_l = [int(90/data_frac)]#[30, 60, 90]
        lam_l = [100]#[1,10,100]
        for lr in lr_l:
            for num_epochs in num_epochs_l:
                 for lam in lam_l:
                    self.vfae.lam = lam
                    self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=lr)
                    for epoch in range(num_epochs):
                        for i, (features, sensitive, semi_labels, labels) in enumerate(trainloader):
                            # Load images
                            self.discriminator.eval()
                            features = features.float().to(self.device)
                            sensitive = sensitive.float().to(self.device)
                            semi_labels = semi_labels.float().to(self.device)

                            # Clear gradients w.r.t. parameters
                            self.optimizer.zero_grad()
                            self.vfae.train()
                            # Forward pass to get output/logits
                            vae_loss, mi_sz, y_prob = self.pytorch_model(features, sensitive, semi_labels, self.discriminator)

                            # Getting gradients w.r.t. parameters
                            vae_loss.backward()

                            # Updating parameters
                            self.optimizer.step()
                            if i % 10 == 0:
                                it = f"{i+1}/{len(trainloader)}"
                                print(f"Epoch, it, itot, loss, mi: {epoch},{it},{itot},{vae_loss}, {mi_sz.mean()}")
                            itot += 1
                    
                    if self.use_validation:
                        self.discriminator.eval()
                        self.vfae.eval()
                        self.pytorch_model.eval()
                        kwargs = {
                            'downstream_lr'     : 1e-4,
                            'downstream_bs'     : 237,
                            'downstream_epochs' : 10,
                            'y_dim'             : 1,
                            's_dim'             : self.s_dim,
                            'z_dim'             : self.z_dim,
                            'hidden_dim'        : self.hidden_dim,
                            'device'            : self.device,
                            "X"                 : [X_valid, S_valid, Y_valid],
                        }

                        y_pred = utils.unsupervised_downstream_predictions(self, self.get_model_params(), X_train, Y_semi_train, X_valid, **kwargs)
                        x_valid_tensor = x_valid_tensor.float().to(self.device)
                        s_valid_tensor = s_valid_tensor.float().to(self.device)
                        y_valid_label = y_valid_label.float().to(self.device)
                        vae_loss, mi_sz, y_prob = self.pytorch_model(x_valid_tensor, s_valid_tensor, y_valid_label, self.discriminator)
                        y_pred_all = vae_loss, mi_sz, y_pred
                        delta_DP = utils.multiclass_demographic_parity(y_pred_all, None, **kwargs)
                        auc = roc_auc_score(Y_valid, y_pred)
                        df = pd.read_csv('./SeldonianExperimentResults/cnn_icvae_test.csv')
                        row = {'auc': auc, 'delta_dp': delta_DP, 'mi': mi_sz.mean().item(),'lam': lam, 'lr': lr, 'epochs':num_epochs, 'data_frac':data_frac}
                        print(row)
                        df = df.append(row, ignore_index=True)
                        df.to_csv('./SeldonianExperimentResults/cnn_icvae_test.csv', index=False)
                    

    def get_representations(self, X):
        return self.vfae.get_representations(X)

class CNNICVAE(nn.Module):
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
                lam,
                activation=nn.ReLU()):
        super(CNNICVAE, self).__init__()
        self.z_dim = z_dim
        self.latent_dim = z_dim
        self.x_dim = x_dim
        self.s_dim = s_dim
        modules = []
        in_channels = 1
        # if hidden_dims is None:
        self.hidden_dims = [16, 32, 64, 128]

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1]*9, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1]*9, self.latent_dim)

        # Build Decoder
        modules = []

        # latent_dim + 1 for adding the sensitve attribute
        self.decoder_input = nn.Linear(self.latent_dim + self.s_dim, self.hidden_dims[-1] * 9)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1],kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(self.hidden_dims[-1], out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

        self.decoder_y = DecoderMLPBinary(z_dim, z_dim, 1, activation)
        self.bce = nn.BCELoss()
        self.lam = lam

    def set_pu(self, pu):
        self.pu = pu
        return

    def set_lagrangian(self, lagrangian, lagrangian_elbo):
        self.lagrangian = lagrangian
        self.lagrangian_elbo = lagrangian_elbo
        return


    def get_representations(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 3, 3)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, sensitive_attributes, labels, discriminator):
        x = input
        s = sensitive_attributes
        y = labels
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z_s = torch.cat([z, s], dim=1)

        s = torch.argmax(s, dim=1)
  
        x_decoded = self.decode(z_s)
        y_decoded = self.decoder_y(z)
        # print(self.mi_sz)
        outputs = {
            # predictive outputs
            'x_decoded': x_decoded,
            'y_decoded': y_decoded,
            'z1_encoded': z,

            # outputs for regularization loss terms
            'z1_enc_logvar': log_var,
            'z1_enc_mu': mu,
        }
        # will return the constraint C2 term. log(qu) - log(pu) instead of y_decoded
        self.vae_loss = self.loss_function(outputs, {'x': x, 's': s, 'y': y})
        self.mi_sz = self.kl_conditional_and_marg(mu, log_var, self.z_dim).mean()
        reconstruct_loss = 0.1 * F.mse_loss(x_decoded, x, reduction='sum') / len(x)

        self.mi_sz += reconstruct_loss

        self.vae_loss += self.lam * self.mi_sz
        self.mi_sz = self.mi_sz.flatten()
        self.pred = y_decoded
        self.s = s
        self.z = z
        self.y_prob = y_decoded.squeeze()
        return self.vae_loss, self.mi_sz, self.y_prob

    def loss_function(self, prediction, actual):
        """
        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log \frac{1}{sigma} + \frac{sigma^2 + mu^2}{2} - \frac{1}{2}
        :return:
        """
        recons = prediction['x_decoded']
        input = actual['x']
        y = actual['y']
        mu = prediction['z1_enc_mu']
        log_var = prediction['z1_enc_logvar']

        recons_loss = F.mse_loss(recons, input, reduction='sum')


        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = 0.1 * (recons_loss + kld_loss) / len(y)  # this is to adjust the step size for the primary objective
        return loss

    @staticmethod
    def all_pairs_gaussian_kl(mu, sigma, add_third_term=False):
        sigma_sq = sigma.square() + 1e-8
        sigma_sq_inv = torch.reciprocal(sigma_sq)

        #dot product of all sigma_inv vectors with sigma is the same as a matrix mult of diag
        first_term = torch.matmul(sigma_sq, sigma_sq_inv.T)
        r = torch.matmul(mu * mu,sigma_sq_inv.T)
        r2 = mu * mu * sigma_sq_inv 
        r2 = torch.sum(r2,1)
        #squared distance
        #(mu[i] - mu[j])\sigma_inv(mu[i] - mu[j]) = r[i] - 2*mu[i]*mu[j] + r[j]
        #uses broadcasting
        second_term = 2*torch.matmul(mu, (mu*sigma_sq_inv).T)
        second_term = r - second_term + (r2.unsqueeze(1)).T
        # log det A = tr log A
        # log \frac{ det \Sigma_1 }{ det \Sigma_0 } =
        #   \tr\log \Sigma_1 - \tr\log \Sigma_0 
        # for each sample, we have B comparisons to B other samples...
        #   so this cancels out

        if(add_third_term):
            r = torch.sum(torch.log(sigma_sq),1)
            r = torch.reshape(r,[-1,1])
            third_term = r - r.T
        else:
            third_term = 0

        #- tf.reduce_sum(tf.log(1e-8 + tf.square(sigma)))\
        # the dim_z ** 3 term comes fro
        #   -the k in the original expression
        #   -this happening k times in for each sample
        #   -this happening for k samples
        #return 0.5 * ( first_term + second_term + third_term - dim_z )
        return 0.5 * ( first_term + second_term + third_term )

    #
    # kl_conditional_and_marg
    #   \sum_{x'} KL[ q(z|x) \| q(z|x') ] + (B-1) H[q(z|x)]
    #

    #def kl_conditional_and_marg(args):
    def kl_conditional_and_marg(self, z_mean, z_log_sigma_sq, dim_z):
        z_sigma = ( 0.5 * z_log_sigma_sq ).exp()
        all_pairs_GKL = self.all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5*dim_z
        return torch.mean(all_pairs_GKL, dim=1)


    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class DecoderMLP(nn.Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = nn.Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = nn.Linear(hidden_dim, latent_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.softmax(self.lin_out(x))

class DecoderMLPBinary(nn.Module):
    """
     Single hidden layer MLP used for decoding.
    """

    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = nn.Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_encoder_2 = nn.Linear(in_features, hidden_dim//2)
        self.lin_out = nn.Linear(hidden_dim//2, latent_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        x = self.activation(self.lin_encoder_2(x))
        return self.sigmoid(self.lin_out(x))