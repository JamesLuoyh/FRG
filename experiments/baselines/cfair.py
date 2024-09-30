import torch
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from math import pi, sqrt
from torch.distributions import Bernoulli

import torch.nn as nn
from torch.nn.functional import softplus
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.nn import init
import pandas as pd
import numpy as np
import experiments.utils as utils
import os
import time
import torch.nn.functional as F
from torch.autograd import Function

class PytorchCFair(SupervisedPytorchBaseModel):
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
            activation=nn.ReLU(),
        ):
        self.vfae = CFair(x_dim,
            s_dim,
            y_dim,
            z1_enc_dim,
            z2_enc_dim,
            z1_dec_dim,
            x_dec_dim,
            z_dim,
            dropout_rate,
            lr,
            activation=activation).to(self.device)
        self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=1)
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.downstream_bs = downstream_bs
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

        # Adult 0.16
        # mus = [1e-1,1e-2]
        # lrs = [1]

        # Tuning

        # mus = [1e-4, 1e-3, 1e-2, 1e-1]
        # lrs = [1e-4, 1e-3, 1e-2, 1e-1]

        # Health ALL
        mus = [1e-2]
        lrs = [1e-1]

        ##############

        # Adult 0.16
        # mus = [1e-2]
        # lrs = [1e-2]

        # # Others
        # mus = [1e-1]
        # lrs = [1]


        num_epochs = 500
        print(
            f"Running gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
        )
        if self.use_validation:
            repeats = 2
        else:
            repeats = 1
        for mu in mus:
            for lr in lrs:
                param_search_id = int(time.time())
                for repeat in range(repeats): # repeat the parameter search for 3 times
                    trainloader = torch.utils.data.DataLoader(
                        train, batch_size=batch_size, shuffle=True
                    )
                    self.vfae.reset_params(self.device)
                    itot = 0
                    self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=lr)
                    self.vfae.mu = mu
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
                                print(f"Epoch, it, itot, loss: {epoch},{it},{itot},{vae_loss}")
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
                        result_log = f'/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/cfair.csv'
                        if not os.path.isfile(result_log):
                            with open(result_log, "w") as myfile:
                                myfile.write("param_search_id,auc,acc,f1,delta_dp,mu,epoch,lr")
                        df = pd.read_csv(result_log)
                        row = {'param_search_id':param_search_id, 'auc': auc, 'acc': acc, 'f1': f1, 'delta_dp': delta_DP, 'mu':mu, 'epoch': num_epochs, 'lr': lr}
                        # print(row)
                        df.loc[len(df)] = row
                        df.to_csv(result_log, index=False)


class CFair(nn.Module):
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
                 activation=nn.ReLU()):
        super().__init__()
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        self.num_classes = 2 # label classes
        hidden_layers = [50]
        self.num_hidden_layers = len(hidden_layers)
        self.num_neurons = [self.x_dim] + hidden_layers
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], self.num_classes)
        # Parameter of the conditional adversary classification layer.
        adversary_layers = [50]
        self.num_adversaries = [self.num_neurons[-1]] + adversary_layers
        self.num_adversaries_layers = len(adversary_layers)
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])
    #     self.to(device)

    # def to(self, device):
    #     self.device = device
        # return super().to(device=device)
    
    def reset_params(self,device):
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)]).to(device)
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], self.num_classes).to(device)
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)]).to(device)
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)]).to(device)

    def forward(self, inputs):
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        y_mean = torch.mean(y)
        # encode
        reweight_target_tensor = torch.tensor([1.0 / (1.0 - y_mean), 1.0 / y_mean])
        train_idx = s == 0
        
        train_base_0, train_base_1 = torch.mean(y[train_idx]), torch.mean(y[~train_idx])
        reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0])
        reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1])
        reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

        x_s = torch.cat([x, s], dim=1)
        h_relu = x
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = y == j
            c_h_relu = h_relu[idx.squeeze()]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        # return logprobs, c_losses
        loss = F.nll_loss(logprobs, y.squeeze().long(), weight=reweight_target_tensor.to(y.device))
        adv_loss = torch.stack([F.nll_loss(c_losses[j], s[y == j].long(), weight=reweight_attr_tensors[j].to(y.device))
                                            for j in range(self.num_classes)])
        loss += self.mu * torch.mean(adv_loss)
        return loss, None, torch.exp(logprobs)

    def get_representations(self, inputs):
        x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
        # encode
        h_relu = x
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        return h_relu


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)