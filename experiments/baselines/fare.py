import torch
from seldonian.models.pytorch_model import SupervisedPytorchBaseModel
from seldonian.utils.alg_utils import DecoderMLP
from math import pi, sqrt
from torch.distributions import Bernoulli
from torch.nn import Module, Linear, ReLU, Dropout, BCELoss, CrossEntropyLoss, Sigmoid, Sequential, Parameter
from torch.nn.functional import softplus
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.nn import init
import pandas as pd
import numpy as np
import experiments.utils as utils
import os
import time
import torch.nn.functional as F
from ctypes import c_uint
import sys 
import argparse
import numpy as np
import numpy.matlib
import os 
from box import Box
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import time
import math
from .alphabeta_adversary import AlphaBetaAdversary


from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree 
def demographic_parity_difference(y, c, y_hat):
    """This will only return max, mean implementation is not there"""
    c = c.reshape(-1)
    #assert y_hat.shape[1] == 2
    #assert y.shape[0] == c.shape[0]
   
    y_ = np.argmax(y_hat, axis=1)
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(c.shape[0]):
        if c[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1]), None
    # dpdist = 0
    # for curr_y in np.arange(y_hat.shape[1]):
    #     selection = (y_pred == curr_y)
    #     dpdist = max(dpdist, fairlearn_metrics.demographic_parity_difference(y, selection, sensitive_features=c))
    #     # true y is not used

    # return dpdist, None

def equal_opportunity_difference(y, c, y_hat, favorable_y=1): 
    c = c.reshape(-1)
    assert y_hat.shape[1] == 2
    assert y.shape[0] == c.shape[0]

    y_pred = y_hat[:, 1] > 0.5


    sel0 = y_pred[(c.ravel()==0) & (y.ravel()==favorable_y)].mean()
    sel1 = y_pred[(c.ravel()==1) & (y.ravel()==favorable_y)].mean()

    return np.abs(sel0 - sel1), None


def equalized_odds_difference(y, c, y_hat):
    eopp1, _ = equal_opportunity_difference(y, c, y_hat, favorable_y=1)
    eopp0, _ = equal_opportunity_difference(y, c, y_hat, favorable_y=0)
    return (eopp0 + eopp1) / 2, None


def learn(data, cat_pos, max_leaf_nodes, min_samples_leaf, alpha, gini_metric):

    x_train, s_train, y_train = data['train']
    x_val, _, _ = data['val'] # unused now
    x_test, s_test, y_test = data['test']
    s_train, s_test = s_train.reshape(-1, 1), s_test.reshape(-1, 1)
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print('getting 0th label!')
        y_train, y_test = y_train[:, 0],  y_test[:, 0]
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    # create and fit the tree
    criterion = f'fair_gini_{gini_metric}' # e.g., fair_gini_dp
    T = DecisionTreeClassifier(criterion=criterion, max_leaf_nodes=max_leaf_nodes, random_state=43, min_samples_leaf=min_samples_leaf)
    T = T.fit(x_train, y_train, s_train, cat_pos=cat_pos, alpha=alpha)
    # # plot tree
    # if not os.path.exists('src/tree/out/'):
    #     os.makedirs('src/tree/out/')
    # plot_tree(T, node_ids=True)
    # plt.savefig(f'src/tree/out/tree_{alpha}.pdf')
    # plt.clf()

    print('tree built and saved')
    return T

def eval(T, data, cat_pos):
    x_train, s_train, y_train = data['train']
    x_val, _, _ = data['val'] # unused now
    x_test, s_test, y_test = data['test']
    s_train, s_test = s_train.reshape(-1, 1), s_test.reshape(-1, 1)
    if  len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print('getting 0th label!')
        y_train, y_test = y_train[:, 0],  y_test[:, 0]
    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    print('############ Eval #################')
    
    base_train = y_train.sum() / y_train.shape[0]
    base_train = max(1-base_train, base_train) * 100
    base_test = y_test.sum() / y_test.shape[0]
    base_test = max(1-base_test, base_test) * 100
    print(f'Base Rates: train={base_train:.3f} test={base_test:.3f}')

    # score acc and DP
    acc_train = T.score(x_train, y_train)*100
    acc_test = T.score(x_test, y_test)*100
    proba_train = T.predict_proba(x_train)
    proba_test = T.predict_proba(x_test)
    y_hat = np.argmax(proba_test, axis=1)
    auc = roc_auc_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat)
    acc = accuracy_score(y_test, y_hat)
    dp_train, _ = demographic_parity_difference(y_train, s_train, proba_train)
    dp_test, _ = demographic_parity_difference(y_test, s_test, proba_test)
    eopp_test, _ = equal_opportunity_difference(y_test, s_test, proba_test)
    eodd_test, _ = equalized_odds_difference(y_test, s_test, proba_test)
    test_performance = {'auc': auc, 'acc':acc, 'f1':f1, 'dp':dp_test, 'eopp':eopp_test, 'eodd':eodd_test}
    print(f"\033[0;32m (DP) TRAIN = ({acc_train:.3f}, {dp_train:.3f}) [&&] TEST = ({acc_test:.3f}, {dp_test:.3f})", flush=True)
    print("\033[0m", flush=True) 

    #############################################

    # get leaf/cell IDs reached from train set
    nb_cells = (T.tree_.children_left == -1).sum()
    cells_train = T.apply(x_train)
    cell_ids = sorted(list(set(cells_train)))
    
    # ensure all cells are present
    assert len(cell_ids) == nb_cells
    #assert len(sorted(list(set(T.apply(x_val))))) == nb_cells
    cells_test = T.apply(x_test)
    assert len(sorted(list(set(cells_test)))) == nb_cells

    #####################################################3

    # get medians for each cell
    medians = {}
    for cid in cell_ids:
        # get all train set xs that go to this cell
        xs = x_train[np.where(cells_train == cid)]
        
        # get median
        median = np.zeros(xs.shape[1])
        for i in range(xs.shape[1]): 
            if i in cat_pos:
                # categorical takes mode
                median[i] = mode(xs[:, i].astype(int))[0] # check
            else:
                # continuous takes median
                median[i] = np.median(xs[:, i]) # needs numpy 1.9.0
                #median[i] = np.mean(xs[:, i])
        medians[cid] = median 

    # encode a set of xs with the tree
    def encode(xs, T):
        cells = T.apply(xs)
        zs = []
        for cell in cells:
            zs.append(medians[cell])
        return np.vstack(zs)
    
    # Return embeddings
    z_train = encode(x_train, T)
    z_val = encode(x_val, T)
    z_test = encode(x_test, T) 

    return nb_cells, z_train, z_val, z_test, test_performance

def downstream_prediction(y_train, y_test, batch_size, z_train, z_test, hidden_dim, device, num_epochs, lr, y_dim):
    print("Training downstream model...")
    loss_list = []
    accuracy_list = []
    iter_list = []
    # x_train, s_train, y_train = data['train']
    # x_train_tensor = torch.from_numpy(x_train)
    y_train_label = torch.from_numpy(y_train)
    z_train_rep = torch.from_numpy(z_train).float()
    z_test_rep = torch.from_numpy(z_test).float().to(device)

    print(z_train_rep.shape)
    print(y_train_label.shape)
    train = torch.utils.data.TensorDataset(y_train_label, z_train_rep)
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    activation = nn.ReLU()
    criterion = nn.BCELoss()
    z_dim = z_train.shape[1]
    downstream_model = DecoderMLP(z_dim, hidden_dim, 1, activation).to(device) # model.vfae.decoder_y
    # downstream_model = model.vfae.decoder_y
    print(
        f"Running downstream gradient descent with batch_size: {batch_size}, num_epochs={num_epochs}"
    )
    itot = 0
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=lr)
    downstream_model.train()
    for epoch in range(num_epochs):
        for i, (labels, reps) in enumerate(trainloader):
            # Load images
            # features = features.float().to(device)
            labels = labels.to(device)
            reps = reps.to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # get representations
            # representations = model.get_representations(features)
            # get prediction
            y_pred = downstream_model.forward(reps)
            print(y_pred.shape)
            print(labels.shape)
            # get loss
            loss = criterion(y_pred, labels.float())
            # loss backward

            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                it = f"{i+1}/{len(trainloader)}"
                print(f"Epoch, it, itot, loss: {epoch},{it},{itot},{loss}")
            itot += 1
    downstream_model.eval()
    # x_test, s_test, y_test = data['test']
    N_eval = len(y_test)
    # x_test = torch.from_numpy(x_test).float().to(device)
    y_pred = np.zeros([N_eval, y_dim])
    loss = 0
    num_batches = math.ceil(N_eval / batch_size)
    batch_start = 0
    for i in range(num_batches):
        batch_end = batch_start + batch_size

        # if type(x_test) == list:
        #     X_test_batch = [x[batch_start:batch_end] for x in x_test]
        # else:
        #     X_test_batch = x_test[batch_start:batch_end]
        # get representations
        # get predictions
        y_batch = downstream_model.forward(z_test_rep[batch_start:batch_end])
        y_pred[batch_start:batch_end] = y_batch.cpu().detach().numpy()

        batch_start = batch_end
    return y_pred

def prep_data(data, meta):
    # Revert 1-hot encoding -> tree gets cats as cats 
    x_train, x_test = [], []
    cat_pos = []
    # for new_idx, idx in enumerate(meta['ft_pos'].values()):
    #     if type(idx) == tuple:
    #         # cat 
    #         slc = data['train'][0][:, idx[0]:idx[1]]
    #         assert slc.max(axis=1).min().item() == 1
    #         x_train.append(slc.argmax(axis=1)+1)

    #         slc = data['test'][0][:, idx[0]:idx[1]]
    #         assert slc.max(axis=1).min().item() == 1
    #         x_test.append(slc.argmax(axis=1)+1)
    #         cat_pos.append(new_idx)
    #     else:
    #         # cont 
    for idx in range(data['train'][0].shape[1]):
        x_train.append(data['train'][0][:, idx])
        x_test.append(data['test'][0][:, idx])
    data['train'] = [np.vstack(x_train).T, data['train'][1], data['train'][2]]
    data['test'] = [np.vstack(x_test).T, data['test'][1], data['test'][2]]
    cat_pos = np.asarray(cat_pos, dtype=np.int32)

    #####################################################################
    # (!) We need internally a validation set 
    n_trainval = data['train'][0].shape[0] 
    n_val = int(0.25 * n_trainval) # 60 : 20 : 20

    perm = np.random.permutation(n_trainval)

    val_idxs = perm[:n_val]
    train_idxs = perm[n_val:]

    data['val'] = []
    for i in range(3):
        data['val'].append(data['train'][i][val_idxs])
    for i in range(3):
        data['train'][i] = data['train'][i][train_idxs]

    for k in ['train', 'val', 'test']:
        s = ''
        for i in range(3):
            s += ' ' + f'{data[k][i].shape}'
        print(s)

    embeddings = {
        'c_train': data['train'][1].reshape(-1,1),
        'c_val': data['val'][1].reshape(-1,1),
        'c_test': data['test'][1].reshape(-1,1),

        'y_train': data['train'][2].reshape(-1,1),
        'y_val': data['val'][2].reshape(-1,1),
        'y_test': data['test'][2].reshape(-1,1)
    }
    
    return data, cat_pos, embeddings


class PytorchFARE(SupervisedPytorchBaseModel):
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
            activation=ReLU()
        ):
        self.vfae = FARE(x_dim,
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
        # self.optimizer = torch.optim.Adam(self.vfae.parameters(), lr=1)
        self.s_dim = s_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.downstream_bs = downstream_bs
        # self.adv_loss = BCELoss()
        self.use_validation = use_validation
        return self.vfae


    def get_representations(self, X):
        return self.vfae.get_representations(X)


    def train(self, X_train, Y_train, batch_size, num_epochs,data_frac, n_valid, X_test):
        x, s, y = X_train[:,:self.x_dim], X_train[:,self.x_dim:self.x_dim+self.s_dim], X_train[:,-self.y_dim:]
        x_test, s_test, y_test = X_test[:,:self.x_dim], X_test[:,self.x_dim:self.x_dim+self.s_dim], X_test[:,-self.y_dim:]
        data = {}
        data['train'] = x, s, y#[:-n_valid],s[:-n_valid],y[:-n_valid]
        data['test'] = x_test, s_test, y_test#x[-n_valid:],s[-n_valid:],y[-n_valid:]
        meta = {}
        meta['ft_pos'] = {
            'age': 0, 'fnlwgt': 1, 'education-num': 2, 'capital-gain': 3, 'capital-loss': 4, 'hrs-wk': 5,
            'workclass': (6, 12),
            'education': (13, 28),
            'marital-status': (29, 35),
            'occupation': (36, 49),
            'relationship': (50, 55),
            'race': (56, 60),
            'native-country': (61, 101)
        }
        data, cat_pos, embeddings = prep_data(data, meta)

        data_embed, cat_pos_embed, embeddings = data, cat_pos, embeddings # use the same dataset
        max_k_l = [2, 4, 8, 16, 32, 64, 128, 200]
        min_ni_l = [50, 100, 200, 500, 1000]
        alpha_l = [0.01, 0.1, 0.5, 1]
        ### Run the algorithm, it will internally use validation 
        for max_k in max_k_l:
            for min_ni in min_ni_l:
                for alpha in alpha_l:
                    param_search_id = int(time.time())
                    print("Training model...")
                    T = learn(data, cat_pos, max_k, min_ni, alpha, 'dp')
                    k, z_train, z_val, z_test, test_performance = eval(T, data_embed, cat_pos_embed)
                    z_train = np.concatenate((z_train,z_val), axis=0)
                    print(z_train.shape)
                    # predict the sensitive attribute
                    proba_test = downstream_prediction(s, s_test, len(z_train), z_train, z_test, z_train.shape[1], self.device, 5, 1e-4, self.y_dim)
                    y_hat = (proba_test > 0.5).astype(np.float32)
                    auc = roc_auc_score(s_test, y_hat)
                    f1 = f1_score(s_test, y_hat)
                    acc = accuracy_score(s_test, y_hat)
                    kwargs = {}
                    kwargs['X'] = X_test
                    proba_test = None, None, proba_test
                    dp_test = utils.demographic_parity(proba_test, s_test, **kwargs)
                    eopp_test = utils.equal_opp(proba_test, s_test, **kwargs)
                    eodd_test = utils.equalized_odds(proba_test, s_test, **kwargs)
                    diff_downstream_preference = {'auc': auc, 'acc':acc, 'f1':f1, 'dp':dp_test, 'eopp':eopp_test, 'eodd':eodd_test}
                    
                    result_log = f'/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/fare_alternative_downstream.csv'
                    if not os.path.isfile(result_log):
                        with open(result_log, "w") as myfile:
                            myfile.write("param_search_id,auc,acc,f1,dp,eopp,eodd,max_k,min_ni,alpha")
                    df = pd.read_csv(result_log)
                    diff_downstream_preference['param_search_id'] = param_search_id
                    diff_downstream_preference['max_k'] = max_k
                    diff_downstream_preference['min_ni'] = min_ni
                    diff_downstream_preference['alpha'] = alpha
                    # print(row)
                    df.loc[len(df)] = diff_downstream_preference
                    df.to_csv(result_log, index=False)
                    
                    embeddings['z_train'] = z_train 
                    embeddings['z_val'] = z_val 
                    embeddings['z_test'] = z_test 


                    # Proof on the embeddings object
                    err_budget = 0.1#05 # find best UB s.t. we are 95% confident
                    # if meta['c_type'] == 'binary':
                        # Usual flow for binary sensitive attributes
                    adv = AlphaBetaAdversary(k, err_budget, eps_glob=0.005, eps_ab=0.005, method='cp', verbose=True)

                    embeds = {}
                    for k, v in embeddings.items():
                        embeds[k] = v 
                        if 'y_' in k and  len(v.shape) > 1 and v.shape[1] > 1:
                            embeds[k] = v[:,0]
                    dp_ub = adv.ub_demographic_parity(embeds)
                    print(dp_ub)
                    print('TREE DONE.')
                    test_performance['dp_ub'] = dp_ub[0]
                    result_log = f'/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/fare.csv'
                    if not os.path.isfile(result_log):
                        with open(result_log, "w") as myfile:
                            myfile.write("param_search_id,auc,acc,f1,dp,eopp,eodd,dp_ub,max_k,min_ni,alpha")
                    df = pd.read_csv(result_log)
                    test_performance['param_search_id'] = param_search_id
                    test_performance['max_k'] = max_k
                    test_performance['min_ni'] = min_ni
                    test_performance['alpha'] = alpha
                    # print(row)
                    df.loc[len(df)] = test_performance
                    df.to_csv(result_log, index=False)


class FARE(Module):
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
        

    # def to(self, device):
    #     self.device = device
        # return super().to(device=device)
    
    # def get_representations(self, inputs):
    #     x, s, y = inputs[:,:self.x_dim], inputs[:,self.x_dim:self.x_dim+self.s_dim], inputs[:,-self.y_dim:]
    #     # encode
    #     x_s = torch.cat([x, s], dim=1)
    #     z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)
    #     return z1_encoded
