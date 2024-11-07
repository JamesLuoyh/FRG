""" Utilities used in the rest of the library """

import os
import pickle
import numpy as np
import math

from seldonian.RL.RL_runner import create_agent, run_trial_given_agent_and_env
from seldonian.utils.stats_utils import weighted_sum_gamma
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.alg_utils import train_downstream, downstream_predictions
import torch.nn as nn
import torch
def generate_resampled_datasets(dataset, n_trials, save_dir, frac_valid=0):
    """Utility function for supervised learning to generate the 
    resampled datasets to use in each trial. Resamples (with replacement)
    features, labels and sensitive attributes to create n_trials versions of these
    of the same shape as the inputs

    :param dataset: The original dataset from which to resample
    :type dataset: pandas DataFrame

    :param n_trials: The number of trials, i.e. the number of
            resampled datasets to make
    :type n_trials: int

    :param save_dir: The parent directory in which to save the
            resampled datasets
    :type save_dir: str

    :param file_format: The format of the saved datasets, options are
            "csv" and "pkl"
    :type file_format: str

    """
    save_subdir = os.path.join(save_dir, "resampled_dataframes")
    os.makedirs(save_subdir, exist_ok=True)
    num_datapoints = dataset.num_datapoints
    np.random.seed(2024)
    for trial_i in range(n_trials):
        savename = os.path.join(save_subdir, f"trial_{trial_i}.pkl")
        # if not os.path.exists(savename):
        all_data = np.array(range(num_datapoints), dtype=np.int64)
        ix_valid = None
        if frac_valid > 0:
            n_valid = round(frac_valid * num_datapoints)
            ix_valid = np.random.choice(
                all_data, n_valid, replace=False
            )
            all_data = all_data[~np.isin(all_data, ix_valid)]
        ix_resamp = np.random.choice(
            all_data, num_datapoints, replace=True
        )
        if ix_valid is not None:
            ix_resamp = np.append(ix_resamp, ix_valid)
        # print(ix_resamp)
        # features can be list of arrays or a single array
        if type(dataset.features) == list:
            resamp_features = [x[ix_resamp] for x in dataset.features]
        else:
            resamp_features = dataset.features[ix_resamp]

        # labels and sensitive attributes must be arrays
        if type(dataset.labels) is list:
            resamp_labels = []
            for label in dataset.labels:
                resamp_labels.append(label[ix_resamp])
        else:
            resamp_labels = dataset.labels[ix_resamp]
        if isinstance(dataset.sensitive_attrs, np.ndarray):
            resamp_sensitive_attrs = dataset.sensitive_attrs[ix_resamp]
        else:
            resamp_sensitive_attrs = []

        resampled_dataset = SupervisedDataSet(
            features=resamp_features,
            labels=resamp_labels,
            sensitive_attrs=resamp_sensitive_attrs,
            num_datapoints=num_datapoints,
            meta_information=dataset.meta_information,
        )

        with open(savename, "wb") as outfile:
            pickle.dump(resampled_dataset, outfile)
        print(f"Saved {savename}")

def generate_episodes_and_calc_J(**kwargs):
    """Calculate the expected discounted return
    by generating episodes

    :return: episodes, J, where episodes is the list
            of generated ground truth episodes and J is
            the expected discounted return
    :rtype: (List(Episode),float)
    """
    # Get trained model weights from running the Seldonian algo
    model = kwargs["model"]
    new_params = model.policy.get_params()

    # create env and agent
    hyperparameter_and_setting_dict = kwargs["hyperparameter_and_setting_dict"]
    agent = create_agent(hyperparameter_and_setting_dict)
    env = hyperparameter_and_setting_dict["env"]

    # set agent's weights to the trained model weights
    agent.set_new_params(new_params)

    # generate episodes
    num_episodes = kwargs["n_episodes_for_eval"]
    episodes = run_trial_given_agent_and_env(
        agent=agent, env=env, num_episodes=num_episodes
    )

    # Calculate J, the discounted sum of rewards
    returns = np.array([weighted_sum_gamma(ep.rewards, env.gamma) for ep in episodes])
    J = np.mean(returns)
    return episodes, J

def unsupervised_downstream_predictions(model, solution, X_train, Y_train, X_test, **kwargs):
    # first train a MLP model
    # then generated predictions
    if type(X_train) == list:
        # For unsupervised learning, we use the sensitive attribute in features list
        # We remove it for downstream prediction
        X_train = X_train[0]
    # else:
    #     X_train = X_train[:, :-1]
    if type(X_test) == list:
        X_test = X_test[0]
    # else:
    #     X_test = X_test[:, :-1]
    
    batch_size = kwargs["downstream_bs"]
    num_epochs = kwargs["downstream_epochs"]
    lr = kwargs["downstream_lr"]
    device = kwargs["device"]
    z_dim = kwargs["z_dim"]
    hidden_dim = kwargs["hidden_dim"]
    y_dim = kwargs["y_dim"]
    if model == 'random':
        y_pred = np.random.binomial(1, 0.5, len(X_test))
        return y_pred
    if not model.params_updated:
        model.update_model_params(solution,**kwargs)
        model.params_updated = True
    downstream_model = train_downstream(model, X_train, Y_train, batch_size,
                                        num_epochs, lr, z_dim, hidden_dim, y_dim, device)
    # downstream_model = model.vfae.decoder_y
    y_pred = downstream_predictions(model, downstream_model, X_test, batch_size, y_dim, device)
    return y_pred

def vae_predictions(model, solution, X_test, **kwargs):
    batch_size = kwargs["eval_batch_size"]
    if type(X_test) == list:
        N_eval = len(X_test[0])
    else:
        N_eval = len(X_test)
    y_pred = np.zeros(N_eval)
    mi = np.zeros(N_eval)
    loss = 0
    num_batches = math.ceil(N_eval / batch_size)
    batch_start = 0
    for i in range(num_batches):
        batch_end = batch_start + batch_size

        if type(X_test) == list:
            X_test_batch = [x[batch_start:batch_end] for x in X_test]
        else:
            X_test_batch = X_test[batch_start:batch_end]
        loss_batch, mi_batch, y_batch = model.predict(solution, X_test_batch)
        y_pred[batch_start:batch_end] = y_batch
        mi[batch_start:batch_end] = mi_batch.flatten()
        loss += loss_batch
        batch_start = batch_end
    return loss, mi, y_pred

def batch_predictions(model, solution, X_test, **kwargs):
    batch_size = kwargs["eval_batch_size"]
    if type(X_test) == list:
        N_eval = len(X_test[0])
    else:
        N_eval = len(X_test)
    if "N_output_classes" in kwargs:
        N_output_classes = kwargs["N_output_classes"]
        y_pred = np.zeros((N_eval, N_output_classes))
    else:
        y_pred = np.zeros(N_eval)
    num_batches = math.ceil(N_eval / batch_size)
    batch_start = 0
    for i in range(num_batches):
        batch_end = batch_start + batch_size

        if type(X_test) == list:
            X_test_batch = [x[batch_start:batch_end] for x in X_test]
        else:
            X_test_batch = X_test[batch_start:batch_end]
        y_pred[batch_start:batch_end] = model.predict(solution, X_test_batch)
        batch_start = batch_end
    return y_pred

def make_batch_epoch_dict_fixedniter(niter,data_fracs,N_max,batch_size):
    """
    Convenience function for figuring out the number of epochs necessary
    to ensure that at each data fraction, the total 
    number of iterations (and batch size) will be fixed. 

    :param niter: The total number of iterations you want run at every data_frac
    :type niter: int
    :param data_fracs: 1-D array of data fractions
    :type data_fracs: np.ndarray 
    :param N_max: The maximum number of data points in the optimization process
    :type N_max: int
    :param batch_size: The fixed batch size 
    :type batch_size: int
    :return batch_epoch_dict: A dictionary where keys are data fractions 
        and values are [batch_size,num_epochs]
    """
    data_sizes=data_fracs*N_max # number of points used in candidate selection in each data frac
    n_batches=data_sizes/batch_size # number of batches in each data frac
    n_batches=np.array([math.ceil(x) for x in n_batches])
    n_epochs_arr=niter/n_batches # number of epochs needed to get to niter iterations in each data frac
    n_epochs_arr = np.array([math.ceil(x) for x in n_epochs_arr])
    batch_epoch_dict = {
        data_fracs[ii]:[batch_size,n_epochs_arr[ii]] for ii in range(len(data_fracs))}
    return batch_epoch_dict

def make_batch_epoch_dict_min_sample_repeat(
    niter_min,
    data_fracs,
    N_max,
    batch_size,
    num_repeats):
    """
    Convenience function for figuring out the number of epochs necessary
    to ensure that the number of iterations for each data frac is:
    max(niter_min,# of iterations s.t. each sample is seen num_repeat times)

    :param niter_min: The minimum total number of iterations you want run at every data_frac
    :type niter_min: int
    :param data_fracs: 1-D array of data fractions
    :type data_fracs: np.ndarray 
    :param N_max: The maximum number of data points in the optimization process
    :type N_max: int
    :param batch_size: The fixed batch size
    :type batch_size: int
    :param num_repeats: The minimum number of times each sample must be seen in the optimization process
    :type num_repeats: int
    :return batch_epoch_dict: A dictionary where keys are data fractions 
        and values are [batch_size,num_epochs]
    """
    batch_epoch_dict = {}
    n_epochs_arr = np.zeros_like(data_fracs)
    for data_frac in data_fracs:
        niter2 = num_repeats*N_max*data_frac/batch_size
        if niter2 > niter_min:
            num_epochs = num_repeats
        else:
            n_batches = max(1,N_max*data_frac/batch_size)
            num_epochs = math.ceil(niter_min/n_batches)
        batch_epoch_dict[data_frac] = [batch_size,num_epochs]
    
    return batch_epoch_dict

##### performance evaluation functions #####

def binary_logistic_loss(y_pred,y,**kwargs):    
    return log_loss(y,y_pred)

def multiclass_logistic_loss(y_pred, y, **kwargs):
    """Calculate average logistic loss
    over all data points for multi-class classification

    :return: logistic loss
    :rtype: float
    """
    # In the multi-class setting, y_pred is an i x k matrix
    # where i is the number of samples and k is the number of classes
    # Each entry is the probability of predicting the kth class
    # for the ith sample. We need to get the probability of predicting
    # the true class for each sample and then take the sum of the
    # logs of that.
    n = len(y)
    probs_trueclasses = y_pred[np.arange(n), y.astype("int")]
    return -1 / n * sum(np.log(probs_trueclasses))

def auc(y_pred, y, **kwargs):
    """For binary classification only.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    # v = np.where(y != 1.0, 1.0 - y_pred, y_pred)
    loss, mi, Y_pred_probs = y_pred
    Y_pred_probs = Y_pred_probs.squeeze()
    # print(Y_pred_probs)
    Y_ = (Y_pred_probs > 0.5).astype(np.float32)
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, Y_)

def acc(y_pred, y, **kwargs):
    loss, mi, Y_pred_probs = y_pred
    Y_pred_probs = Y_pred_probs.squeeze()
    # print(Y_pred_probs)
    Y_ = (Y_pred_probs > 0.5).astype(np.float32)
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, Y_)

def f1_score(y_pred, y, **kwargs):
    """For binary classification only.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    # v = np.where(y != 1.0, 1.0 - y_pred, y_pred)
    loss, mi, Y_pred_probs = y_pred
    Y_pred_probs = Y_pred_probs.squeeze()
    # print(Y_pred_probs)
    Y_ = (Y_pred_probs > 0.5).astype(np.float32)
    from sklearn.metrics import f1_score
    return f1_score(y, Y_)
    # return sum(v) / 

def multiclass_f1_score(y_pred, y, **kwargs):
    """For binary classification only.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    # v = np.where(y != 1.0, 1.0 - y_pred, y_pred)
    loss, mi, y_pred = y_pred
    # Y_pred_probs = Y_pred_probs.squeeze()
    # print(Y_pred_probs)
    from sklearn.metrics import f1_score
    return f1_score(y, np.argmax(y_pred, axis=1), average="micro")

def multiclass_demographic_parity(y_pred, y, **kwargs):
    # y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    loss, mi, y_pred = y_pred
    y_ = (y_pred > 0.5).astype(np.float32)
    X = kwargs["X"]
    s_dim = kwargs["s_dim"]
    if type(X) == list:
        X, S, Y = X
    else:
        S = X[:, -1 - s_dim: -1]
    n_classes = S.shape[1]
    S = np.argmax(S, axis=1)
    g, uc = np.zeros([n_classes]), np.zeros([n_classes]) + 1e-15 # avoid division by 0
    for i in range(S.shape[0]):
        uc[S[i]] += 1.0
        g[S[i]] += y_[i]
    print(g)
    print(uc)
    g = g / uc

    # return np.max(np.abs(np.max(g, axis=1) - np.min(g, axis=1)))
    return np.abs(np.max(g) - np.min(g))


def demographic_parity(y_pred, y, **kwargs):
    # y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    s_dim = kwargs["s_dim"]
    if s_dim > 1:
        multiclass_demographic_parity(y_pred, y, **kwargs)
    loss, mi, y_prob = y_pred
    y_ = (y_prob > 0.5).astype(np.float32)
    X = kwargs["X"]
    if type(X) == list:
        X, S = X
    else:
        S = X[:, -2]
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(S.shape[0]):
        if S[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc

    return np.abs(g[0] - g[1])

def bounded_dp(y_pred, y, epsilon, **kwargs):
    dp = demographic_parity(y_pred, y, kwargs)
    return dp <= epsilon

def equal_opp(y_pred, y, **kwargs):
    X = kwargs["X"]
    if type(X) == list:
        X, S = X
    else:
        S = X[:, -2]
    loss, mi, y_prob = y_pred
    y_ = (y_prob > 0.5).astype(np.float32)

    sel0 = y_[np.where((S==0) & (y==1))[0]]
    sel1 = y_[np.where((S==1) & (y==1))[0]]
    if len(sel0) == 0:
        sel0_mean = 0
    else:
        sel0_mean = sel0.mean()
    if len(sel1) == 0:
        sel1_mean = 0
    else:
        sel1_mean = sel1.mean()
    return np.abs(sel0_mean - sel1_mean)



def equalized_odds(y_pred, y, **kwargs):
    # The mean of positive and negative equal opportunities
    loss, mi, y_prob = y_pred
    neg_y_prob = 1 - y_prob
    neg_y_pred = loss, mi, neg_y_prob
    
    eopp1 = equal_opp(y_pred, y, **kwargs)
    eopp0 = equal_opp(neg_y_pred, 1 - y, **kwargs)
    return (eopp0 + eopp1) / 2

def probabilistic_accuracy(y_pred, y, **kwargs):
    """For binary classification only.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    # v = np.where(y != 1.0, 1.0 - y_pred, y_pred)
    loss, mi, Y_pred_probs = y_pred
    Y_pred_probs = Y_pred_probs.squeeze()
    # print(Y_pred_probs)
    # v = np.where(y!=1,1.0-Y_pred_probs,Y_pred_probs)
    # print(np.sum(v)/len(v))
    # return np.sum(v)/len(v)
    # return sum(v) / 
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, Y_pred_probs)

def multiclass_accuracy(y_pred,y,**kwargs):
    """For multi-class classification.
    1 - error rate. Use when output of 
    model y_pred is a probability

    :param y_pred: Array of predicted probabilities of each label
    :param y: Array of true labels, 1-dimensional

    """
    n = len(y)
    return np.sum(y_pred[np.arange(n),y.astype("int")])/n

def deterministic_accuracy(y_pred, y, **kwargs):
    """The fraction of correct samples. Best to use
    only when the output of the model, y_pred
    is 0 or 1. 

    :param y_pred: Array of predicted labels
    :param y: Array of true labels

    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(y,y_pred > 0.5)


def MSE(y_pred, y, **kwargs):
    """Calculate sample mean squared error

    :param y_pred: Array of predicted labels
    :param y: Array of true labels
    """
    n = len(y)
    res = sum(pow(y_pred - y, 2)) / n
    return res
