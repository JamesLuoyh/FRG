# tensorflow_mnist.py
import autograd.numpy as np   # Thinly-wrapped version of Numpy
import pandas as pd
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.utils.io_utils import load_pickle,save_pickle
from seldonian.models import objectives
from seldonian.models.pytorch_advdp_cnn import PytorchAdvdpCNN
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
	make_parse_trees_from_constraints)
import torch

sub_regime = "classification"
N=23700
print("Loading features,labels,sensitive_attrs from file...")

savename_features = './face/features.pkl'
savename_gender_labels = './face/gender_labels.pkl'
savename_age_labels = './face/age_labels.pkl'
savename_sensitive_attrs = './face/sensitive_attrs.pkl'
savename_race_labels = './face/race_labels.pkl'

save_dir = "./SeldonianExperimentSpecs/advdp/spec/"
features = load_pickle(savename_features)
age_labels = load_pickle(savename_age_labels)
gender_labels = load_pickle(savename_gender_labels)
race_labels = load_pickle(savename_race_labels)
sensitive_attrs = load_pickle(savename_sensitive_attrs)

frac_data_in_safety = 0.5
sensitive_col_names = ['0','1', '2', '3', '4']

meta_information = {}
meta_information['feature_col_names'] = ['img']
meta_information['label_col_names'] = ['label']
meta_information['sensitive_col_names'] = sensitive_col_names
meta_information['sub_regime'] = sub_regime
meta_information['self_supervised'] = True
print("Making SupervisedDataSet...")

dataset = SupervisedDataSet(
    features=[features,sensitive_attrs, gender_labels],
    labels=[gender_labels, age_labels, (race_labels==0).astype(np.float32)],
    sensitive_attrs=sensitive_attrs,
    num_datapoints=N,
    meta_information=meta_information)
regime='supervised_learning'

deltas = [0.1]
print("sensitive_attrs", sensitive_attrs.shape)
epsilon = 0.48
    # constraint_strs = [f'max(abs((PR_ADV | [M]) - (PR_ADV | [F])),abs((NR_ADV | [M]) - (NR_ADV | [F]))) <= {epsilon}']
constraint_strs = [f'DP_ADV_multi_class <= {epsilon}']

print("Making parse trees for constraint(s):")
print(constraint_strs," with deltas: ", deltas)
parse_trees = make_parse_trees_from_constraints(
    constraint_strs,deltas,regime=regime,
    sub_regime=sub_regime,columns=sensitive_col_names)
z_dim = 100
hidden_dim = 100
device = torch.device(0)
model = PytorchAdvdpCNN(device, **{"x_dim": features.shape[1],
        "s_dim": 5,
        "y_dim": 1,
        "z1_enc_dim": z_dim,
        "z2_enc_dim": z_dim,
        "z1_dec_dim": z_dim,
        "x_dec_dim": z_dim,
        "z_dim": z_dim,
        "dropout_rate": 0.0,
        "alpha_adv": 1e-3,
        "mi_version": 1
        })
        
lambda_init = 5.0
initial_solution_fn = model.get_model_params
spec = SupervisedSpec(
    dataset=dataset,
    model=model,
    parse_trees=parse_trees,
    frac_data_in_safety=frac_data_in_safety,
    primary_objective=objectives.vae_loss,
    use_builtin_primary_gradient_fn=False,
    sub_regime=sub_regime,
    initial_solution_fn=initial_solution_fn,
    optimization_technique='gradient_descent',
    optimizer='adam',
    optimization_hyperparams={
        'epsilon'       : epsilon,
        'lambda_init'   : np.array([lambda_init]),
        'alpha_theta'   : 1e-4,
        'alpha_lamb'    : 1e-3,
        'beta_velocity' : 0.6,
        'beta_rmsprop'  : 0.95,
        'use_batches'   : True,
        'batch_size'    : 2370, #237
        'n_epochs'      : 40,
        'gradient_library': "autograd",
        'hyper_search'  : None,
        'verbose'       : True,
        'hidden_dim'    : hidden_dim,
        'downstream_lr' : 1e-4,
        'downstream_bs'     : 2370,
        'downstream_epochs' : 10,
        'y_dim'             : 1,
        'z_dim'             : z_dim,
        'n_adv_rounds'      : 3,
        's_dim'             : sensitive_attrs.shape[1]
    },
    
    batch_size_safety=2370
)
spec_save_name = os.path.join(
  save_dir, f"face_{epsilon}_{deltas[0]}.pkl"
)
save_pickle(spec_save_name, spec)
print(f"Saved Spec object to: {spec_save_name}")
