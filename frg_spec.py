import autograd.numpy as np   # Thinly-wrapped version of Numpy
import os

from seldonian.spec import SupervisedSpec
from seldonian.dataset import SupervisedDataSet
from seldonian.dataset import DataSetLoader
from seldonian.models.pytorch_advdp import PytorchADVDP
from seldonian.models import objectives
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.parse_tree.parse_tree import (
    make_parse_trees_from_constraints)
from seldonian.utils.io_utils import load_json, save_pickle, load_pickle
import argparse

import torch

ADULTS = "adults"
HEALTH = "health"
INCOME = "income"


save_dir = "./SeldonianExperimentSpecs/advdp/spec"
os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--dataset', choices=[ADULTS, HEALTH, INCOME], help='config file path which stores the hyperparameters')
    parser.add_argument('--epsilon', help='epsilon')
    parser.add_argument('--delta', help='delta')

    args = parser.parse_args()

    torch.manual_seed(2023)
    dataname = args.dataset
    epsilon = float(args.epsilon)
    deltas = [float(args.delta)]
    
    if dataname == ADULTS:
        metadata_pth = "./adults_vfae/metadata_vfae.json"
        data_pth = "./adults_vfae/vfae_adults.csv"
        x_dim = 117
        z_dim = 50
        hidden_dim = 100
        bs = 0
        n_epochs = 1000
        lr = 1e-4
        s_dim = 1
        use_batches = False
        constraint_type = 'DP_ADV'
    elif dataname == HEALTH:
        metadata_pth = "./health/metadata_health_gender_age.json" #"./health/metadata_health_gender.json"
        data_pth = "./health/health_normalized_gender_age.csv" # "./health/health_normalized_gender.csv"
        x_dim = 121 # 130 # 123 if using age as senstive attribute
        z_dim = 50
        hidden_dim = 100
        bs = 0
        n_epochs = 1000
        lr = 1e-4
        s_dim = 1
        constraint_type = 'DP_ADV'
        use_batches = False
    elif dataname == INCOME: # Not used 
        metadata_pth = "./income/metadata_income.json" #"./health/metadata_health_gender.json"
        data_pth = "./income/income.csv" # "./health/health_normalized_gender.csv"
        x_dim = 23
        z_dim = 100
        hidden_dim = 100
        bs = 10000
        n_epochs = 1000
        lr = 1e-4
        s_dim = 5
        constraint_type = 'DP_ADV_multi_class'
        use_batches = True
    else:
        raise NotImplementedError
    
    save_base_dir = 'interface_outputs'
    # save_base_dir='.'
    # Load metadata
    regime='supervised_learning'
    sub_regime='classification'

    loader = DataSetLoader(
        regime=regime)

    dataset = loader.load_supervised_dataset(
        filename=data_pth,
        metadata_filename=metadata_pth,
        file_type='csv')


    # constraint_strs = [f'max(abs((PR_ADV | [M]) - (PR_ADV | [F])),abs((NR_ADV | [M]) - (NR_ADV | [F]))) <= {epsilon}']
    constraint_strs = [f'{constraint_type} <= {epsilon}']
    columns = ["M", "F"] # for Adult
    # columns = ["sexMALE", "sexFEMALE"]
    parse_trees = make_parse_trees_from_constraints(
        constraint_strs,deltas,regime=regime,
        sub_regime=sub_regime, columns=columns)
    device = torch.device(0)
    model = PytorchADVDP(device, **{"x_dim": x_dim,
        "s_dim": s_dim,
        "y_dim": 1,
        "z1_enc_dim": 100,
        "z2_enc_dim": 100,
        "z1_dec_dim": 100,
        "x_dec_dim": hidden_dim,
        "z_dim": z_dim,
        "dropout_rate": 0.3,
        "alpha_adv": lr,
        "mi_version": 1}
    )

    initial_solution_fn = model.get_model_params
    frac_data_in_safety = 0.4
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
            'lambda_init'   : np.array([0.5]),
            'alpha_theta'   : 1e-4,
            'alpha_lamb'    : 1e-4,
            'beta_velocity' : 0.9,
            'beta_rmsprop'  : 0.95,
            'use_batches'   : use_batches,
            'batch_size'    : bs,
            'n_epochs'      : n_epochs,
            'num_iters'     : n_epochs,
            'gradient_library': "autograd",
            'hyper_search'  : None,
            'verbose'       : True,
            'x_dim'         : x_dim,
            'z_dim'         : z_dim,
            'hidden_dim'    : hidden_dim,
            'lr'            : lr,
            'epsilon'       : epsilon,
            'downstream_lr' : 1e-4,
            'downstream_bs'     : 100,
            'downstream_epochs' : 5,
            'y_dim'             : 1,
            'n_adv_rounds'      : 5,
        },
        # batch_size_safety=2000
    )
    spec_save_name = os.path.join(
        save_dir, f"advdp_{dataname}_{epsilon}_{deltas[0]}_unsupervised.pkl"
    )
    save_pickle(spec_save_name, spec)
    print(f"Saved Spec object to: {spec_save_name}")
