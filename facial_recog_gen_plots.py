## Run VFAE experiments as described in this example: 
### Possible constraint names: 
# [

# ]
### Possible epsilon values:
# 0.01 

import argparse
import numpy as np
import os
from experiments.generate_plots import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.utils import probabilistic_accuracy, probabilistic_auc, multiclass_demographic_parity, f1_score
from seldonian.utils.io_utils import load_pickle
from sklearn.model_selection import train_test_split
from seldonian.dataset import SupervisedDataSet
from seldonian.models.pytorch_cnn_vfae import PytorchFacialVAE

import torch

def vfae_example(
    spec_rootdir,
    results_base_dir,
    constraints = [],
    psi=1.18,
    n_trials=10,
    data_fracs=np.logspace(-3,0,5),
    baselines = [],
    performance_metric="auc",
    n_workers=1,
    version="0",
    validation=False,
    device_id=0,
    epsilon=0.08,
):  
    data_fracs = [1.0, 0.65, 0.40, 0.25, 0.15, 0.1]
    if performance_metric == "f1":
        perf_eval_fn = f1_score
    elif performance_metric == "auc":
        perf_eval_fn = probabilistic_auc
    elif performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
    elif performance_metric == "dp":
        perf_eval_fn = multiclass_demographic_parity


    z_dim = 100
    device = torch.device(device_id)
    model = PytorchFacialVAE(device, **{"x_dim": 1,
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
    
    print("version", version)
    specfile = os.path.join(
        spec_rootdir,
        f"unsupervised_cnn_vfae_mutual_information_{psi}.pkl"
    )
    print(f"unsupervised_cnn_vfae_mutual_information_{psi}.pkl")
    spec = load_pickle(specfile)
    spec.model = model
    torch.manual_seed(2023)
    np.random.seed(2023)
    #1.18
    alpha_l = [spec.optimization_hyperparams["alpha_theta"]]
    alpha_lambda_l = [spec.optimization_hyperparams["alpha_lamb"]]
    lambda_init_l = [spec.optimization_hyperparams["lambda_init"][0]]
    epochs_l = [spec.optimization_hyperparams["n_epochs"]]
    for lambda_init in lambda_init_l:
        for alpha in alpha_l:
            for alpha_lambda in alpha_lambda_l:
                for epochs in epochs_l:
                    spec.optimization_hyperparams["lambda_init"] = np.array([lambda_init])
                    spec.optimization_hyperparams["alpha_theta"] = alpha
                    spec.optimization_hyperparams["alpha_lamb"] = alpha_lambda
                    batch_epoch_dict = {
                        0.1:[237*2,int(epochs/0.1)],
                        0.15: [237*2,int(epochs/0.15)],
                        0.25:[237*2,int(epochs/0.25)],
                        0.40:[237*2,int(epochs/0.4)],
                        0.65:[237*2,int(epochs/0.65)],
                        1.0: [237*2,epochs],
                    }

                    if validation:
                        suffix = "validation"
                    else:
                        suffix = "test"
                    results_dir = os.path.join(results_base_dir,
                        f"cnn_vfae_mutual_information_{epsilon}_{psi}_{alpha}_{alpha_lambda}_{lambda_init}_{epochs}_{suffix}")

                    orig_features = spec.dataset.features
                    orig_features_X, orig_features_S, orig_features_Y = orig_features
                    orig_labels_gender = spec.dataset.labels[0]
                    orig_labels_age = spec.dataset.labels[1]
                    orig_sensitive_attrs = spec.dataset.sensitive_attrs
                    # First, shuffle features
                    (train_features_X,test_features_X,train_features_S, test_features_S, 
                    train_features_Y, test_features_Y, train_gender_labels,
                    test_gender_labels,train_age_labels, test_age_labels,train_sensitive_attrs,
                    test_sensitive_attrs
                        ) = train_test_split(
                            orig_features_X,
                            orig_features_S,
                            orig_features_Y,
                            orig_labels_gender,
                            orig_labels_age,
                            orig_sensitive_attrs,
                            shuffle=True,
                            test_size=0.2,
                            random_state=42)
                    new_dataset = SupervisedDataSet(
                    features=[train_features_X, train_features_S, train_features_Y], 
                    labels=[train_gender_labels, train_age_labels],
                    sensitive_attrs=train_sensitive_attrs, 
                    num_datapoints=len(train_features_X),
                    meta_information=spec.dataset.meta_information)
                    # Set spec dataset to this new dataset
                    spec.dataset = new_dataset
                    # Setup performance evaluation function and kwargs 
                    perf_eval_kwargs = {
                        'X':[test_features_X, test_features_S, test_features_Y],
                        'y':[test_gender_labels, test_age_labels],
                        'performance_metric':['auc', 'dp'],
                        'device': torch.device(device_id),
                        's_dim': orig_sensitive_attrs.shape[1]
                    }

                    plot_generator = SupervisedPlotGenerator(
                        spec=spec,
                        n_trials=n_trials,
                        data_fracs=data_fracs,
                        n_workers=n_workers,
                        batch_epoch_dict=batch_epoch_dict,
                        datagen_method='resample',
                        perf_eval_fn=[probabilistic_auc, multiclass_demographic_parity],
                        constraint_eval_fns=[],
                        results_dir=results_dir,
                        perf_eval_kwargs=perf_eval_kwargs,
                        n_downstreams=2,
                    )
                    if version != '0':
                        plot_generator.run_seldonian_experiment(verbose=verbose,model_name='FRG', validation=validation, dataset_name='Face')
                    else:
                        for baseline_model in baselines:
                            plot_generator.run_baseline_experiment(
                                model_name=baseline_model, verbose=verbose,validation=validation, dataset_name='Face'
                            )
                    for i in range(2):
                        plot_savename = os.path.join(
                            results_dir, f"cnn_vfae_mutual_information_{epsilon}_{psi}_{performance_metric}_downstream_{i}.pdf"
                        )
                        plot_generator.make_plots(
                            fontsize=12,
                            legend_fontsize=8,
                            performance_label=['AUC', 'Probability $\Delta_{\mathrm{DP}}$ > ' + str(epsilon)],
                            prob_performance_below=[None, epsilon],
                            performance_yscale="linear",
                            savename=plot_savename,
                            result_filename_suffix=f"_downstream_{i}"
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--include_baselines', help='include_baselines', action="store_true")
    parser.add_argument('--verbose', help='verbose', action="store_true")
    parser.add_argument('--version', help='which frg version to use. 0 uses baselines, 1 uses I1, 2 uses I2, 3 uses I1 and I2', required=True)
    parser.add_argument('--validation', help='validation', action="store_true")
    parser.add_argument('--device', help='device id', default=0)
    parser.add_argument('--epsilon', help='epsilon')
    parser.add_argument('--psi', help='psi')

    args = parser.parse_args()

    include_baselines = args.include_baselines
    verbose = args.verbose
    version = args.version
    validation = args.validation
    device_id = int(args.device)
    epsilon = float(args.epsilon)
    psi = float(args.psi)

    baselines = ["cnn_lmifr_all","cnn_icvae","cnn_vfae_baseline","cnn_vae","cnn_controllable_vfae"]

    performance_metric="auc_dp"

    results_base_dir = f"./SeldonianExperimentResults"
    vfae_example(
        spec_rootdir="./SeldonianExperimentSpecs/vfae/spec",
        results_base_dir=results_base_dir,

        performance_metric=performance_metric,
        baselines = baselines,
        version = version,
        validation=validation,
        device_id=device_id,
        epsilon=epsilon,
        psi=psi,
    )
    
