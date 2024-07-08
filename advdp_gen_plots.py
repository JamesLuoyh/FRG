## Run VFAE experiments as described in this example: 

### Possible psi values:
# 0.32, 0.0044 

import argparse
import numpy as np
import os
from experiments.generate_plots_by_epsilon import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.utils import probabilistic_accuracy, probabilistic_auc, demographic_parity, multiclass_demographic_parity
from seldonian.utils.io_utils import load_pickle
from sklearn.model_selection import train_test_split
from seldonian.dataset import SupervisedDataSet
import torch
from seldonian.models.pytorch_advdp import PytorchADVDP
from seldonian.utils.plot_utils import plot_gradient_descent
import matplotlib.pyplot as plt
import time
import pandas as pd

ADULTS = "adults"
GERMAN = "german"
HEALTH = "health"
torch.manual_seed(2023)
np.random.seed(2023)
def advdp_example(
    spec_rootdir,
    results_base_dir,
    constraints = [],
    n_trials=20,
    data_fracs=np.logspace(-3,0,5),
    baselines = [],
    performance_metric="auc",
    n_workers=1,
    dataset=ADULTS,
    validation=True,
    device_id=0,
    version=0,
    epsilon=0.08,
    delta=0.1,
):  
    data_fracs = [1] #[0.1, 0.15,0.25,0.40,0.65,]
    
    z_dim = 50
    dropout_rate = 0.3
    device = torch.device(device_id)
    # model = PytorchADVDP(device, **{"x_dim": 117,
    #     "s_dim": 1,
    #     "y_dim": 1,
    #     "z1_enc_dim": 100,
    #     "z2_enc_dim": 100,
    #     "z1_dec_dim": 100,
    #     "x_dec_dim": 100,
    #     "z_dim": z_dim,
    #     "dropout_rate": dropout_rate,
    #     "alpha_adv": 1e-4,
    #     "mi_version": 1}
    # )

    if performance_metric == "auc":
        perf_eval_fn = probabilistic_auc
    elif performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
    elif performance_metric == "dp":
        perf_eval_fn = demographic_parity
    else:
        raise NotImplementedError(
            "Performance metric must be 'auc' or 'accuracy' or 'dp' for this example")
    specfile = os.path.join(
        spec_rootdir,
        f"advdp_{dataset}_{epsilon}_{delta}_unsupervised_hidden.pkl"
    )
    spec = load_pickle(specfile)


    # theorectical psi=0.0044
    # alpha_l = [1e-4]
    # alpha_lambda_l = [1e-4]
    # lambda_init_l = [10.0]
    # epochs_l = [30]
    spec.dataset.meta_information['self_supervised'] = True

    # alpha_l =  [1e-3]#, 1e-4]
    # alpha_lambda_l = [1e-3]#, 1e-4] 1e-4,
    # alpha_adv_l = [1e-4]
    # lambda_init_l = [0.1]#0.5
    # epochs_l = [10]
    # adv_rounds = [2]
    
    # practical psi=0.32
    # alpha_l =  [1e-3,1e-4]#, 1e-4]
    # alpha_lambda_l = [1e-3, 1e-4]#, 1e-4] 1e-4,
    # alpha_adv_l = [1e-4]
    # lambda_init_l = [0.1, 0.5]#0.5
    # epochs_l = [10000]
    # adv_rounds = [2,5]
    # frac_data_in_safety = 0.2
    # alpha_l = [spec.optimization_hyperparams["alpha_theta"]]
    # alpha_lambda_l = [spec.optimization_hyperparams["alpha_lamb"]]
    # lambda_init_l = [spec.optimization_hyperparams["lambda_init"][0]]
    # epochs_l = [spec.optimization_hyperparams["n_epochs"]]

    # epsilon = 0.04
    # alpha_l =  [1e-3]#,1e-4]#, 1e-4]
    # alpha_lambda_l = [1e-3]#, 1e-4] 1e-4,
    # alpha_adv_l = [1e-4]
    # lambda_init_l = [0.5]#, 0.5]#0.5
    # epochs_l = [10000]
    # adv_rounds = [5]

    # epsilon = 0.08
    alpha_l =  [1e-3]#,1e-4]#, 1e-4]
    alpha_lambda_l = [1e-4]#, 1e-4] 1e-4,
    alpha_adv_l = [1e-4]
    lambda_init_l = [0.1]#, 0.5]#0.5
    epochs_l = [10000]
    adv_rounds = [5]

    # epsilon = 0.12
    # alpha_l =  [1e-3]#,1e-4]#, 1e-4]
    # alpha_lambda_l = [1e-4]#, 1e-4] 1e-4,
    # alpha_adv_l = [1e-4]
    # lambda_init_l = [0.5]#, 0.5]#0.5
    # epochs_l = [10000]
    # adv_rounds = [2]

    # # epsilon = 0.16
    # alpha_l =  [1e-3]#,1e-4]#, 1e-4]
    # alpha_lambda_l = [1e-4]#, 1e-4] 1e-4,
    # alpha_adv_l = [1e-4]
    # lambda_init_l = [0.5]#, 0.5]#0.5
    # epochs_l = [10000]
    # adv_rounds = [2]

    frac_data_in_safety = 0.2

    for lambda_init in lambda_init_l:
        for alpha in alpha_l:
            for alpha_lambda in alpha_lambda_l:
                for alpha_adv in alpha_adv_l:
                    for epochs in epochs_l:
                        for n_adv_rounds in adv_rounds:
                            spec.optimization_hyperparams["lambda_init"] = np.array([lambda_init])
                            spec.optimization_hyperparams["alpha_theta"] = alpha
                            spec.optimization_hyperparams["alpha_lamb"] = alpha_lambda
                            spec.optimization_hyperparams["n_adv_rounds"] = n_adv_rounds
                            spec.frac_data_in_safety = frac_data_in_safety
                            batch_epoch_dict = {
                                0.1:[500,int(epochs/0.1)],
                                0.15: [500,int(epochs/0.15)],
                                0.25:[500,int(epochs/0.25)],
                                0.40:[500,int(epochs/0.40)],
                                0.65:[500,int(epochs/0.65)],
                                1.0: [0,epochs],
                            }
                            spec.optimization_hyperparams["num_iters"] = epochs
                            if validation:
                                suffix = "validation"
                            else:
                                suffix = "test"
                            bs = batch_epoch_dict[1.0][0]
                            ts = int(time.time())

                            log_id = f"{dataset}_{epsilon}_{delta}_{alpha}_{alpha_lambda}_{alpha_adv}_{lambda_init}_{epochs}_{bs}_{n_adv_rounds}_{frac_data_in_safety}_{dropout_rate}_{suffix}_unsupervised"
                            if validation:
                                results_dir = os.path.join(results_base_dir,
                                    f"{ts}_{log_id}")#
                            else:
                                results_dir = os.path.join(results_base_dir,
                                    f"{dataset}_{delta}")#{log_id}"
                            plot_savename = os.path.join(
                                results_dir, f"{log_id}.png"
                            )
                            logfilename = os.path.join(
                                results_dir, f"candidate_selection_{log_id}.p"
                            )
                            orig_features = spec.dataset.features
                            orig_labels = spec.dataset.labels
                            orig_sensitive_attrs = spec.dataset.sensitive_attrs
                            # First, shuffle features
                            (train_features,test_features,train_labels,
                            test_labels,train_sensitive_attrs,
                            test_sensitive_attrs
                                ) = train_test_split(
                                    orig_features,
                                    orig_labels,
                                    orig_sensitive_attrs,
                                    shuffle=True,
                                    test_size=0.2,
                                    random_state=2023)
                            new_dataset = SupervisedDataSet(
                            features=train_features, 
                            labels=train_labels,
                            sensitive_attrs=train_sensitive_attrs, 
                            num_datapoints=len(train_features),
                            meta_information=spec.dataset.meta_information)
                            # Set spec dataset to this new dataset
                            spec.dataset = new_dataset
                            # Setup performance evaluation function and kwargs 
                            perf_eval_kwargs = {
                                'X':test_features,
                                'y':test_labels,
                                'performance_metric':['auc', 'dp'],
                                'device': torch.device(device),
                                's_dim': orig_sensitive_attrs.shape[1]
                            }

                            plot_generator = SupervisedPlotGenerator(
                                spec=spec,
                                n_trials=n_trials,
                                epsilons=[epsilon],
                                data_fracs=data_fracs,
                                n_workers=n_workers,
                                batch_epoch_dict=batch_epoch_dict,
                                datagen_method='resample',
                                perf_eval_fn=[probabilistic_auc, demographic_parity],
                                constraint_eval_fns=[],
                                results_dir=results_dir,
                                perf_eval_kwargs=perf_eval_kwargs,
                            )
                            if int(version) == 1:
                                plot_generator.run_seldonian_experiment(verbose=verbose, model_name='FRG',validation=validation, dataset_name='Adult', logfilename=logfilename)
                            else:
                                for baseline_model in baselines:
                                    plot_generator.run_baseline_experiment(
                                        model_name=baseline_model, verbose=verbose,validation=validation, dataset_name='Adult'
                                    )
                            plot_generator.make_plots(
                                fontsize=12,
                                legend_fontsize=8,
                                # performance_label=['AUC', 'Probability $\Delta_{\mathrm{DP}}$ > ' + str(epsilon)],
                                performance_label=['Average AUC', 'Average $\Delta_{\mathrm{DP}}$'],
                                # prob_performance_below=[None, epsilon],
                                prob_performance_below=[None, None],
                                performance_yscale="linear",
                                savename=plot_savename,
                            )
                            if verbose and int(version) == 1:
                                solution_dict = load_pickle(logfilename)
                                cs_plot_savename = os.path.join(
                                    results_dir, f"candidate_selection_{log_id}.png"
                                )
                                fig = plot_gradient_descent(solution_dict,
                                    primary_objective_name='vae loss',
                                    save=False)
                                plt.savefig(cs_plot_savename)
                            if int(version) == 1:
                                results = pd.read_csv(os.path.join(
                                    results_dir, "FRG_results", "FRG_results.csv"
                                ))
                                avg_auc = results.probabilistic_auc.mean()
                                avg_dp = results.demographic_parity.mean()
                                solution = results.passed_safety.mean()
                                success = 1 - results.failed.mean()
                                
                                result_log = os.path.join(results_base_dir, f"advdp_{dataset}_{epsilon}_{delta}_test.csv")
                                # check if file exists
                                if not os.path.isfile(result_log):
                                    with open(result_log, "w") as myfile:
                                        myfile.write("logid,avg_auc,avg_dp,solution,success,n_trials")
                                with open(result_log, "a") as myfile:
                                    # record timestamp to find the logs, record average auc/deltaDP, NSF rate, success rate,n_trials
                                    row = ','.join((str(ts), str(avg_auc), str(avg_dp), str(solution), str(success), str(n_trials))) 
                                    myfile.write(f'{row}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--include_baselines', help='include_baselines', action="store_true")
    parser.add_argument('--verbose', help='verbose', action="store_true")
    parser.add_argument('--validation', help='verbose', action="store_true")
    parser.add_argument('--version', help='version', default=0)
    parser.add_argument('--device', help='device id', default=0)
    parser.add_argument('--epsilon', help='epsilon')
    parser.add_argument('--delta', help='delta')

    args = parser.parse_args()

    include_baselines = args.include_baselines
    verbose = args.verbose
    validation = args.validation
    version = int(args.version)
    device_id = int(args.device)
    epsilon = float(args.epsilon)
    delta = float(args.delta)

    baselines = []#["LMIFR","ICVAE","VFAE", "VAE","controllable_vfae"]

    performance_metric="dp"

    results_base_dir = f"/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults"
    dataset = ADULTS
    advdp_example(
        spec_rootdir="./SeldonianExperimentSpecs/advdp/spec",
        results_base_dir=results_base_dir,
        performance_metric=performance_metric,
        dataset = dataset,
        baselines = baselines,
        validation = validation,
        device_id=device_id,
        version=version,
        epsilon=epsilon,
        delta=delta,
    )
    
