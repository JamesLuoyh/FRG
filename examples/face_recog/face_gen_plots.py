## Run VFAE experiments as described in this example: 
import argparse
import numpy as np
import os
from experiments.generate_plots_by_epsilon import SupervisedPlotGenerator
from experiments.base_example import BaseExample
from experiments.utils import probabilistic_accuracy, auc, f1_score, acc, demographic_parity, equal_opp,equalized_odds, bounded_dp, multiclass_demographic_parity
from seldonian.utils.io_utils import load_pickle
from sklearn.model_selection import train_test_split
from seldonian.dataset import SupervisedDataSet
import torch
from seldonian.models.pytorch_advdp import PytorchADVDP
from seldonian.utils.plot_utils import plot_gradient_descent
import matplotlib.pyplot as plt
import time
import pandas as pd

FACE = "Face"
torch.manual_seed(2023)
np.random.seed(2023)

def face_advdp_example(
    spec_rootdir,
    results_base_dir,
    constraints = [],
    n_trials=1,
    data_fracs=np.logspace(-3,0,5),
    baselines = [],
    performance_metric="auc",
    n_workers=1,
    dataset=FACE,
    validation=True,
    device_id=0,
    version=0,
    epsilon=0.08,
    delta=0.1,
):  
    data_fracs = [1]
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
        perf_eval_fn = auc
    elif performance_metric == "accuracy":
        perf_eval_fn = probabilistic_accuracy
    elif performance_metric == "dp":
        perf_eval_fn = demographic_parity
    else:
        raise NotImplementedError(
            "Performance metric must be 'auc' or 'accuracy' or 'dp' for this example")
    specfile = os.path.join(
        spec_rootdir,
        f"face_{epsilon}_{delta}.pkl"
    )
    
    
    # Face

    # alpha_l =  [1e-3]#, 1e-4]
    # alpha_lambda_l = [1e-4,1e-3]#, 1e-3]#, 1e-4] 1e-4,
    # alpha_adv_l = [1e-4]
    # lambda_init_l = [1.0,0.5]#,0.1]#, , 1.0]#, 0.5]#0.5
    # epochs_l = [500,1000, 2000]#00,1000]#, 1000]#, 1000, 10000]
    # adv_rounds = [10]#,2,5]#10]#, 2, 5]
    
    # 0.48 supervised
    alpha_l =  [1e-3]#, 1e-4]
    alpha_lambda_l = [1e-3]#, 1e-3]#, 1e-4] 1e-4,
    alpha_adv_l = [1e-4]
    lambda_init_l = [1.0]#,0.1]#, , 1.0]#, 0.5]#0.5
    epochs_l = [1]#00]#00]#,1000, 2000]#00,1000]#, 1000]#, 1000, 10000]
    adv_rounds = [10]

    frac_data_in_safety = 0.25
    n_downstreams= 1 if validation else 3
    for lambda_init in lambda_init_l:
        for alpha in alpha_l:
            for alpha_lambda in alpha_lambda_l:
                for alpha_adv in alpha_adv_l:
                    for epochs in epochs_l:
                        for n_adv_rounds in adv_rounds:
                            spec = load_pickle(specfile)
                            print(len(spec.dataset.meta_information['feature_col_names']))
                            spec.dataset.meta_information['self_supervised'] = True
                            spec.model.set_dropout(dropout_rate)
                            # spec.model.set_alpha_sup(alpha_sup)
                            spec.optimization_hyperparams["dropout"] = dropout_rate
                            spec.optimization_hyperparams["lambda_init"] = np.array([lambda_init])
                            spec.optimization_hyperparams["alpha_theta"] = alpha
                            spec.optimization_hyperparams["alpha_lamb"] = alpha_lambda
                            spec.optimization_hyperparams["n_adv_rounds"] = n_adv_rounds
                            spec.frac_data_in_safety = frac_data_in_safety
                            batch_epoch_dict = {
                                1.0: [2370,epochs],
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
                                dirname = f"{dataset}_{delta}_full_metrics_unsupervised"
                
                                results_dir = os.path.join(results_base_dir, dirname)#{log_id}"
                            
                            logfilename = os.path.join(
                                results_dir, f"candidate_selection_{log_id}.p"
                            )
                            orig_features = spec.dataset.features
                            orig_features_X, orig_features_S, orig_features_Y = orig_features
                            orig_labels_gender = np.expand_dims(spec.dataset.labels[0], axis=1)
                            orig_labels_age = np.expand_dims(spec.dataset.labels[1], axis=1)
                            orig_labels_race = np.expand_dims(spec.dataset.labels[2], axis=1)
                            orig_labels = np.concatenate((orig_labels_gender,orig_labels_age,orig_labels_race), axis=1)
                            orig_sensitive_attrs = spec.dataset.sensitive_attrs
                            # First, shuffle features
                            (train_features_X,test_features_X,train_features_S, test_features_S, 
                            train_features_Y, test_features_Y, train_labels,
                            test_labels,train_sensitive_attrs,test_sensitive_attrs
                                ) = train_test_split(
                                    orig_features_X,
                                    orig_features_S,
                                    orig_features_Y,
                                    orig_labels,
                                    orig_sensitive_attrs,
                                    shuffle=True,
                                    test_size=0.2,
                                    random_state=42)
                            new_dataset = SupervisedDataSet(
                            features=[train_features_X, train_features_S, train_features_Y], 
                            labels=train_labels,
                            sensitive_attrs=train_sensitive_attrs, 
                            num_datapoints=len(train_features_X),
                            meta_information=spec.dataset.meta_information)
                            # Set spec dataset to this new dataset
                            spec.dataset = new_dataset
                            # Setup performance evaluation function and kwargs 
                            perf_eval_kwargs = {
                                'X':[test_features_X, test_features_S, test_features_Y],
                                'y':test_labels,
                                'performance_metric':['auc', 'acc', 'f1', 'dp', 'eo', 'eodd'],
                                'device': torch.device(device_id),
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
                                perf_eval_fn=[auc, auc, auc, multiclass_demographic_parity,multiclass_demographic_parity,multiclass_demographic_parity],
                                constraint_eval_fns=[],#bounded_dp
                                results_dir=results_dir,
                                perf_eval_kwargs=perf_eval_kwargs,
                                n_downstreams=n_downstreams,
                            )
                            if int(version) == 1:
                                plot_generator.run_seldonian_experiment(verbose=verbose, model_name='FRG',validation=validation, dataset_name=dataset, logfilename=logfilename)
                            else:
                                for baseline_model in baselines:
                                    plot_generator.run_baseline_experiment(
                                        model_name=baseline_model, verbose=verbose,validation=validation, dataset_name=dataset
                                    )
                            if n_downstreams > 1:
                                for i in range(n_downstreams):
                                    plot_savename = os.path.join(
                                        results_dir, f"{log_id}_downstream_{i}.pdf"
                                    )
                                    plot_generator.make_plots(
                                        fontsize=12,
                                        legend_fontsize=8,
                                        performance_label=['Average AUC', 'Average ACC', 'Average F1', 'Average $\Delta_{\mathrm{DP}}$', 'Average EOPP', 'Average EODD'],
                                        prob_performance_below=[None, None, None, None, None, None],
                                        performance_yscale="linear",
                                        savename=plot_savename,
                                        result_filename_suffix=f"_downstream_{i}"
                                    )
                            else:
                                plot_savename = os.path.join(
                                        results_dir, f"{log_id}.png"
                                    )
                                plot_generator.make_plots(
                                    fontsize=12,
                                    legend_fontsize=8,
                                    performance_label=['Average AUC', 'Average ACC', 'Average F1', 'Average $\Delta_{\mathrm{DP}}$', 'Average EO', 'Average EODD'],
                                    prob_performance_below=[None, None, None, None, None, None],
                                    performance_yscale="linear",
                                    savename=plot_savename,
                                    result_filename_suffix=""
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
                            if validation and int(version) == 1:
                                results = pd.read_csv(os.path.join(
                                    results_dir, "FRG_results", "FRG_results.csv"
                                ))
                                avg_auc = results.auc.mean()
                                avg_dp = results.multiclass_demographic_parity.mean()
                                solution = results.passed_safety.mean()
                                success = 1 - results.failed.mean()
                                
                                result_log = os.path.join(results_base_dir, f"advdp_{dataset}_{epsilon}_{delta}_valid.csv")
                                # check if file exists
                                if not os.path.isfile(result_log):
                                    with open(result_log, "w") as myfile:
                                        myfile.write("logid,avg_auc,avg_dp,solution,success,n_trials\n")
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

    baselines = ['FARE']#["LAFTR"]#["CFAIR"]#["FCRL"]#["ICVAE"]#["LMIFR"]#["ICVAE"]#,,"VFAE", "VAE","controllable_vfae"]

    performance_metric="dp"

    results_base_dir = f"./SeldonianExperimentResults"
    dataset = FACE
    face_advdp_example(
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
    
