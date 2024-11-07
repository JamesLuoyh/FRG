""" Module for running Seldonian Experiments """

import os
import pickle
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from seldonian.utils.io_utils import load_pickle
from seldonian.dataset import SupervisedDataSet, RLDataSet
from seldonian.seldonian_algorithm import SeldonianAlgorithm
from seldonian.spec import RLSpec
from seldonian.models.models import (
    LinearRegressionModel,
    BinaryLogisticRegressionModel,
    DummyClassifierModel,
    RandomClassifierModel,
)

from .utils import batch_predictions, vae_predictions, unsupervised_downstream_predictions, demographic_parity, multiclass_demographic_parity

try:
    from fairlearn.reductions import ExponentiatedGradient
    from fairlearn.metrics import (
        MetricFrame,
        selection_rate,
        false_positive_rate,
        true_positive_rate,
        false_negative_rate,
    )
    from fairlearn.reductions import (
        DemographicParity,
        FalsePositiveRateParity,
        EqualizedOdds,
    )
except ImportError:
    print(
        "\nWARNING: The module 'fairlearn' was not imported. "
        "If you want to use the fairlearn baselines, then do:\n"
        "pip install fairlearn==0.7.0\n"
    )

import warnings
from seldonian.warnings.custom_warnings import *

warnings.filterwarnings("ignore", category=FutureWarning)


class Experiment:
    def __init__(self, model_name, results_dir):
        """Base class for running experiments

        :param model_name: The string name of the baseline model,
                e.g 'logistic_regression'
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str

        """
        self.model_name = model_name
        self.results_dir = results_dir

    def aggregate_results(self, **kwargs):
        """Group together the data in each
        trial file into a single CSV file.
        """
        savedir_results = os.path.join(self.results_dir, f"{self.model_name}_results")
        os.makedirs(savedir_results, exist_ok=True)
        savename_results = os.path.join(
            savedir_results, f"{self.model_name}_results.csv"
        )

        trial_dir = os.path.join(
            self.results_dir, f"{self.model_name}_results", "trial_data"
        )
        df_list = []

        # for epsilon in [0.04,0.08,0.12,0.16]:#:
        # for trial_i in range(kwargs["n_trials"]):
        if kwargs["n_downstreams"] > 1:
            for i in range(kwargs["n_downstreams"]):
                if len(df_list) <= i:
                    df_list.append([])
                for root, dirs, files in os.walk(trial_dir):
                    for file in files:
                        if file.endswith(f'downstream_{i}.csv'):
                            filename = os.path.join(
                                trial_dir, file#f"epsilon_{epsilon:.4f}_trial_{trial_i}_downstream_{i}.csv"
                            )
                            df = pd.read_csv(filename)
                            df['failed'] = df['demographic_parity'] > df['epsilon']

                            df_list[i].append(df)
        else:
            for root, dirs, files in os.walk(trial_dir):
                for file in files:
                    if file.endswith('.csv'):
                        filename = os.path.join(
                            trial_dir, file#f"epsilon_{epsilon:.4f}_trial_{trial_i}.csv"
                        )
                        df = pd.read_csv(filename)
                        df_list.append(df)

        if kwargs["n_downstreams"] > 1:
            for i in range(kwargs["n_downstreams"]):
                result_df = pd.concat(df_list[i])
                savename_results = os.path.join(
                    savedir_results, f"{self.model_name}_results_downstream_{i}.csv"
                )
                result_df.to_csv(savename_results, index=False)
        else:
            result_df = pd.concat(df_list)
            result_df.to_csv(savename_results, index=False)
        print(f"Saved {savename_results}")
        return

    def write_trial_result(self, data, colnames, trial_dir, downstream_i=None, verbose=False):
        """Write out the results from a single trial
        to a file.

        :param data: The information to save
        :type data: List

        :param colnames: Names of the items in the list.
                These will comprise the header of the saved file
        :type colnames: List(str)

        :param trial_dir: The directory in which to save the file
        :type trial_dir: str

        :param verbose: if True, prints out saved filename
        :type verbose: bool
        """
        result_df = pd.DataFrame([data])
        result_df.columns = colnames
        epsilon, data_frac, trial_i = data[0:3]
        if downstream_i is not None:
            savename = os.path.join(
                trial_dir, f"epsilon_{epsilon:.4f}_trial_{trial_i}_downstream_{downstream_i}.csv"
            )
        else:
            savename = os.path.join(
                trial_dir, f"epsilon_{epsilon:.4f}_trial_{trial_i}.csv"
            )

        result_df.to_csv(savename, index=False)
        if verbose:
            print(f"Saved {savename}")
        return


class BaselineExperiment(Experiment):
    def __init__(self, model_name, results_dir):
        """Class for running baseline experiments
        against which to compare Seldonian Experiments

        :param model_name: The string name of the baseline model,
                e.g 'logistic_regression'
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str
        """
        super().__init__(model_name, results_dir)

    def run_experiment(self, **kwargs):
        """Run the baseline experiment"""
        partial_kwargs = {
            key: kwargs[key] for key in kwargs if key not in ["data_fracs", "n_trials"]
        }

        helper = partial(self.run_baseline_trial, **partial_kwargs)

        data_fracs = kwargs["data_fracs"]
        n_trials = kwargs["n_trials"]
        n_workers = kwargs["n_workers"]

        data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
        trials_vector = np.array(
            [x for y in range(len(data_fracs)) for x in range(n_trials)]
        )

        if n_workers == 1:
            for ii in range(len(data_fracs_vector)):
                data_frac = data_fracs_vector[ii]
                trial_i = trials_vector[ii]
                helper(data_frac, trial_i)
        elif n_workers > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp.get_context("fork")
            ) as ex:
                results = tqdm(
                    ex.map(helper, data_fracs_vector, trials_vector),
                    total=len(data_fracs_vector),
                )
                for exc in results:
                    if exc:
                        print(exc)
        else:
            raise ValueError(f"value of {n_workers} must be >=1 ")

        self.aggregate_results(**kwargs)

    def run_baseline_trial(self, data_frac, trial_i, **kwargs):
        """Run a trial of the baseline model. Currently only
        supports supervised learning experiments.

        :param data_frac: Fraction of overall dataset size to use
        :type data_frac: float

        :param trial_i: The index of the trial
        :type trial_i: int
        """

        spec = kwargs["spec"]
        if isinstance(spec, RLSpec):
            raise NotImplementedError("Baselines are not yet implemented for RL")
        dataset = spec.dataset
        parse_trees = spec.parse_trees
        verbose = kwargs["verbose"]
        datagen_method = kwargs["datagen_method"]
        perf_eval_fn = kwargs["perf_eval_fn"]
        perf_eval_kwargs = kwargs["perf_eval_kwargs"]
        batch_epoch_dict = kwargs["batch_epoch_dict"]
        constraint_eval_kwargs = kwargs["constraint_eval_kwargs"]
        validation = kwargs["validation"]
        dataset_name = kwargs["dataset_name"]
        epsilon = spec.optimization_hyperparams['epsilon']
        if (
            batch_epoch_dict == {}
            and (spec.optimization_technique == "gradient_descent")
            and (spec.optimization_hyperparams["use_batches"] == True)
        ):
            warning_msg = (
                "WARNING: No batch_epoch_dict was provided. "
                "Each data_frac will use the same values "
                "for batch_size and n_epochs. "
                "This can have adverse effects, "
                "especially for small values of data_frac."
            )
            warnings.warn(warning_msg)

        trial_dir = os.path.join(
            self.results_dir, f"{self.model_name}_results", "trial_data"
        )

        os.makedirs(trial_dir, exist_ok=True)

        savename = os.path.join(
            trial_dir, f"data_frac_{epsilon:.4f}_trial_{trial_i}.csv"
        )
        if kwargs['n_downstreams'] > 1:
            savename = os.path.join(
                trial_dir, f"epsilon_{epsilon:.4f}_trial_{trial_i}_downstream_0.csv"
            )

        if os.path.exists(savename):
            if verbose:
                print(
                    f"Trial {trial_i} already run for "
                    f"this epsilon: {epsilon}. Skipping this trial. "
                )
            return

        ##############################################
        """ Setup for running baseline algorithm """
        ##############################################

        if datagen_method == "resample":
            # resampled_filename = os.path.join(
            #     self.results_dir, "resampled_dataframes", f"trial_{trial_i}.pkl"
            # )
            if dataset_name == 'adults':
                resampled_filename = os.path.join(
                    "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Adults", "resampled_dataframes", f"trial_{trial_i}.pkl"
                )
                # else:
                #     resampled_filename = os.path.join(
                #         "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Adult", "resampled_dataframes", f"trial_{trial_i}.pkl"
                #     )
            elif dataset_name == 'health':
                resampled_filename = os.path.join(
                    "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/health", "resampled_dataframes", f"trial_{trial_i}.pkl"
                )
            elif dataset_name == 'income':
                resampled_filename = os.path.join(
                    "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/income", "resampled_dataframes", f"trial_{trial_i}.pkl"
                )
            elif dataset_name == 'Face':
                if validation:
                    resampled_filename = os.path.join(
                        "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Face", "resampled_dataframes", f"trial_{trial_i}.pkl"
                    )
                else:
                    resampled_filename = os.path.join(
                        "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Face", "resampled_dataframes", f"trial_{trial_i}.pkl"
                    )
            resampled_dataset = load_pickle(resampled_filename)
            num_datapoints_tot = resampled_dataset.num_datapoints
            n_points = int(round(data_frac * num_datapoints_tot))

            if verbose:
                print(
                    f"Using resampled dataset {resampled_filename} "
                    f"with {num_datapoints_tot} datapoints"
                )
        else:
            raise NotImplementedError(
                f"datagen_method: {datagen_method} "
                f"not supported for regime: {regime}"
            )

        # Prepare features and labels
        features = resampled_dataset.features
        labels = resampled_dataset.labels
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels_l = []
            for i in range(labels.shape[-1]):
                labels_l.append(labels[:, i])
            labels = labels_l
        # Only use first n_points for this trial
        if type(features) == list:
            features = [x[:n_points] for x in features]
        else:
            features = features[:n_points]
        
        if type(labels) == list:
            labels = [x[:n_points] for x in labels]
        else:
            labels = labels[:n_points]

        # resample n data ponts. For representation learning use case
        # ix_resamp = np.random.choice(
        #     range(n_points), num_datapoints_tot, replace=True
        # )
        # if type(features) == list:
        #     features = [x[ix_resamp] for x in features]
        # else:
        #     features = features[ix_resamp]
        # if type(labels) == list:
        #     labels = [x[ix_resamp] for x in labels]
        # else:
        #     labels = labels[ix_resamp]
        # # sensitive_attrs = sensitive_attrs[ix_resamp]

        # # print(sensitive_attrs.shape)
        # n_points = num_datapoints_tot

        ####################################################
        """" Instantiate model and fit to resampled data """
        ####################################################
        X_test_baseline = perf_eval_kwargs["X"]
        device = perf_eval_kwargs["device"]
        batch_size, n_epochs = batch_epoch_dict[data_frac]
        s_dim = spec.optimization_hyperparams.get('s_dim', 1)
        print(" spec.optimization_hyperparams.get('s_dim', 1)",  spec.optimization_hyperparams.get('s_dim', 1))
        z_dim = spec.optimization_hyperparams["z_dim"]
        baseline_args = {"x_dim": spec.optimization_hyperparams.get("x_dim",0),
            "s_dim": s_dim,
            "y_dim": 1,
            "z1_enc_dim": 100,
            "z2_enc_dim": 100,
            "z1_dec_dim": 100,
            "x_dec_dim": 100,
            "z_dim": z_dim,
            "dropout_rate": spec.optimization_hyperparams["dropout"],
            "lr": spec.optimization_hyperparams.get("lr",0),
            "use_validation": validation,
            "downstream_bs": spec.optimization_hyperparams["downstream_bs"],
        }
        n_valid = round(spec.frac_data_in_safety * n_points)
        if self.model_name == "VFAE" or self.model_name == "VAE":
            if self.model_name == "VFAE":
                from .baselines.vfae_baseline import PytorchVFAE
            elif self.model_name == "random":
                from .baselines.random_vae import PytorchVFAE
            else:
                from .baselines.vae_baseline import PytorchVFAE
            baseline_args["epsilon"] = spec.optimization_hyperparams["epsilon"]
            baseline_model = PytorchVFAE(device=device, **baseline_args)
            # perf_eval_kwargs['eval_batch_size'] = len(X_test_baseline)
            if self.model_name == "random":
                y_pred = None, None, None
            else:
                y_pred = vae_predictions(
                    baseline_model, solution, X_test_baseline, **perf_eval_kwargs
                )
        if self.model_name == "controllable_vfae" or self.model_name == 'LMIFR':
            if self.model_name == "controllable_vfae":
                from .baselines.controllable_vfae import PytorchLMIFR
            else:
                from.baselines.lmifr_all import PytorchLMIFR
            baseline_args["lambda_init"] = spec.optimization_hyperparams["lambda_init"][0],
            baseline_args["epsilon"] = spec.optimization_hyperparams["epsilon"]
            baseline_args["hidden_dim"] = spec.optimization_hyperparams["hidden_dim"]
            baseline_model = PytorchLMIFR(device=device, **baseline_args)
            pu = np.mean(features[:, - s_dim - 1: -1], axis=0)
            baseline_model.set_pu(pu)
            # perf_eval_kwargs['eval_batch_size'] = len(X_test_baseline)
            # y_pred = vae_predictions(
            #     baseline_model, solution, X_test_baseline, **perf_eval_kwargs
            # )
        if self.model_name == 'FARE':
            if s_dim > 1:
                from.baselines.fare_multiclass import PytorchFARE
            else:
                from.baselines.fare import PytorchFARE
            baseline_args["labels"] = resampled_dataset.labels
            baseline_model = PytorchFARE(device=device, **baseline_args)
        if self.model_name == 'CFAIR':
            from.baselines.cfair import PytorchCFair
            baseline_model = PytorchCFair(device=device, **baseline_args)
        if self.model_name == "LAFTR":
            from.baselines.laftr import PytorchLAFTR
            baseline_args["hidden_dim"] = spec.optimization_hyperparams["hidden_dim"]
            baseline_model = PytorchLAFTR(device=device, **baseline_args)
        if self.model_name == "ICVAE":
            from .baselines.icvae_noadv import PytorchICVAEBaseline
            baseline_model = PytorchICVAEBaseline(device=device, **baseline_args)
            # perf_eval_kwargs['eval_batch_size'] = len(X_test_baseline)
            # y_pred = vae_predictions(model=baseline_model, solution=solution, X_test=X_test_baseline, **perf_eval_kwargs)
        if self.model_name == "FCRL":
            from .baselines.fcrl_baseline import PytorchFCRLBaseline
            baseline_args["s_num"] = s_dim if s_dim > 1 else 2 # number of different sensitive groups
            baseline_args["nce_size"] = z_dim
            baseline_model = PytorchFCRLBaseline(device=device, **baseline_args)
        if self.model_name == "cnn_controllable_vfae" or self.model_name == "cnn_icvae" or self.model_name == "cnn_lmifr_all" or self.model_name == "cnn_vfae_baseline" or self.model_name == "cnn_vae" or self.model_name == 'random':
            if self.model_name == "cnn_controllable_vfae":
                from .baselines.cnn_controllable_vfae import PytorchCNNLMIFR as model
            elif self.model_name == "cnn_lmifr_all":
                from .baselines.cnn_lmifr_all import PytorchCNNLMIFR as model
            elif self.model_name == "cnn_icvae":
                from .baselines.cnn_icvae import PytorchCNNICVAE as model
            elif self.model_name == "cnn_vfae_baseline":
                from .baselines.cnn_vfae_baseline import PytorchCNNVFAE as model
            elif self.model_name == "cnn_vae":
                from .baselines.cnn_vae import PytorchCNNVAE as model
            if self.model_name == 'random':
                from .baselines.random_vae import PytorchVFAE as model
                baseline_args["epsilon"] = spec.optimization_hyperparams["epsilon"]
                baseline_model = model(device=device, **baseline_args)
            else:
                baseline_args["epsilon"] = spec.optimization_hyperparams["epsilon"]
                baseline_args["lambda_init"] = spec.optimization_hyperparams["lambda_init"][0]
                    
                baseline_model = model(device=device, **baseline_args)
            X, S, Y = features
            pu = np.mean(S, axis=0)
            baseline_model.set_pu(pu)
            if self.model_name == 'random':
                baseline_model = self.model_name
        baseline_model.train(
            features, labels, batch_size=batch_size, num_epochs=n_epochs,data_frac=data_frac, n_valid=n_valid,
            X_test=perf_eval_kwargs["X"]
        )
        solution = baseline_model.get_model_params()
        # perf_eval_kwargs['eval_batch_size'] = spec.optimization_hyperparams["downstream_bs"]
        perf_eval_kwargs["downstream_bs"] = spec.optimization_hyperparams["downstream_bs"]
        perf_eval_kwargs["downstream_epochs"] = spec.optimization_hyperparams["downstream_epochs"]
        perf_eval_kwargs["downstream_lr"] = spec.optimization_hyperparams["downstream_lr"]
        perf_eval_kwargs["z_dim"] = spec.optimization_hyperparams["z_dim"]
        perf_eval_kwargs["hidden_dim"] = spec.optimization_hyperparams["hidden_dim"]
        perf_eval_kwargs["y_dim"] = spec.optimization_hyperparams["y_dim"]

        # if validation:
        if type(labels) == list:
            Y_valid = [y[-n_valid:] for y in labels]
            labels = [y[:-n_valid] for y in labels]
        else:
            Y_valid = labels[-n_valid:]
            labels = labels[:-n_valid]

        # perf_eval_kwargs["y"] = Y_valid
        if type(features) == list:
            X_valid = [x[-n_valid:] for x in features]
            features = [x[:-n_valid] for x in features]
        else:
            X_valid = features[-n_valid:]                        
            features = features[:-n_valid]
        # perf_eval_kwargs["X"] = X_valid
        
        if type(labels) == list:
            y_preds = []
            for y in labels:
                y_pred = unsupervised_downstream_predictions(
                    model=baseline_model, solution=solution,  X_train=features, Y_train=y, X_test=perf_eval_kwargs["X"], **perf_eval_kwargs)
                y_pred = None, None, y_pred
                y_preds.append(y_pred)
            y_pred = y_preds
        else:
            y_pred = unsupervised_downstream_predictions(
                model=baseline_model, solution=solution,  X_train=features, Y_train=labels, X_test=perf_eval_kwargs["X"], **perf_eval_kwargs)
            y_pred = None, None, y_pred

        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################
        # Handle whether solution was found
        solution_found = True
        if type(solution) == str and solution == "NSF":
            solution_found = False

        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################

        failed = False  # flag for whether we were actually safe on test set
        if solution_found:
            if type(y_pred) != list:
                performance  = [fn(y_pred, **perf_eval_kwargs) for fn in perf_eval_fn]
            else:
                performances = []
                labels = perf_eval_kwargs['y']
                if type(labels) != list and len(labels.shape) > 1 and labels.shape[-1] > 1:
                    labels_l = []
                    for i in range(labels.shape[-1]):
                        labels_l.append(labels[:, i])
                    labels = labels_l
                orig_y = perf_eval_kwargs['y']
                for i in range(len(y_pred)):
                    perf_eval_kwargs['y'] = labels[i]
                    performance = [fn(y_pred[i], **perf_eval_kwargs) for fn in perf_eval_fn]
                    performances.append(performance)
                performance = performances
                perf_eval_kwargs['y'] = orig_y

                # perf_eval_kwargs['y'] = labels
            if verbose:
                print(f"Performance = {performance}")
            if "eval_batch_size" in constraint_eval_kwargs:
                batch_size_safety = constraint_eval_kwargs["eval_batch_size"]
            elif spec.batch_size_safety is not None:
                batch_size_safety = spec.batch_size_safety
            else:
                batch_size_safety = spec.optimization_hyperparams.get("downstream_bs")

            # Determine whether this solution
            # violates any of the constraints
            # on the test dataset, which is the dataset from spec
            # if baseline_model != 'random':
            #     for parse_tree in parse_trees:
            #         parse_tree.reset_base_node_dict(reset_data=True)
            #         parse_tree.evaluate_constraint(
            #             theta=solution,
            #             dataset=dataset,
            #             model=baseline_model,
            #             regime="supervised_learning",
            #             branch="safety_test",
            #             batch_size_safety=batch_size_safety,
            #         )

            #         g = parse_tree.root.value
            #         parse_tree.reset_base_node_dict(reset_data=True)
            #         if g > 0 or np.isnan(g):
            #             failed = True
            #             if verbose:
            #                 print("Failed on test set")
            #         if verbose:
            #             print(f"g (baseline={self.model_name}) = {g}")
            # else:
            g = -1
        else:
            print("NSF")
            performance = np.nan

        # Write out file for this data_frac,trial_i combo
        colnames = ["epsilon", "data_frac", "trial_i", *[fn.__name__ for fn in perf_eval_fn], "g", "failed"]
        if type(performance) == list and type(performance[0]) == list :
            for i in range(len(performance)):
                if s_dim == 1:
                    dp = demographic_parity(y_pred[i], **perf_eval_kwargs)
                else:
                    dp = multiclass_demographic_parity(y_pred[i], **perf_eval_kwargs)
                failed = dp > epsilon
                data = [spec.optimization_hyperparams["epsilon"], data_frac, trial_i, *(performance[i]), g, failed]
                self.write_trial_result(data, colnames, trial_dir, downstream_i=i, verbose=kwargs["verbose"])
        else:
                data = [spec.optimization_hyperparams["epsilon"], data_frac, trial_i, *performance, g, failed]
                self.write_trial_result(data, colnames, trial_dir, verbose=kwargs["verbose"])
        return


class SeldonianExperiment(Experiment):
    def __init__(self, model_name, results_dir):
        """Class for running Seldonian experiments

        :param model_name: The string name of the Seldonian model,
                only option is currently: 'qsa' (quasi-Seldonian algorithm)
        :type model_name: str

        :param results_dir: Parent directory for saving any
                experimental results
        :type results_dir: str

        """
        super().__init__(model_name, results_dir)
        # if self.model_name != "qsa":
        #     raise NotImplementedError(
        #         "Seldonian experiments for model: "
        #         f"{self.model_name} are not supported."
        #     )

    def run_experiment(self, **kwargs):
        """Run the Seldonian experiment"""
        n_workers = kwargs["n_workers"]
        partial_kwargs = {
            key: kwargs[key] for key in kwargs if key not in ["data_fracs", "n_trials"]
        }
        partial_kwargs['model_name'] = self.model_name
        # Pass partial_kwargs onto self.QSA()
        helper = partial(self.run_QSA_trial, **partial_kwargs)

        data_fracs = kwargs["data_fracs"]
        epsilon = kwargs["spec"].optimization_hyperparams["epsilon"]
        kwargs['epsilons'] = [epsilon]
        n_trials = kwargs["n_trials"]
        data_fracs_vector = np.array([x for x in data_fracs for y in range(n_trials)])
        trials_vector = np.array(
            [x for y in range(len(data_fracs)) for x in range(n_trials)]
        )

        if n_workers == 1:
            for ii in range(len(data_fracs_vector)):
                data_frac = data_fracs_vector[ii]
                trial_i = trials_vector[ii]
                helper(data_frac, trial_i)
        elif n_workers > 1:
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp.get_context("fork")
            ) as ex:
                results = tqdm(
                    ex.map(helper, data_fracs_vector, trials_vector),
                    total=len(data_fracs_vector),
                )
                for exc in results:
                    if exc:
                        print(exc)
        else:
            raise ValueError(f"n_workers value of {n_workers} must be >=1 ")

        self.aggregate_results(**kwargs)

    def run_QSA_trial(self, data_frac, trial_i, **kwargs):
        """Run a trial of the quasi-Seldonian algorithm

        :param data_frac: Fraction of overall dataset size to use
        :type data_frac: float

        :param trial_i: The index of the trial
        :type trial_i: int
        """
        spec = kwargs["spec"]
        verbose = kwargs["verbose"]
        datagen_method = kwargs["datagen_method"]
        perf_eval_fn = kwargs["perf_eval_fn"]
        perf_eval_kwargs = kwargs["perf_eval_kwargs"]
        constraint_eval_fns = kwargs["constraint_eval_fns"]
        constraint_eval_kwargs = kwargs["constraint_eval_kwargs"]
        batch_epoch_dict = kwargs["batch_epoch_dict"]
        validation = kwargs["validation"]
        model_name = kwargs["model_name"]
        dataset_name = kwargs["dataset_name"]
        logfilename = kwargs["logfilename"]
        epsilon = spec.optimization_hyperparams['epsilon']
        if batch_epoch_dict == {} and spec.optimization_technique == "gradient_descent":
            warning_msg = (
                "WARNING: No batch_epoch_dict was provided. "
                "Each data_frac will use the same values "
                "for batch_size and n_epochs. "
                "This can have adverse effects, "
                "especially for small values of data_frac."
            )
            warnings.warn(warning_msg)
        regime = spec.dataset.regime

        trial_dir = os.path.join(self.results_dir, f"{model_name}_results", "trial_data")

        savename = os.path.join(
            trial_dir, f"epsilon_{epsilon:.4f}_trial_{trial_i}.csv"
        )

        if kwargs['n_downstreams'] > 1:
            savename = os.path.join(
                trial_dir, f"epsilon_{epsilon:.4f}_trial_{trial_i}_downstream_0.csv"
            )

        if os.path.exists(savename):
            if verbose:
                print(
                    f"Trial {trial_i} already run for "
                    f"this epsilon: {epsilon}. Skipping this trial. "
                )
            return

        os.makedirs(trial_dir, exist_ok=True)

        parse_trees = spec.parse_trees
        dataset = spec.dataset

        ##############################################
        """ Setup for running Seldonian algorithm """
        ##############################################

        if regime == "supervised_learning":
            if datagen_method == "resample":
                if dataset_name == 'adults':
                    # if validation:
                    resampled_filename = os.path.join(
                        "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Adults", "resampled_dataframes", f"trial_{trial_i}.pkl"
                    )
                    # else:
                    #     resampled_filename = os.path.join(
                    #         "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Adult", "resampled_dataframes", f"trial_{trial_i}.pkl"
                    #     )
                elif dataset_name == 'health':
                    resampled_filename = os.path.join(
                            "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/health", "resampled_dataframes", f"trial_{trial_i}.pkl"
                        )
                elif dataset_name == 'income':
                    resampled_filename = os.path.join(
                            "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/income", "resampled_dataframes", f"trial_{trial_i}.pkl"
                        )
                elif dataset_name == 'Face':
                    if validation:
                        resampled_filename = os.path.join(
                            "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Face", "resampled_dataframes", f"trial_{trial_i}.pkl"
                        )
                    else:
                        resampled_filename = os.path.join(
                            "/work/pi_pgrabowicz_umass_edu/yluo/SeldonianExperimentResults/Face", "resampled_dataframes", f"trial_{trial_i}.pkl"
                        )
                resampled_dataset = load_pickle(resampled_filename)
                num_datapoints_tot = resampled_dataset.num_datapoints
                n_points = int(round(data_frac * num_datapoints_tot))
                if verbose:
                    print(
                        f"Using resampled dataset {resampled_filename} "
                        f"with {num_datapoints_tot} datapoints"
                    )
                    if n_points < 1:
                        raise ValueError(
                            f"This data_frac={data_frac} "
                            f"results in {n_points} data points. "
                            "Must have at least 1 data point to run a trial."
                        )

                features = resampled_dataset.features.copy()
                labels = resampled_dataset.labels.copy()
                if type(labels) != list and len(labels.shape) > 1 and labels.shape[-1] > 1:
                    labels_l = []
                    for i in range(labels.shape[-1]):
                        labels_l.append(labels[:, i])
                    labels = labels_l
                sensitive_attrs = resampled_dataset.sensitive_attrs.copy()
                # Only use first n_points for this trial
                if type(features) == list:
                    features = [x[:n_points] for x in features]
                else:
                    features = features[:n_points]
                if type(labels) == list:
                    labels = [x[:n_points] for x in labels]
                else:
                    labels = labels[:n_points]
                sensitive_attrs = sensitive_attrs[:n_points]
            else:
                raise NotImplementedError(
                    f"Eval method {datagen_method} "
                    f"not supported for regime={regime}"
                )
            dataset_for_experiment = SupervisedDataSet(
                features=features.copy(),
                labels=labels.copy(),
                sensitive_attrs=sensitive_attrs.copy(),
                num_datapoints=n_points,
                meta_information=resampled_dataset.meta_information,
            )

            # Make a new spec object
            # and update the dataset

            spec_for_experiment = copy.deepcopy(spec)
            spec_for_experiment.dataset = dataset_for_experiment

        # If optimizing using gradient descent,
        # and using mini-batches,
        # update the batch_size and n_epochs
        # using batch_epoch_dict
        if spec_for_experiment.optimization_technique == "gradient_descent":
            if spec_for_experiment.optimization_hyperparams["use_batches"] == True:
                batch_size, n_epochs = batch_epoch_dict[data_frac]
                spec_for_experiment.optimization_hyperparams["batch_size"] = batch_size
                spec_for_experiment.optimization_hyperparams["n_epochs"] = n_epochs
        ################################
        """" Run Seldonian algorithm """
        ################################
        # try:
        SA = SeldonianAlgorithm(spec_for_experiment)
        passed_safety, solution = SA.run(write_cs_logfile=verbose, debug=verbose, logfilename=logfilename)
        # except (ValueError, ZeroDivisionError):
        #     passed_safety = False
        #     solution = "NSF"

        if verbose:
            print("Solution from running seldonian algorithm:")
            print(solution)
            print()

        # Handle whether solution was found
        solution_found = True
        if type(solution) == str and solution == "NSF":
            solution_found = False

        #########################################################
        """" Calculate performance and safety on ground truth """
        #########################################################

        failed = False  # flag for whether we were actually safe on test set
        g = 0
        if solution_found or validation:
            solution = copy.deepcopy(solution)
            # If passed the safety test, calculate performance
            # using solution
            if passed_safety or validation:
                if verbose:
                    print("Passed safety test! Calculating performance")

                #############################
                """ Calculate performance """
                #############################
                if regime == "supervised_learning":
                    #evaluation set
                    if validation:
                        val_label = labels
                        if type(labels) == list:
                            val_label = labels[0]
                        Y_test=val_label[-SA.n_safety:]
                        labels = val_label[:-SA.n_safety]
                        perf_eval_kwargs["y"] = Y_test
                        if type(features) == list:
                            X_test = [x[-SA.n_safety:] for x in features]
                            features = [x[:-SA.n_safety] for x in features]
                        else:
                            X_test = features[-SA.n_safety:]                        
                            features = features[:-SA.n_safety]

                        perf_eval_kwargs["X"] = X_test
                    else:
                    # test set
                        X_test = perf_eval_kwargs["X"]
                        Y_test = perf_eval_kwargs["y"]
                        if len(Y_test.shape) > 1 and Y_test.shape[-1] > 1:
                            Y_test_l = []
                            for i in range(Y_test.shape[-1]):
                                Y_test_l.append(Y_test[:, i])
                            Y_test = Y_test_l
                    
                    model = SA.model
                    if spec.dataset.meta_information.get("self_supervised", False):
                        print("train downstream supervised model")
                        perf_eval_kwargs["downstream_bs"] = spec_for_experiment.optimization_hyperparams["downstream_bs"]
                        perf_eval_kwargs["eval_batch_size"] = spec_for_experiment.optimization_hyperparams["downstream_bs"]
                        perf_eval_kwargs["downstream_epochs"] = spec_for_experiment.optimization_hyperparams["downstream_epochs"]
                        perf_eval_kwargs["downstream_lr"] = spec_for_experiment.optimization_hyperparams["downstream_lr"]
                        print("spec_for_experiment.optimization_hyperparams[z_dim]", spec_for_experiment.optimization_hyperparams["z_dim"])
                        perf_eval_kwargs["z_dim"] = spec_for_experiment.optimization_hyperparams["z_dim"]
                        perf_eval_kwargs["hidden_dim"] = spec_for_experiment.optimization_hyperparams["hidden_dim"]
                        perf_eval_kwargs["y_dim"] = spec_for_experiment.optimization_hyperparams["y_dim"]
                        if type(labels) == list:
                            y_preds = []
                            for y in labels:
                                y_pred = unsupervised_downstream_predictions(
                                    model=model, solution=solution,  X_train=features, Y_train=y, X_test=X_test, **perf_eval_kwargs)
                                y_pred = None, None, y_pred
                                y_preds.append(y_pred)
                            y_pred = y_preds
                        else:
                            y_pred = unsupervised_downstream_predictions(
                                model=model, solution=solution,  X_train=features, Y_train=labels, X_test=X_test, **perf_eval_kwargs)
                            y_pred = None, None, y_pred
                    else:
                        # Batch the prediction if specified
                        if "eval_batch_size" in perf_eval_kwargs:
                            y_pred = batch_predictions(
                                model=model,
                                solution=solution,
                                X_test=X_test,
                                **perf_eval_kwargs,
                            )
                        else:
                            y_pred = model.predict(solution, X_test)
                    if type(y_pred) != list:
                        performance  = [fn(y_pred, model=model, **perf_eval_kwargs) for fn in perf_eval_fn]
                    else:
                        performances = []
                        labels = perf_eval_kwargs['y']
                        if len(labels.shape) > 1 and labels.shape[-1] > 1:
                            labels_l = []
                            for i in range(labels.shape[-1]):
                                labels_l.append(labels[:, i])
                            labels = labels_l
                        orig_y = perf_eval_kwargs['y']
                        for i in range(len(y_pred)):
                            perf_eval_kwargs['y'] = labels[i]
                            performance = [fn(y_pred[i], model=model, **perf_eval_kwargs) for fn in perf_eval_fn]
                            performances.append(performance)
                        performance = performances
                        perf_eval_kwargs['y'] = orig_y
                        # perf_eval_kwargs['y'] = labels
                
                print("features final", len(spec_for_experiment.dataset.features))
                if verbose:
                    print(f"Performance = {performance}")

                ########################################
                """ Calculate safety on ground truth """
                ########################################
                if verbose:
                    print(
                        "Determining whether solution "
                        "is actually safe on ground truth"
                    )

                if constraint_eval_fns == []:
                    constraint_eval_kwargs["model"] = model
                    constraint_eval_kwargs["spec_orig"] = spec
                    constraint_eval_kwargs["spec_for_experiment"] = spec_for_experiment
                    constraint_eval_kwargs["regime"] = regime
                    constraint_eval_kwargs["branch"] = "safety_test"
                    constraint_eval_kwargs["verbose"] = verbose

                failed, g = False, -0.01#
                # self.evaluate_constraint_functions(
                #     solution=solution,
                #     constraint_eval_fns=constraint_eval_fns,
                #     constraint_eval_kwargs=constraint_eval_kwargs,
                # )

                if verbose:
                    if failed:
                        print("Solution was not actually safe on ground truth!")
                    else:
                        print("Solution was safe on ground truth")
                    print()
            else:
                if verbose:
                    print("Failed safety test ")
                    if type(labels) == list:
                        performance = []
                        for i in range(len(labels)):
                            performance.append([np.nan] * len(perf_eval_fn))
                    else:
                        performance = [np.nan] * len(perf_eval_fn)
        else:
            if verbose:
                print("NSF")
            if type(labels) == list:
                performance = []
                for i in range(len(labels)):
                    performance.append([np.nan] * len(perf_eval_fn))
            else:
                performance = [np.nan] * len(perf_eval_fn)
        
        for parse_tree in spec_for_experiment.parse_trees:
            parse_tree.reset_base_node_dict(reset_data=True)
        # Write out file for this data_frac,trial_i combo
        colnames = ["epsilon", "data_frac", "trial_i", *[fn.__name__ for fn in perf_eval_fn], "passed_safety", "g", "failed"]
        if kwargs['n_downstreams'] > 1:
            s_dim = spec.optimization_hyperparams.get('s_dim', 1)
            for i in range(len(performance)):
                
                if s_dim == 1:
                    if solution_found and passed_safety:
                        dp = demographic_parity(y_pred[i], **perf_eval_kwargs)
                        failed = dp > epsilon
                else:
                    if solution_found and passed_safety:
                        dp = multiclass_demographic_parity(y_pred[i], **perf_eval_kwargs)
                        failed = dp > epsilon
                data = [spec_for_experiment.optimization_hyperparams["epsilon"], data_frac, trial_i, *(performance[i]), passed_safety, g, failed]
                self.write_trial_result(data, colnames, trial_dir, downstream_i=i, verbose=kwargs["verbose"])
        else:
                data = [spec_for_experiment.optimization_hyperparams["epsilon"], data_frac, trial_i, *performance, passed_safety, g, failed]
                self.write_trial_result(data, colnames, trial_dir, verbose=kwargs["verbose"])
        return

    def evaluate_constraint_functions(
        self, solution, constraint_eval_fns, constraint_eval_kwargs
    ):
        """Helper function for QSA() to evaluate
        the constraint functions to determine
        whether solution was safe on ground truth

        :param solution: The weights of the model found
                during candidate selection in a given trial
        :type solution: numpy ndarray

        :param constraint_eval_fns: List of functions
                to use to evaluate each constraint.
                An empty list results in using the parse
                tree to evaluate the constraints
        :type constraint_eval_fns: List(function)

        :param constraint_eval_kwargs: keyword arguments
                to pass to each constraint function
                in constraint_eval_fns
        :type constraint_eval_kwargs: dict
        """
        # Use safety test branch so the confidence bounds on
        # leaf nodes are not inflated
        failed = False
        if constraint_eval_fns == []:
            """User did not provide their own functions
            to evaluate the constraints. Use the default:
            the parse tree has a built-in way to evaluate constraints.
            """
            constraint_eval_kwargs["theta"] = solution
            spec_orig = constraint_eval_kwargs["spec_orig"]
            spec_for_experiment = constraint_eval_kwargs["spec_for_experiment"]
            regime = constraint_eval_kwargs["regime"]
            if "eval_batch_size" in constraint_eval_kwargs:
                constraint_eval_kwargs["batch_size_safety"] = constraint_eval_kwargs[
                    "eval_batch_size"
                ]
            if regime == "supervised_learning":
                # Use the original dataset as ground truth
                constraint_eval_kwargs["dataset"] = spec_orig.dataset

            elif regime == "reinforcement_learning":
                episodes_for_eval = constraint_eval_kwargs["episodes_for_eval"]

                dataset_for_eval = RLDataSet(
                    episodes=episodes_for_eval,
                    meta_information=spec_for_experiment.dataset.meta_information,
                    regime=regime,
                )

                constraint_eval_kwargs["dataset"] = dataset_for_eval

            for parse_tree in spec_for_experiment.parse_trees:
                parse_tree.reset_base_node_dict(reset_data=True)
                parse_tree.evaluate_constraint(batch_size_safety=spec_for_experiment.batch_size_safety,
                                               **constraint_eval_kwargs)

                g = parse_tree.root.value
                if g > 0 or np.isnan(g):
                    failed = True

        else:
            # User provided functions to evaluate constraints
            for eval_fn in constraint_eval_fns:
                g = eval_fn(solution)
                if g > 0 or np.isnan(g):
                    failed = True
        return failed, g