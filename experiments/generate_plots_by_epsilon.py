""" Module for making the three plots """

import os
import glob
import pickle
import autograd.numpy as np  # Thinly-wrapped version of Numpy
import pandas as pd
import matplotlib
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import style

from seldonian.utils.io_utils import save_pickle

from .experiments import BaselineExperiment, SeldonianExperiment
from .utils import generate_resampled_datasets

seldonian_model_set = set(["FRG_0.1_sup", "FRG_0.1_unsup", "FRG_supervised", "qsa","headless_qsa", "sa", "qsa_cvfae", "qsa_icvae", "qsa_fcrl"])
plot_colormap = matplotlib.cm.get_cmap("tab10")
marker_list = ["s", "p", "d", "*", "x", "h", "+", 'o']


class PlotGenerator:
    def __init__(
        self,
        spec,
        n_trials,
        epsilons,
        data_fracs,
        datagen_method,
        perf_eval_fn,
        results_dir,
        n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs={},
        constraint_eval_kwargs={},
        batch_epoch_dict={},
        n_downstreams=1,
    ):
        """Class for running Seldonian experiments
        and generating the three plots:
        1) Performance
        2) Solution rate
        3) Failure rate
        all plotted vs. amount of data used

        :param spec: Specification object for running the
                Seldonian algorithm
        :type spec: seldonian.spec.Spec object

        :param n_trials: The number of times the
                Seldonian algorithm is run for each data fraction.
                Used for generating error bars
        :type n_trials: int

        :param epsilons: The error allowed for the constraints
                (the horizontal axis on the three plots).
        :type epsilons: List(float)

        :param datagen_method: Method for generating data that is used
                to run the Seldonian algorithm for each trial
        :type datagen_method: str, e.g. "resample"

        :param perf_eval_fn: Function used to evaluate the performance
                of the model obtained in each trial, with signature:
                func(theta,**kwargs), where theta is the solution
                from candidate selection
        :type perf_eval_fn: function or class method

        :param results_dir: The directory in which to save the results
        :type results_dir: str

        :param n_workers: The number of workers to use if
                using multiprocessing
        :type n_workers: int

        :param constraint_eval_fns: List of functions used to evaluate
                the constraints on ground truth. If an empty list is provided,
                the constraints are evaluated using the parse tree
        :type constraint_eval_fns: List(function or class method),
                defaults to []

        :param perf_eval_kwargs: Extra keyword arguments to pass to
                perf_eval_fn
        :type perf_eval_kwargs: dict

        :param constraint_eval_kwargs: Extra keyword arguments to pass to
                the constraint_eval_fns
        :type constraint_eval_kwargs: dict

        :param batch_epoch_dict: Instruct batch sizes and n_epochs
                for each data frac
        :type batch_epoch_dict: dict
        """
        self.spec = spec
        self.n_trials = n_trials
        self.epsilons = epsilons
        self.data_fracs = data_fracs
        self.datagen_method = datagen_method
        self.perf_eval_fn = perf_eval_fn
        self.results_dir = results_dir
        self.n_workers = n_workers
        self.constraint_eval_fns = constraint_eval_fns
        self.perf_eval_kwargs = perf_eval_kwargs
        self.constraint_eval_kwargs = constraint_eval_kwargs
        self.batch_epoch_dict = batch_epoch_dict
        self.n_downstreams = n_downstreams

    def make_plots(
        self,
        model_label_dict={},
        fontsize=12,
        legend_fontsize=8,
        performance_label=["accuracy"],
        performance_yscale="linear",
        performance_ylims=[],
        marker_size=20,
        save_format="png",
        show_title=True,
        custom_title=None,
        include_legend=True,
        savename=None,
        prob_performance_below=[None],
        result_filename_suffix="",
        xticks=None
    ):
        """Make the three plots from results files saved to
        self.results_dir

        :param model_label_dict: An optional dictionary where keys
                are model names and values are the names you want
                shown in the legend.
        :type model_label_dict: int

        :param fontsize: The font size to use for the axis labels
        :type fontsize: int

        :param legend_fontsize: The font size to use for text
                in the legend
        :type legend_fontsize: int

        :param performance_label: The y axis label on the performance
                plot you want to use.
        :type performance_label: str, defaults to "accuracy"

        :param savename: If not None, the filename path to which the plot
                will be saved on disk.
        :type savename: str, defaults to None
        """
        assert(len(performance_label) == len(self.perf_eval_fn))
        plt.style.use("bmh")
        # plt.style.use('grayscale')
        regime = self.spec.dataset.regime
        tot_data_size = self.spec.dataset.num_datapoints

        # Read in constraints
        parse_trees = self.spec.parse_trees

        constraint_dict = {}
        for pt_ii, pt in enumerate(parse_trees):
            delta = pt.delta
            constraint_str = pt.constraint_str
            constraint_dict[f"constraint_{pt_ii}"] = {
                "delta": delta,
                "constraint_str": constraint_str,
            }

        constraints = list(constraint_dict.keys())

        # Figure out what experiments we have from subfolders in results_dir
        subfolders = [
            os.path.basename(f) for f in os.scandir(self.results_dir) if f.is_dir()
        ]
        all_models = [
            x.split("_results")[0] for x in subfolders if x.endswith("_results")
        ]
        seldonian_models = list(set(all_models).intersection(seldonian_model_set))
        baselines = sorted(list(set(all_models).difference(seldonian_model_set)))
        if not (seldonian_models or baselines):
            print("No results for Seldonian models or baselines found ")
            return

        ## BASELINE RESULTS SETUP
        baseline_dict = {}
        for baseline in baselines:
            baseline_dict[baseline] = {}
            savename_baseline = os.path.join(
                self.results_dir, f"{baseline}_results", f"{baseline}_results{result_filename_suffix}.csv"
            )
            df_baseline = pd.read_csv(savename_baseline)
            validation_pass = os.path.join(
                self.results_dir,
                f"{baseline}_results",
                f"{baseline}_valid_pass.csv",
            )
            # df_valid = pd.read_csv(validation_pass)
            fn_name = self.perf_eval_fn[3].__name__
            df_baseline["solution_returned"] = df_baseline[fn_name].apply(
                lambda x: ~np.isnan(x)
            )

            valid_mask = ~np.isnan(df_baseline[fn_name])
            df_baseline_valid = df_baseline[valid_mask]
            # Get the list of all epsilons
            X_all = df_baseline.groupby("epsilon").mean().index# * tot_data_size
            # Get the list of epsilons for which there is at least one trial that has non-nan performance
            X_valid = (
                df_baseline_valid.groupby("epsilon").mean().index# * tot_data_size
            )

            baseline_dict[baseline]["df_baseline"] = df_baseline.copy()
            baseline_dict[baseline]["df_baseline_valid"] = df_baseline_valid.copy()
            baseline_dict[baseline]["X_all"] = X_all
            baseline_dict[baseline]["X_valid"] = X_valid
            # baseline_dict[baseline]["df_valid"] = df_valid.copy()

        # SELDONIAN RESULTS SETUP
        seldonian_dict = {}
        for seldonian_model in seldonian_models:
            seldonian_dict[seldonian_model] = {}
            savename_seldonian = os.path.join(
                self.results_dir,
                f"{seldonian_model}_results",
                f"{seldonian_model}_results{result_filename_suffix}.csv",
            )
            validation_pass = os.path.join(
                self.results_dir,
                f"{seldonian_model}_results",
                f"{seldonian_model}_valid_pass.csv",
            )
            df_seldonian = pd.read_csv(savename_seldonian)
            # df_seldonian_valid = pd.read_csv(validation_pass)
            passed_mask = df_seldonian["passed_safety"] == True
            df_seldonian_passed = df_seldonian[passed_mask]
            # Get the list of all epsilons
            X_all = df_seldonian.groupby("epsilon").mean().index #* tot_data_size
            # Get the list of epsilons for which there is at least one trial that passed the safety test
            X_passed = (
                df_seldonian_passed.groupby("epsilon").mean().index #* tot_data_size
            )
            seldonian_dict[seldonian_model]["df_seldonian"] = df_seldonian.copy()
            seldonian_dict[seldonian_model][
                "df_seldonian_passed"
            ] = df_seldonian_passed.copy()
            seldonian_dict[seldonian_model]["X_all"] = X_all
            seldonian_dict[seldonian_model]["X_passed"] = X_passed

        ## PLOTTING SETUP
        if include_legend:
            figsize = (18, 4.5)
        else:
            figsize = (18, 4)
        fig = plt.figure(figsize=figsize)
        plot_index = 1
        n_rows = len(constraints)
        n_cols = 2 + len(self.perf_eval_fn)
        fontsize = fontsize
        legend_fontsize = legend_fontsize
        legend_handles = []
        legend_labels = []
        ## Loop over constraints and make three plots for each constraint
        for ii, constraint in enumerate(constraints):
            constraint_str = constraint_dict[constraint]["constraint_str"]
            delta = constraint_dict[constraint]["delta"]

            # SETUP FOR PLOTTING
            ax_performances = []
            for idx, label in enumerate(performance_label):
                if idx == 0:
                    ax_performance = fig.add_subplot(n_rows, n_cols, plot_index)
                else:
                    ax_performance = fig.add_subplot(n_rows, n_cols, plot_index, sharex=ax_performance)
                ax_performance.set_ylabel(label, fontsize=fontsize)
                plot_index += 1
                if ii == len(constraints) - 1:
                    ax_performance.set_xlabel("ε", fontsize=fontsize)
                # ax_performance.set_xscale("log")
                if performance_yscale.lower() == "log":
                    ax_performance.set_yscale("log")
                ax_performances.append(ax_performance)
            ax_sr = fig.add_subplot(n_rows, n_cols, plot_index, sharex=ax_performance)
            plot_index += 1
            ax_fr = fig.add_subplot(n_rows, n_cols, plot_index, sharex=ax_performance)
            plot_index += 1

            # Plot title (put above middle plot)
            if show_title:
                if custom_title:
                    title = custom_title
                else:
                    title = f"constraint: \ng={constraint_str}"
                ax_sr.set_title(title, y=1.05, fontsize=10)

            # Plot labels
            
            ax_sr.set_ylabel("Probability of solution", fontsize=fontsize)
            ax_fr.set_ylabel("Pr$(\~{g}_{\epsilon}'(\phi) > 0)$", fontsize=fontsize)
            # ax_fr.set_ylabel(performance_label[1] + " on validation set", fontsize=fontsize)

            # Only put horizontal axis labels on last row of plots
            if ii == len(constraints) - 1:
                # ax_performance.set_xlabel('Training samples',fontsize=fontsize)
                # ax_sr.set_xlabel('Training samples',fontsize=fontsize)
                # ax_fr.set_xlabel('Training samples',fontsize=fontsize)

                ax_sr.set_xlabel("ε", fontsize=fontsize)
                ax_fr.set_xlabel("ε", fontsize=fontsize)

            # axis scaling
            
            # ax_sr.set_xscale("log")
            # ax_fr.set_xscale("log")

            # locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
            # locmin = matplotlib.ticker.LogLocator(
            #     base=10.0,
            #     subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            #     numticks=12,
            # )
            for ax in [*ax_performances, ax_sr, ax_fr]:
                ax.minorticks_on()
                
                ax.xaxis.set_ticks(xticks)
                ax.set_xlim(xticks[0]-0.01, xticks[-1] + 0.01)
                # ax.xaxis.set_major_locator(locmaj)
                # ax.xaxis.set_minor_locator(locmin)
                # ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

            ########################
            ### PERFORMANCE PLOT ###
            ########################

            # Baseline performance
            for baseline_i, baseline in enumerate(baselines):
                baseline_color = plot_colormap(
                    baseline_i + len(seldonian_models)
                )  # 0 is reserved for Seldonian model
                this_baseline_dict = baseline_dict[baseline]
                df_baseline_valid = this_baseline_dict["df_baseline_valid"]
                n_trials = df_baseline_valid["trial_i"].max() + 1

                # Performance
                X_valid_baseline = this_baseline_dict["X_valid"]
                for idx, ax_performance in enumerate(ax_performances):
                    if prob_performance_below[idx] is not None:
                        baseline_performance = df_baseline_valid[self.perf_eval_fn[idx].__name__]
                        df_baseline_valid[self.perf_eval_fn[idx].__name__] = (
                            baseline_performance - prob_performance_below[idx] > 0)
                    baseline_mean_performance = df_baseline_valid.groupby(
                        "epsilon"
                    ).mean()[self.perf_eval_fn[idx].__name__]
                    baseline_std_performance = df_baseline_valid.groupby("epsilon").std()[
                        self.perf_eval_fn[idx].__name__
                    ]
                    baseline_ste_performance = [std / np.sqrt(n_trials) for std in baseline_std_performance]
                    (pl,) = ax_performance.plot(
                        X_valid_baseline,
                        baseline_mean_performance,
                        color=baseline_color,
                        label=baseline,
                    )
                    
                    
                    ax_performance.scatter(
                        X_valid_baseline,
                        baseline_mean_performance,
                        color=baseline_color,
                        s=marker_size,
                        marker=marker_list[baseline_i],
                    )
                    ax_performance.fill_between(
                        X_valid_baseline,
                        baseline_mean_performance - baseline_ste_performance,
                        baseline_mean_performance + baseline_ste_performance,
                        color=baseline_color,
                        alpha=0.5,
                    )
                legend_handles.append(pl)
                if baseline in model_label_dict:
                    legend_labels.append(model_label_dict[baseline])
                else:
                    legend_labels.append(baseline)
            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i)
                df_seldonian_passed = this_seldonian_dict["df_seldonian_passed"]
                X_passed_seldonian = this_seldonian_dict["X_passed"]
                for idx, ax_performance in enumerate(ax_performances):
                    eval_name = self.perf_eval_fn[idx].__name__
                    if prob_performance_below[idx] is not None:
                        performance = df_seldonian_passed[eval_name].fillna(0)
                        df_seldonian_passed[eval_name] = (
                            performance - prob_performance_below[idx] > 0)
                        ax_performance.axhline(
                            y=prob_performance_below[idx], color="k", linestyle="--", label=f"epsilon={prob_performance_below[idx]}"
                        )
                    mean_performance = df_seldonian_passed.groupby("epsilon").mean()[
                        eval_name
                    ]
                    std_performance = df_seldonian_passed.groupby("epsilon").std()[
                        eval_name
                    ]
                    n_passed = df_seldonian_passed.groupby("epsilon").count()[
                        eval_name
                    ]
                    ste_performance = std_performance / np.sqrt(n_passed)
                    
                    (pl,) = ax_performance.plot(
                        X_passed_seldonian,
                        mean_performance,
                        color=seldonian_color,
                        linestyle="--",
                    )
                    
   
                    ax_performance.scatter(
                        X_passed_seldonian,
                        mean_performance,
                        color=seldonian_color,
                        s=marker_size,
                        marker="o",
                    )
                    ax_performance.fill_between(
                        X_passed_seldonian,
                        mean_performance - ste_performance,
                        mean_performance + ste_performance,
                        color=seldonian_color,
                        alpha=0.5,
                    )
                legend_handles.append(pl)
                if seldonian_model in model_label_dict:
                    legend_labels.append(model_label_dict[seldonian_model])
                else:
                    legend_labels.append(seldonian_model)
            ax_performance.set_ylim(-0.05, 1.05)
            if performance_ylims:
                for ax_performance in ax_performances:
                    ax_performance.set_ylim(*performance_ylims)
            
            ##########################
            ### SOLUTION RATE PLOT ###
            ##########################

            # Plot baseline solution rate
            # (sometimes it doesn't return a solution due to not having enough training data
            # to run model.fit() )
            for baseline_i, baseline in enumerate(baselines):
                this_baseline_dict = baseline_dict[baseline]
                X_all_baseline = this_baseline_dict["X_all"]
                baseline_color = plot_colormap(baseline_i + len(seldonian_models))
                df_baseline = this_baseline_dict["df_baseline"]
                n_trials = df_baseline["trial_i"].max() + 1
                mean_sr = df_baseline.groupby("epsilon").mean()["solution_returned"]
                std_sr = df_baseline.groupby("epsilon").std()["solution_returned"]
                ste_sr = std_sr / np.sqrt(n_trials)

                X_all_baseline = this_baseline_dict["X_all"]

                ax_sr.plot(
                    X_all_baseline, mean_sr, color=baseline_color, label=baseline
                )
                ax_sr.scatter(
                    X_all_baseline,
                    mean_sr,
                    color=baseline_color,
                    s=marker_size,
                    marker=marker_list[baseline_i],
                )
                ax_sr.fill_between(
                    X_all_baseline,
                    mean_sr - ste_sr,
                    mean_sr + ste_sr,
                    color=baseline_color,
                    alpha=0.5,
                )

            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i)
                df_seldonian = this_seldonian_dict["df_seldonian"]
                n_trials = df_seldonian["trial_i"].max() + 1
                mean_sr = df_seldonian.groupby("epsilon").mean()["passed_safety"]
                std_sr = df_seldonian.groupby("epsilon").std()["passed_safety"]
                ste_sr = std_sr / np.sqrt(n_trials)

                X_all_seldonian = this_seldonian_dict["X_all"]

                ax_sr.plot(
                    X_all_seldonian,
                    mean_sr,
                    color=seldonian_color,
                    linestyle="--",
                    label="FRG",
                )
                ax_sr.scatter(
                    X_all_seldonian,
                    mean_sr,
                    color=seldonian_color,
                    s=marker_size,
                    marker="o",
                )
                ax_sr.fill_between(
                    X_all_seldonian,
                    mean_sr - ste_sr,
                    mean_sr + ste_sr,
                    color=seldonian_color,
                    alpha=0.5,
                )

            ax_sr.set_ylim(-0.05, 1.05)

            ##########################
            ### FAILURE RATE PLOT ###
            ##########################

            # Baseline failure rate
            
            for baseline_i, baseline in enumerate(baselines):
                baseline_color = plot_colormap(baseline_i + len(seldonian_models))
                # Baseline performance
                this_baseline_dict = baseline_dict[baseline]
                df_baseline_valid = this_baseline_dict["df_baseline_valid"]
                n_trials = df_baseline_valid["trial_i"].max() + 1

                print(baseline)

                baseline_mean_fr = df_baseline_valid.groupby("epsilon").mean()[
                    "failed"
                ]
                # df_valid = this_baseline_dict["df_valid"]
                # baseline_mean_fr = df_valid.groupby("data_frac").mean()[
                #     "passed_safety"
                # ]
                # baseline_mean_fr = 1 - baseline_mean_fr / 10.0
                
                baseline_std_fr = df_baseline_valid.groupby("epsilon").std()["failed"]
                # baseline_std_fr = df_valid.groupby("data_frac").std()["passed_safety"]
                
                baseline_ste_fr = baseline_std_fr / np.sqrt(n_trials)

                X_valid_baseline = this_baseline_dict["X_valid"]
                ax_fr.plot(
                    X_valid_baseline,
                    baseline_mean_fr,
                    color=baseline_color,
                    label=baseline,
                )
                ax_fr.scatter(
                    X_valid_baseline,
                    baseline_mean_fr,
                    color=baseline_color,
                    marker=marker_list[baseline_i],
                    s=marker_size,
                )
                ax_fr.fill_between(
                    X_valid_baseline,
                    baseline_mean_fr - baseline_ste_fr,
                    baseline_mean_fr + baseline_ste_fr,
                    color=baseline_color,
                    alpha=0.5,
                )

            for seldonian_i, seldonian_model in enumerate(seldonian_models):
                this_seldonian_dict = seldonian_dict[seldonian_model]
                seldonian_color = plot_colormap(seldonian_i)
                df_seldonian = this_seldonian_dict["df_seldonian"]
                n_trials = df_seldonian["trial_i"].max() + 1
                mean_fr = df_seldonian.groupby("epsilon").mean()["failed"]
                std_fr = df_seldonian.groupby("epsilon").std()["failed"]
                # mean_fr = df_seldonian_valid.groupby("data_frac").mean()["passed_safety"]
                # mean_fr =1 - mean_fr/10.0
                # std_fr = df_seldonian_valid.groupby("data_frac").std()["passed_safety"]
                ste_fr = std_fr / np.sqrt(n_trials)

                X_all_seldonian = this_seldonian_dict["X_all"]
                
                ax_fr.plot(
                    X_all_seldonian,
                    mean_fr,
                    color=seldonian_color,
                    linestyle="--",
                    label="FRG",
                )
                ax_fr.fill_between(
                    X_all_seldonian,
                    mean_fr - ste_fr,
                    mean_fr + ste_fr,
                    color=seldonian_color,
                    alpha=0.5,
                )
                ax_fr.scatter(
                    X_all_seldonian,
                    mean_fr,
                    color=seldonian_color,
                    s=marker_size,
                    marker="o",
                )
                ax_fr.axhline(
                    y=delta, color="k", linestyle="--", label=f"delta={delta}"
                )
            ax_fr.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        if include_legend:
            fig.subplots_adjust(bottom=0.25)
            ncol = 8
            fig.legend(
                legend_handles[::-1],
                legend_labels[::-1],
                bbox_to_anchor=(0.5, 0.15),
                loc="upper center",
                ncol=ncol,
            )

        if savename:
            # plt.savefig(savename,format=save_format,dpi=600)
            plt.savefig(savename, format=save_format)
            print(f"Saved {savename}")
        else:
            plt.show()


class SupervisedPlotGenerator(PlotGenerator):
    def __init__(
        self,
        spec,
        n_trials,
        epsilons,
        data_fracs,
        datagen_method,
        perf_eval_fn,
        results_dir,
        n_workers,
        constraint_eval_fns=[],
        perf_eval_kwargs={},
        constraint_eval_kwargs={},
        batch_epoch_dict={},
        n_downstreams=1,
    ):
        """Class for running supervised Seldonian experiments
                and generating the three plots

        :param spec: Specification object for running the
                Seldonian algorithm
        :type spec: seldonian.spec.Spec object

        :param n_trials: The number of times the
                Seldonian algorithm is run for each data fraction.
                Used for generating error bars
        :type n_trials: int

        :param epsilons: error allowed for the constraints
                (the horizontal axis on the three plots).
        :type epsilons: List(float)

        :param datagen_method: Method for generating data that is used
                to run the Seldonian algorithm for each trial
        :type datagen_method: str, e.g. "resample"

        :param perf_eval_fn: Function used to evaluate the performance
                of the model obtained in each trial, with signature:
                func(theta,**kwargs), where theta is the solution
                from candidate selection
        :type perf_eval_fn: function or class method

        :param results_dir: The directory in which to save the results
        :type results_dir: str

        :param n_workers: The number of workers to use if
                using multiprocessing
        :type n_workers: int

        :param constraint_eval_fns: List of functions used to evaluate
                the constraints on ground truth. If an empty list is provided,
                the constraints are evaluated using the parse tree
        :type constraint_eval_fns: List(function or class method),
                defaults to []

        :param perf_eval_kwargs: Extra keyword arguments to pass to
                perf_eval_fn
        :type perf_eval_kwargs: dict

        :param constraint_eval_kwargs: Extra keyword arguments to pass to
                the constraint_eval_fns
        :type constraint_eval_kwargs: dict

        :param batch_epoch_dict: Instruct batch sizes and n_epochs
                for each data frac
        :type batch_epoch_dict: dict
        """

        super().__init__(
            spec=spec,
            n_trials=n_trials,
            n_downstreams=n_downstreams,
            epsilons=epsilons,
            data_fracs = data_fracs,
            datagen_method=datagen_method,
            perf_eval_fn=perf_eval_fn,
            results_dir=results_dir,
            n_workers=n_workers,
            constraint_eval_fns=constraint_eval_fns,
            perf_eval_kwargs=perf_eval_kwargs,
            constraint_eval_kwargs=constraint_eval_kwargs,
            batch_epoch_dict=batch_epoch_dict,
        )
        self.regime = "supervised_learning"
    
    def run_seldonian_experiment(self, verbose=False, model_name="qsa", validation=True, dataset_name=None, logfilename=None):
        """Run a supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("generating resampled datasets")
            if dataset_name == 'adults':
                # if validation:
                generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Adults", self.spec.frac_data_in_safety)
                # else:
                #     generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Adult_test", self.spec.frac_data_in_safety)
            if dataset_name == 'health':
                generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/health", self.spec.frac_data_in_safety)
            if dataset_name == 'income':
                generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/income", self.spec.frac_data_in_safety)
            elif dataset_name == 'Face':
                if validation:
                    generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Face", self.spec.frac_data_in_safety)
                else:
                    generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Face", self.spec.frac_data_in_safety)
            else:
                generate_resampled_datasets(dataset, self.n_trials, self.results_dir)
            print("Done generating resampled datasets")
            print()

        run_seldonian_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_downstreams=self.n_downstreams,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
            validation=validation,
            dataset_name=dataset_name,
            logfilename=logfilename,
        )

        ## Run experiment
        sd_exp = SeldonianExperiment(model_name=model_name, results_dir=self.results_dir)

        sd_exp.run_experiment(**run_seldonian_kwargs)
        return
    
    def run_alternative_seldonian_experiment(self, model_name, spec, batch_epoch_dict=None, verbose=False):
        """Run a supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """
        assert(model_name in seldonian_model_set)
        dataset = spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("generating resampled datasets")
            generate_resampled_datasets(dataset, self.n_trials, self.results_dir)
            print("Done generating resampled datasets")
            print()

        run_seldonian_kwargs = dict(
            spec=spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=batch_epoch_dict or self.batch_epoch_dict,
            verbose=verbose,
        )

        ## Run experiment
        sd_exp = SeldonianExperiment(model_name=model_name, results_dir=self.results_dir)

        sd_exp.run_experiment(**run_seldonian_kwargs)
        return


    def run_baseline_experiment(self, model_name, verbose=False, validation=True, dataset_name=None):
        """Run a supervised Seldonian experiment using the spec attribute
        assigned to the class in __init__().

        :param model_name: The name of the baseline model to use

        :type model_name: str

        :param verbose: Whether to display results to stdout
                while the Seldonian algorithms are running in each trial
        :type verbose: bool, defaults to False
        """

        dataset = self.spec.dataset

        if self.datagen_method == "resample":
            # Generate n_trials resampled datasets of full length
            # These will be cropped to data_frac fractional size
            print("checking for resampled datasets")
            if dataset_name == 'adults':
                # if validation:
                generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Adults", self.spec.frac_data_in_safety)
                # else:
                #     generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Adult_test", self.spec.frac_data_in_safety)
            if dataset_name == 'health':
                generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/health", self.spec.frac_data_in_safety)
            if dataset_name == 'income':
                generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/income", self.spec.frac_data_in_safety)
            elif dataset_name == 'Face':
                if validation:
                    generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Face", self.spec.frac_data_in_safety)
                else:
                    generate_resampled_datasets(dataset, self.n_trials, "./SeldonianExperimentResults/Face", self.spec.frac_data_in_safety)
            print("Done checking for resampled datasets")
            print()

        run_baseline_kwargs = dict(
            spec=self.spec,
            data_fracs=self.data_fracs,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
            datagen_method=self.datagen_method,
            perf_eval_fn=self.perf_eval_fn,
            perf_eval_kwargs=self.perf_eval_kwargs,
            constraint_eval_fns=self.constraint_eval_fns,
            constraint_eval_kwargs=self.constraint_eval_kwargs,
            batch_epoch_dict=self.batch_epoch_dict,
            verbose=verbose,
            n_downstreams=self.n_downstreams,
            validation=validation,
            dataset_name=dataset_name,
        )

        ## Run experiment
        bl_exp = BaselineExperiment(model_name=model_name, results_dir=self.results_dir)

        bl_exp.run_experiment(**run_baseline_kwargs)
        return
