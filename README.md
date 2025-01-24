# Learning Fair Representations with High-confidence Guarantees (FRG)

This repository contains the official implementation of FRG as described in the paper: Learning Fair Representations with High-confidence Guarantees (In submission).

## Requirements
We have tested our code on `Python 3.9` with `PyTorch 1.12.0`, and `CUDA 11.4`. Please follow the following steps to create a virtual environment and install the required packages. Please refer to `environment.yml` for more details.


Create a virtual environment:
```
conda env create -f environment.yml
conda activate seld
```

## Instructions on Acquiring Datasets
We include the processed Adult dataset in the repository in the `./adults_vfae/` directory.

The health dataset is downloaded from this [repo](https://github.com/ermongroup/lag-fairness/blob/master/health.csv). We provide a script to process the data.
```bash
python ./health/health_gender.py
```

The income dataset can be downloaded by runing the script below.
```bash
python ./income/income_data.py
```


<!-- Since the UTK-Face dataset is larger, we cannot include the processed data in the repository. You may download the data from [Kaggle](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv?resource=download) to the `./face_recog/` directory.
We provide a script to process the data.
```bash
python face_recog_data.py
``` -->

## Training Commands

To train **FRG**, we first have to create a config file that sets the hyperparameters such as the learning rate. In the `config` directory, we provide the configs for reproducing our main experiments.

#### Examples:

* To unsupervisedly train **FRG** on the Adults dataset with epsilon=0.04:
```bash
python frg_spec.py --epsilon 0.04 --delta 0.1 --dataset adults
python frg_gen_plots.py --epsilon 0.04 --delta 0.1 --dataset adults --config config/adults_0.04_unsupervised.json
```

* To supervisedly train **FRG** on the Health dataset with epsilon=0.12:
```bash
python frg_spec.py --epsilon 0.12 --delta 0.1 --dataset health
python frg_gen_plots.py --epsilon 0.12 --delta 0.1 --dataset health --config config/health_0.12_supervised.json
```


* To train baselines with the Income dataset:
```bash
python frg_spec.py --epsilon 0.08 --delta 0.1 --dataset income
python frg_gen_plots.py --run_baselines --epsilon 0.08 --delta 0.1 --dataset income
```

<!-- * To train **FRG** with practical adjustment the UTK-Face dataset:
```bash
python facial_recog_frg_i1.py
python facial_recog_gen_plots.py --version 1 --epsilon 0.08 --psi 1.18
```
To train baselines with the Adult dataset:
```bash
python facial_recog_gen_plots.py --version 0 --epsilon 0.08 --psi 1.18
``` -->

* To train **FRG_MI** with high-confidence fairness guarantees using the mutual information bound (Appendix) the Adult dataset:
```bash
python frg_mi_spec.py
python frg_mi_gen_plots.py --version 1 --epsilon 0.20 --psi 0.0044
```

## Usage Summary
```
usage: Interface for FRG
  --run_baselines A boolean. If set to true, the baselines will be run. Otherwise, FRG will be run.
  --device cuda device
  --validation A boolean. If set to true, a validation set will be created from the training dataset. This is used for hyperparamter tuning.
  --verbose verbose
  --epsilon epsilon: the desired error threshold as described in the paper
  --delta delta: the desired confidence level as described in the paper
  --dataset The dataset used for training e.g., adults, health, income.
  --config a config of the hyperparameters for FRG
  <!-- --psi  the desired upper bound on the mutual information \Tilde{I}_1(Z;S) as describe in the paper -->
```

## Acknowledgement
This is adapted from two source code repositories, [Seldonian Engine](https://github.com/seldonian-toolkit/Engine/) and [Seldonian Experiments](https://github.com/seldonian-toolkit/Experiments). They are libraries that implement [Seldonian](https://seldonian.cs.umass.edu/) algorithms. We thank the authors for sharing their code.
