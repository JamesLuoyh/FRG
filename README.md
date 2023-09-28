# Learning Fair Representations with High-confidence Guarantees (FRG)

This repository contains the official implementation of FRG as described in the paper: Learning Fair Representations with High-confidence Guarantees (submitted to ICLR 2024).

## Requirements
We have tested our code on `Python 3.9` with `PyTorch 1.12.0`, and `CUDA 11.4`. Please follow the following steps to create a virtual environment and install the required packages. Please refer to `environment.yml` for more details.


Create a virtual environment:
```
conda env create -f environment.yml
conda activate seld
```

## Instructions on Acquiring Datasets
We include the processed Adult dataset in the repository in the `./adults_vfae/` directory.

Since the UTK-Face dataset is larger, we cannot include the processed data in the repository. You may download the data from [Kaggle](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv?resource=download) to the `./face_recog/` directory.
We provide a script to process the data.
```bash
python face_recog_data.py
```

## Training Commands

#### Examples:

* To train **FRG** with high-confidence fairness guarantees the Adult dataset:
```bash
python vfae_i1_theoretical.py
python vfae_gen_plots.py --version 1 --epsilon 0.20 --psi 0.0044
```
* To train **FRG** with practical adjustment the Adult dataset:
```bash
python vfae_i1.py
python vfae_gen_plots.py --version 1 --epsilon 0.08 --psi 0.32
```
To train baselines with the Adult dataset:
```bash
python vfae_gen_plots.py --version 0 --epsilon 0.08 --psi 0.32
```

* To train **FRG** with practical adjustment the UTK-Face dataset:
```bash
python facial_recog_frg_i1.py
python facial_recog_gen_plots.py --version 1 --epsilon 0.08 --psi 1.18
```
To train baselines with the Adult dataset:
```bash
python facial_recog_gen_plots.py --version 0 --epsilon 0.08 --psi 1.18
```

## Usage Summary
```
usage: Interface for FRG
  --version Version 0 runs the baselines. Version 1 runs FRG.
  --device cuda device
  --validation create validation set from the training dataset. used for hyperparamter tuning
  --verbose verbose
  --epsilon epsilon as describe in the paper
  --psi  the desired upper bound on the mutual information \Tilde{I}_1(Z;S) as describe in the paper
```

## Acknowledgement
This is adapted from two source code repositories, [Seldonian Engine](https://github.com/seldonian-toolkit/Engine/) and [Seldonian Experiments](https://github.com/seldonian-toolkit/Experiments). They are libraries that implement [Seldonian](https://seldonian.cs.umass.edu/) algorithms. We thank the authors for sharing their code.
