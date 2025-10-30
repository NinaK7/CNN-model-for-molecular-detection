# CNN-model-for-molecular-detection

This repository contains scripts to use the CNN-model from the article "Identification of molecular line emission using
Convolutional Neural Networks" (Kessler et al. 2025).

# Framework

Ones need to install the framework CIANNA (Cornu 2024) which can be found at: https://github.com/Deyht/CIANNA.

This can be done by git-cloning the current repository and by running :
```shell
uv sync -U
uv run pytest
```
which will create a 'cianna-demo' virtual environment. The framework can then be loaded in python with :

```shell
import CIANNA
```

# CNN-model

Ones needs to download the CNN-model from the
repository : https://zenodo.org/records/16899524?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ5NzgyZDhkLWE5NjgtNGQyYS1iZjgzLWFjMDEzZmEzNDBhNiIsImRhdGEiOnt9LCJyYW5kb20iOiJlZDczYzFjOTZkYWFhNzU2MWRmMTVhNWVjOGU0OWY4OCJ9.Zw0pqst9z0m0VvokEFTQI0t6x-Qjj2Q7Do1vfiVmfYwEA_i4DeD5Vyn5rXYCpV1lexe17tJ-JJWDIjQlOJYDVg

The CNN-model to be used is stored as "net0_s0099.dat" in the folder "model/".
To infer the model, one needs to take into account the "model/mask.npy" file that contains a mask that should be
multiplied by the spectrum on which the prediction is made.

# Datasets

In the later Zenodo repository, one can also find :

- The test dataset used in the article can be found in the folder "test_dataset/", which is divided into the subsets
  "recipe/" and "unconstrained/", each one containing the spectra ("data_test.npy") and the associated labels
  ("target_test.npy").

- The file "spectrum/hot_core_spectrum.npy" is an example of synthetic spectrum where all the molecules are detectable.

# Using the CNN-model

The script "pred_AVG.py" presents the steps to follow to obtain a prediction on the averaged model.
Whereas, the script "pred_MC_dropout.py" shows how to produce results through a Monte Carlo drop out approach.

Both can be run this way:

```shell
uv run python pred_AVG.py
uv run python pred_MC_dropout.py
```
