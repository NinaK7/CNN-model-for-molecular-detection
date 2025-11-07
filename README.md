# CNN-model-for-molecular-detection

This repository contains scripts to use the CNN-model from the article "Identification of molecular line emission using
Convolutional Neural Networks" (Kessler et al. 2025).

# Dependencies

You will need to have [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed on your machine, see
their website on how to install it.

# Framework

Ones need to install the framework CIANNA (Cornu 2024) which can be found at: https://github.com/Deyht/CIANNA.

This can be done by git-cloning the current repository and by running :

```shell
uv sync -U
```

which will create a virtual environment where the framework can then be loaded in python with :

```shell
uv run python
>>> import CIANNA
```

# Downloading the CNN-model and datasets

Ones need to download the model from zenodo available at https://zenodo.org/records/16899524

To fetch the data and extract them in the correct place, you can run:
```shell
uvx zenodo_get 16899524 -o data/

unzip -n data/model.zip -d data
unzip -n data/test_dataset.zip -d data
unzip -n data/spectrum.zip -d data
```

In the later Zenodo repository, one can find :

- The model and the mask needed for inference.

- The test dataset used in the article can be found in the folder "test_dataset/", which is divided into the subsets
  "recipe/" and "unconstrained/", each one containing the spectra ("data_test.npy") and the associated labels
  ("target_test.npy").

- The file "spectrum/hot_core_spectrum.npy" is an example of synthetic spectrum where all the molecules are detectable.

# Using the CNN-model

The [CNN-model](data/model/net0_s0099.dat) to be used is stored as "net0_s0099.dat" in the folder "data/model/".
To infer the model, one needs to take into account the ["data/model/mask.npy"](data/model/mask.npy) file that contains a mask that should be
multiplied by the spectrum on which the prediction is made.

The script "pred_AVG.py" presents the steps to follow to obtain a prediction on the averaged model.
Whereas, the script "pred_MC_dropout.py" shows how to produce results through a Monte Carlo drop out approach.

Both can be run this way:

```shell
uv run python pred_AVG.py
```

```shell
uv run python pred_MC_dropout.py
```

The output is an array which dimension is the number of spectra as a function of the number of molecules to be detected. The order of molecules is the one defined in the variable "molecules". For each molecule of each spectrum, the CNN-model gives a model score. 

The predictions are stored in the "fwd_res/" folder under the file "net0_0099.dat", which is replaced at each inference. 
We added in the scripts an extra line to save the prediction in two news files : "./fwd_res/pred_AVG_test_dataset.npy" and "./fwd_res/pred_MC_dropout_spectrum.npy" respectively.

# Calibrating the model score into a detection probability

The test dataset can be used to calibrate the model score into a detection probability by using the script "model_score_calibration.py". This script computes the statistical probability for a prediction to actually be a true detection by using the test dataset. In this example, the model scores obtained in "./fwd_res/pred_MC_dropout_spectrum.npy" are converted into probabilities. More details can be found in the script at commented lines.

This script also includes the computation of the median absolute deviation (MAD) which is used to estimate the uncertainty on a MC drop out prediction.

If you have any issue or question regarding the use of the model, please contact : nina.kessler@u-bordeaux.fr.
