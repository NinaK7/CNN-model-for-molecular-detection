# CNN-model-for-molecular-detection

This repository contains elements to use the CNN-model from the article "Identification of molecular line emission using Convolutional Neural Networks" (Kessler et al. sub). Ones needs to install and use the framework CIANNA (Cornu 2025) which can be found at : https://github.com/Deyht/CIANNA.

The CNN-model to be used is stored in the folder "net_save/" as well as the values of the loss computed on the validation dataset ("error.txt") obtained during CNN training. 

To infer the model, one needs to take into account the "mask.npy" file that contains a mask that should be multipled by the spectrum on which the prediction is made. 

The script "pred_AVG.py" presents the steps to follow to obtain a prediction on the averaged model. Whereas, the script "pred_MC_dropout.py" shows how to produce results through a Monte Carlo drop out approach. 

A test dataset can be found at : https://zenodo.org/records/16899524?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ5NzgyZDhkLWE5NjgtNGQyYS1iZjgzLWFjMDEzZmEzNDBhNiIsImRhdGEiOnt9LCJyYW5kb20iOiJlZDczYzFjOTZkYWFhNzU2MWRmMTVhNWVjOGU0OWY4OCJ9.Zw0pqst9z0m0VvokEFTQI0t6x-Qjj2Q7Do1vfiVmfYwEA_i4DeD5Vyn5rXYCpV1lexe17tJ-JJWDIjQlOJYDVg
