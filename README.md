# CNN-model-for-molecular-detection

This repository contains elements to use the CNN-model from the article "Identification of molecular line emission using Convolutional Neural Networks" (Kessler et al. sub). Ones needs to install and use the framework CIANNA (Cornu 2025) which can be found at : https://github.com/Deyht/CIANNA.

The CNN-model to be used is stored in the folder "net_save/" as well as the values of the loss computed on the validation dataset ("error.txt") obtained during CNN training. 

To infer the model, one needs to take into account the "mask.npy" file that contains a mask that should be multipled by the spectrum on which the prediction is made. 

The script "pred_AVG.py" presents the steps to follow to obtain a prediction on the averaged model. Whereas, the script "pred_MC_dropout.py" shows how to produce results through a Monte Carlo drop out approach. 

The file "hot_core_spectrum.npy" is an exemple of synthetic spectrum where all the molecules are detectable. 

A test dataset can be found at : url to come 
