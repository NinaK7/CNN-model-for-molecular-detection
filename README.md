# CNN-model-for-molecular-detection

This repository contains elements to use the CNN-model from the article "Identification of molecular line emission using Convolutional Neural Networks" (Kessler et al. sub). Ones needs to install and use the framework CIANNA (Cornu 2025) which can be found at : https://github.com/Deyht/CIANNA.

The CNN-model to be used is stored in the folder "net_save/" as well as the values of the loss computed on the validation dataset ("error.txt") obtained during CNN training. 

To infer the model, one needs to take into account the "mask.npy" file that contains a mask that should be multipled by the spectrum on which the prediction is made. 

The script "pred_AVG.py" presents the steps to follow to obtain a prediction on the averaged model. Whereas, the script "pred_MC_dropout.py" shows how to produce results through a Monte Carlo drop out approach. 

A test dataset can be found at : https://zenodo.org/records/16899524?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI4MTNmNTc0LTBkMjMtNDNiNi1hZDM5LTIwYWE5MmE3ZjJmOCIsImRhdGEiOnt9LCJyYW5kb20iOiIxZDZjMmM4MzJhNGI4OWI1ZDJlYTg4Yzg3OWFlZTA1ZCJ9.M2YiHRYfw8sQW8feuvvQvES5RjuJ9WjlkefkK3C_HOzhNnR-2yBfNSoDKRV-zqdGhAC7ShzhwTATuugP6p6YVQ
