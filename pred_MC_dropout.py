import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

import sys
sys.path.insert(0,'CIANNA/src/build/lib.linux-x86_64-cpython-311') #your path to CIANNA
import CIANNA as cnn

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def transfo(data_set, target) : # normalization
	shape = data_set.shape

    if len(shape)==1 : 
        data_3chan = np.zeros((1, 3*shape[0])) 
    else : 
    	data_3chan = np.zeros((shape[0], 3*shape[1])) 

	alpha = 1.5

	data_norm = np.copy(data_set)
	data_tanh = np.copy(data_set)*0
	for i in range(shape[0]) :
		data_norm[i] = data_norm[i]/np.max(data_norm[i])
		data_tanh[i] = np.tanh(alpha*data_norm[i])/np.max(np.tanh(alpha*data_norm[i]))

	# linear normalization :
	data_3chan[:, :shape[1]] = data_norm
	# tanh normalization :
	data_3chan[:, shape[1]:2*shape[1]] = data_tanh
	# polynomial normalization
	data_3chan[:, 2*shape[1]:] = data_norm**3

	return data_3chan, target

channels = 35000
freq = np.linspace(80, 115, channels)

molecules = ['aGg\'-(CH2OH)2','C2H3CN','C2H5CN','C2H5OH','C3H7CN','CH3CCH','CH3CHO','CH3CN','CH3COCH3', 
             'CH3NH2', 'CH3OCH3','CH3OCHO','CH3OH','CH2NH','gGg\'-(CH2OH)2','HCCCN', 'HC(O)NH2','t-HCOOH', 
             'H2CS', 'NH2CN']

nb_mol = len(molecules)

nb_data = 1
data = np.zeros((nb_data, channels))

mask_raw = np.load('./mask.npy')
mask = np.ones(channels) - mask_raw[:-1]
path_test = './SPECTRA2TEST/'
name = ['spec1', 'spec2']
data = np.nan_to_num(np.load(path_test + "spectrum_to_test.npy"))* mask

target = np.zeros((nb_data, nb_mol))
data_norm, target =  transfo(data, target)
nb_test = len(target)


nb_data_MC = 100
target_MC = np.zeros((nb_data_MC, nb_mol))
data_MC = np.tile(data_norm[spec], (nb_data_MC, 1))
nb_test = len(target_MC)

cnn.init(in_dim=i_ar([channels]), in_nb_ch=3, out_dim=nb_mol, bias=0.1, 
        b_size=32, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A", inference_only=1, no_logo=1)
cnn.create_dataset("TEST", size=i_ar(nb_data_MC), input=f_ar(data_MC), target=f_ar(target_MC))

path_cnn = './net_save/'
load_epoch = 99
cnn.load(path_cnn + "net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
cnn.forward(drop_mode='MC_MODEL',no_error = 1, repeat=1, saving=2, silent=1)

pred = np.fromfile("./fwd_res/net0_%04d.dat"%(load_epoch), dtype='float32')
pred = np.reshape(pred,(nb_data_MC,nb_mol+1))
print(pred)


