
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'CIANNA/src/build/lib.linux-x86_64-cpython-311')
import CIANNA as cnn

############################################################################
##              Data reading (your mileage may vary)
############################################################################

def i_ar(int_list):
	return np.array(int_list, dtype="int")

def f_ar(float_list):
	return np.array(float_list, dtype="float32")

def transfo(data_set, target) : #normalization
	shape = data_set.shape
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

molecules = ['aGg\'-(CH2OH)2','C2H3CN','C2H5CN','C2H5OH','C3H7CN','CH3CCH','CH3CHO','CH3CN','CH3COCH3', 
			'CH3NH2', 'CH3OCH3','CH3OCHO','CH3OH','CH2NH','gGg\'-(CH2OH)2','HCCCN', 'HC(O)NH2','t-HCOOH', 
			'H2CS', 'NH2CN']

nb_mol = len(molecules)

nb_test = 20000
channels = 35000
data = np.zeros((nb_test, channels))
target = np.zeros((nb_test, nb_mol))

ratio_set = int(nb_test * 1/2)

path_test = 'TEST_DATASET/TEST.recipe/'
data[:ratio_set] = np.load(path_test + "data_test.npy")[:ratio_set]
target[:ratio_set] = np.load(path_test + "target_test.npy")[:ratio_set]

path_test = 'TEST_DATASET/TEST.unconstrained/'
data[ratio_set:] = np.load(path_test + "data_test.npy")#[ratio_set:]
target[ratio_set:] = np.load(path_test + "target_test.npy")#[ratio_set:]

data, target =  transfo(data, target)
    

############################################################################
##               CIANNA network construction and use
############################################################################

cnn.init(in_dim=i_ar([channels]), in_nb_ch=3, out_dim=nb_mol, \
		bias=0.1, b_size=32, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A")

cnn.create_dataset("TEST", size=i_ar(nb_test), input=f_ar(data), target=f_ar(target*0))

path_cnn = './net_save/'
load_epoch = 99

cnn.load(path + "net0_s%04d.dat"%load_epoch,load_epoch, bin=1)
cnn.forward(drop_mode='AVG_MODEL', no_error=1, saving=2)
pred = np.fromfile("fwd_res/net0_%04d.dat"%load_epoch, dtype='float32')
pred = np.reshape(pred,(nb_test,nb_mol+1))






