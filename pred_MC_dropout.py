import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'CIANNA/src/build/lib.linux-x86_64-cpython-311') # your path to CIANNA version
import CIANNA as cnn

############################################################################
##              Data reading and normalizaton
############################################################################

def i_ar(int_list): # number to integer
	return np.array(int_list, dtype="int")

def f_ar(float_list): # number to float 32
	return np.array(float_list, dtype="float32")

def transfo(data_set) : # normalization
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

	return data_3chan

channels = 35000

# list of molecules that are detected and the classification follows this order
molecules = ['aGg\'-(CH2OH)2','C2H3CN','C2H5CN','C2H5OH','C3H7CN','CH3CCH','CH3CHO','CH3CN','CH3COCH3', 
             'CH3NH2', 'CH3OCH3','CH3OCHO','CH3OH','CH2NH','gGg\'-(CH2OH)2','HCCCN', 'HC(O)NH2','t-HCOOH', 
             'H2CS', 'NH2CN']
nb_mol = len(molecules)

# loading of the spectrum to be tested
path_test = './SPECTRA2TEST/'
nb_data = 1
data = np.nan_to_num(np.load(path_test + "classical_hot_core_spectrum.npy")) # needs to be reshaped to (1, 35000) if it is not yet the case
target = np.zeros((nb_data, nb_mol)) # targets to zero

mask = np.load('./mask.npy')[:-1] # mask loading
data_norm, target =  transfo(data*mask) # normalization of the spectrum multiplied by the mask

nb_data_MC = 100 # number of realizations to be done
target_MC = np.zeros((nb_data_MC, nb_mol)) # targets to zero
data_MC = np.tile(data_norm, (nb_data_MC, 1)) # production of an array with nb_data_MC times the same normalized spectrum

############################################################################
##               CIANNA network construction and use
############################################################################

# initialisation of the backbone
cnn.init(in_dim=i_ar([channels]), in_nb_ch=3, out_dim=nb_mol, bias=0.1, 
        b_size=32, comp_meth="C_CUDA", dynamic_load=1, mixed_precision="FP32C_FP32A", inference_only=1, no_logo=1)

# loading of the data to test by CIANNA
cnn.create_dataset("TEST", size=i_ar(nb_data_MC), input=f_ar(data_MC), target=f_ar(target_MC))

path_cnn = './net_save/' # path to the CNN-model weights
load_iteration = 99 # iteration to be loaded

cnn.load(path_cnn + "net0_s%04d.dat"%load_iteration, load_iteration, bin=1) # weights loading
cnn.forward(drop_mode='MC_MODEL',no_error = 1, repeat=1, saving=2, silent=1) # Forward propagation 

pred = np.fromfile("./fwd_res/net0_%04d.dat"%(load_iteration), dtype='float32') # loading of the prediction
pred = np.reshape(pred,(nb_data_MC,nb_mol+1)) # reshaping of the prediction according to the classes

