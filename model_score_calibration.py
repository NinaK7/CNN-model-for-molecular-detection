import numpy as np

# Used to estimate the errors on a prediction 
def MAD(data) :
    median_value = np.median(data)
    residuals = data - median_value
    
    negative_residuals_data = data[residuals < 0]
    positive_residuals_data = data[residuals > 0]
    
    negative_MAD = np.median(np.abs(negative_residuals_data - median_value))
    positive_MAD = np.median(np.abs(positive_residuals_data - median_value))

    if np.round(100*negative_MAD,1) == 0.0 :
        negative_MAD = 0.1/100

    if np.round(100*positive_MAD,1) == 0.0 :
        positive_MAD = 0.1/100
    
    return np.nan_to_num(negative_MAD), np.nan_to_num(positive_MAD)
    

molecules = ['aGg\'-(CH2OH)2','C2H3CN','C2H5CN','C2H5OH','C3H7CN','CH3CCH','CH3CHO','CH3CN','CH3COCH3', 
             'CH3NH2', 'CH3OCH3','CH3OCHO','CH3OH','CH2NH','gGg\'-(CH2OH)2','HCCCN', 'HC(O)NH2','t-HCOOH', 
             'H2CS', 'NH2CN']

nb_mol = len(molecules)

# Calibrating the model score obtained for the "hot_core_spectrum" by using the test dataset.
# To do this, one needs to first run "pred_AVG.py" and "pred_MC_dropout.py" 
# to use the predictions stored in "fwd/pred_AVG_test_dataset.npy" and "fwd/pred_MC_dropout_spectrum.npy" respectively.

# Loading the targets and the predictions for the test dataset, i.e. the dataset used for calibration.
nb_test = 20000
target_test_set = np.zeros((nb_test, nb_mol))
ratio_set = int(nb_test * 1 / 2)
target_test_set[:ratio_set] = np.load("data/test_dataset/recipe/target_test.npy")[:ratio_set]
target_test_set[ratio_set:] = np.load("data/test_dataset/unconstrained/target_test.npy") 

prediction_test_set = np.load('fwd_res/pred_AVG_test_dataset.npy')

# Loading the model score from the "hot_core_spectrum"
prediction_spectrum = np.load('fwd_res/pred_MC_dropout_spectrum.npy')
nb_data_MC = 100

# Sampling the [0,1] interval
bin_sampling = 100
bin_step = 1.0/bin_sampling

# One defines an interval centered on the model score value (for each molecule) obtained for the hot core spectrum. 
# Then, by using the test dataset, one computes the number of true detections out of the number of spectra for the considered interval.
# Thus, the detection probability is a direct result of the statistics from the test dataset on the model score interval.
# This calibration was used to obtain the results from the paper.

detection_probability = np.zeros((nb_mol, nb_data_MC))

for m in range(nb_mol) :
    for i in range(nb_data_MC) :
        molecule_model_score = prediction_spectrum.T[m][i]
    
        # Look for the spectra within the test dataset that correspond to the model score interval
        found_bin = np.where((prediction_test_set.T[m] >= molecule_model_score - bin_step/2) &
                             (prediction_test_set.T[m] < molecule_model_score + bin_step/2))[0]
    
        # Same for the true detections 
        true_detections = np.where((prediction_test_set.T[m] >= molecule_model_score - bin_step/2) & 
                                   (prediction_test_set.T[m] < molecule_model_score + bin_step/2) & 
                                   (target_test_set.T[m]==1))[0]
        
        # The detection probability for the hot core spectrum is then the statistical probability computed on the test dataset
        detection_probability[m][i] = len(true_detections)/len(found_bin)


    # Computation of the median absolute deviation (MAD)
    MAD_superior, MAD_inferior = MAD(detection_probability[m])

    print(molecules[m], ': ', str(np.round(100*np.median(detection_probability[m]),1)) + str(' $^{+%s}_{-%s}$ '%(np.round(100*MAD_superior,1), np.round(100*MAD_inferior,1))) + '%')

