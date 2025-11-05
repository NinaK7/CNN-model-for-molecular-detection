from pathlib import Path

import CIANNA as cnn
import numpy as np
from loguru import logger
from numba import jit, prange


############################################################################
##              Data reading and normalization
############################################################################


def i_ar(int_list):  # function to conversion numbers to integers
    return np.array(int_list, dtype="int")


def f_ar(float_list):  # function to conversion numbers to float32
    return np.array(float_list, dtype="float32")


@jit(nogil=True, fastmath=True, cache=True, parallel=True)
def transfo(data_set):  # normalization of the data according to three transformations
    shape = data_set.shape
    data_3chan = np.zeros((shape[0], 3 * shape[1]))

    alpha = 1.5

    data_norm = np.copy(data_set)
    data_tanh = np.copy(data_set) * 0
    for i in prange(shape[0]):
        data_norm[i] = data_norm[i] / np.max(data_norm[i])
        data_tanh[i] = np.tanh(alpha * data_norm[i]) / np.max(
            np.tanh(alpha * data_norm[i])
        )

    # linear normalization :
    data_3chan[:, : shape[1]] = data_norm
    # tanh normalization :
    data_3chan[:, shape[1] : 2 * shape[1]] = data_tanh
    # polynomial normalization
    data_3chan[:, 2 * shape[1] :] = data_norm**3

    return data_3chan


# Path to the directories containing the dataset and the model.
data_path = Path("data")
dataset_path = data_path / "test_dataset"
model_path = data_path / "model"

# list of molecules to be detected, the results will be given following to this classification
molecules = [
    "aGg'-(CH2OH)2",
    "C2H3CN",
    "C2H5CN",
    "C2H5OH",
    "C3H7CN",
    "CH3CCH",
    "CH3CHO",
    "CH3CN",
    "CH3COCH3",
    "CH3NH2",
    "CH3OCH3",
    "CH3OCHO",
    "CH3OH",
    "CH2NH",
    "gGg'-(CH2OH)2",
    "HCCCN",
    "HC(O)NH2",
    "t-HCOOH",
    "H2CS",
    "NH2CN",
]

nb_mol = len(molecules)

nb_test = 20000  # number of spectra to be tested
channels = 35000  # number of channels within each spectrum

data = np.zeros((nb_test, channels))
target = np.zeros((nb_test, nb_mol))
ratio_set = int(nb_test * 1 / 2)  # 50% recipe and 50% unconstrained

logger.info("Loading data")
# loading of the recipe test dataset and the corresponding target
recipe_path = dataset_path / "recipe"

logger.debug("recipe-data_test")
data[:ratio_set] = np.load(recipe_path / "data_test.npy")[:ratio_set]
logger.debug("recipe-target_test")
target[:ratio_set] = np.load(recipe_path / "target_test.npy")[:ratio_set]

# loading of the unconstrained test dataset and the corresponding target
unconstrained_path = dataset_path / "unconstrained"

logger.debug("unconstrained-data_test")
data[ratio_set:] = np.load(unconstrained_path / "data_test.npy")  # [ratio_set:]
logger.debug("unconstrained-target_test")
target[ratio_set:] = np.load(unconstrained_path / "target_test.npy")  # [ratio_set:]

logger.debug("mask")
mask = np.load(model_path / "mask.npy")[:-1]  # loading of the mask
data = transfo(
    data * mask
)  # normalization of the full test dataset multiplied by the mask

############################################################################
##               CIANNA network construction and use
############################################################################
logger.info("CIANNA: `init`")
cnn.init(
    in_dim=i_ar([channels]),
    in_nb_ch=3,
    out_dim=nb_mol,
    bias=0.1,
    b_size=32,
    comp_meth="C_BLAS",
    dynamic_load=1,
    mixed_precision="FP32C_FP32A",
)  # initialization of the CNN backbone

logger.info("CIANNA: `create_dataset`")
cnn.create_dataset(
    "TEST", size=i_ar(nb_test), input=f_ar(data), target=f_ar(target * 0)
)  # loading of the test dataset in CIANNA

load_iteration = 99  # model from the 99th iteration

logger.info("CIANNA: `load`")
cnn.load(
    (model_path / f"net0_s{load_iteration:04d}.dat").as_posix(), load_iteration, bin=1
)  # CIANNA loads the CNN-model with all the weights

logger.info("CIANNA: `forward`")
cnn.forward(drop_mode="AVG_MODEL", no_error=1, saving=2)  # Forward propagation

pred = np.fromfile(
    f"fwd_res/net0_{load_iteration:04d}.dat", dtype="float32"
)  # loading of the model scores given bien the CNN-model
pred = np.reshape(
    pred, (nb_test, nb_mol + 1)
)  # reshaping of the model score to the number of classes

logger.info("Prediction:")
print(molecules, pred)
