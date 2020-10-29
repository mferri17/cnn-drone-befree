################################
### IMPORTS
######

### KNOWN IMPORTS

import math
import os
import time
import errno
import random
import sys
import gc
import socket
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # https://stackoverflow.com/a/64448353/10866825

# %tensorflow_version 1.x
import tensorflow as tf

# when importing keras, please notice:
#   https://stackoverflow.com/a/57298275/10866825
#   https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# !pip install resnet
# import resnet as kr

import pickle
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.metrics

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# %matplotlib inline

# # -- for GradCAM
# # !pip install tf-keras-vis
# from tf_keras_vis.utils import normalize
# from tf_keras_vis.gradcam import Gradcam
# from matplotlib import cm


### PATHS

# this file goes into the 'PROJECT_ROOT/src v2/' folder to work properly with the following paths
this_folder = os.path.dirname(os.path.realpath(__file__))
lib_folder = os.path.join(this_folder, './lib/')
original_models_folder = os.path.join(this_folder, './../dev-models/_originals/') # Dario's original trained models (https://drive.switch.ch/index.php/s/Idsyf8WIwQpvRMF)
original_datasets_folder = os.path.join(this_folder, './../dev-datasets/_originals/') # Dario's original dataset (https://drive.switch.ch/index.php/s/8clDQNH645ZjWDD)
backgrounds_folder = os.path.join(this_folder, './../dev-datasets/_backgrounds/')
new_models_folder = os.path.join(this_folder, './../dev-models/')
new_datasets_folder = os.path.join(this_folder, './../dev-datasets/')
visualization_folder = os.path.join(this_folder, './../dev-visualization/')

dario_model_path = original_models_folder + 'v1_model_train_size_50000_rep_1.h5'
dario_train_path = original_datasets_folder + 'dario/v1_train.pickle'
dario_test_path = original_datasets_folder + 'dario/v1_test.pickle'


### CUSTOM IMPORTS

# see https://stackoverflow.com/a/59703673/10866825 and https://stackoverflow.com/a/63523670/10866825
sys.path.append('.')
from functions import general_utils
from functions import network_utils


################################
### GLOBAL VARIABLES
######


################################
### FUNCTIONS
######

def train_with_generator(data_folder, data_size = None,
                         regression = True, classification = False, 
                         retrain_from = None, batch_size = 64, epochs = 20,
                         augmentation = False, backgrounds_folder = None, backgrounds_name = '',
                         view_stats = False, save_stats = True, save_model = True, save_folder = ''):
    
    list_files = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
    list_files = list_files[:data_size]

    with open(list_files[0], 'br') as first:
        input_shape = pickle.load(first)['image'].shape

    timestr = time.strftime("%Y%m%d_%H%M%S")
    model_name = '{} {} - {}{}_size{}_{}{}_ep{}'.format(
        timestr,
        socket.gethostname(),
        'regr' if regression else '', 
        'class' if classification else '', 
        len(list_files),
        'retrainfrom{}'.format(retrain_from) if retrain_from else 'notrain',
        '_augm_{}'.format(backgrounds_name) if augmentation else '',
        epochs
    )

    replace_imgs = general_utils.load_images_from_folder(backgrounds_folder) if augmentation and backgrounds_folder is not None else None
    model = network_utils.network_create(dario_model_path, input_shape, regression, classification, retrain_from, view_summary=False)
    model, history = network_utils.network_train_generator(model, list_files, regression, classification, augmentation, replace_imgs, batch_size, epochs)
    network_utils.network_stats(history, regression, classification, view_stats, save_stats, save_folder, model_name)

    if save_model:
        network_utils.network_save(model, save_folder, model_name)



################################
### MAIN
######

def main():

    data_folder = 'C:/Users/96mar/Desktop/orig_train 63720/'
    
    data_size = 256
    batch_size = 32
    regression, classification = True, False
    retrain_from = 24
    epochs = 2
    augmentation = True
    backgrounds_folder = new_datasets_folder + '_backgrounds/backgrounds-20'
    save_folder = os.path.join(new_models_folder, 'training_generator')

    train_with_generator(data_folder, data_size, regression, classification,
                         retrain_from, batch_size, epochs,
                         augmentation, backgrounds_folder, 'bg20',
                         view_stats=False, save_stats=True, save_model=True, save_folder=save_folder)



if __name__ == "__main__":
#   tf.config.experimental.list_physical_devices('GPU')
#   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
  main()
