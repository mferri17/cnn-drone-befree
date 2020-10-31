################################
### SETTINGS
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
import glob
from datetime import datetime

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
## see https://stackoverflow.com/a/59703673/10866825 
## and https://stackoverflow.com/a/63523670/10866825

sys.path.append('.')
from functions import general_utils
from functions import network_utils



################################
### GLOBAL VARIABLES
######



################################
### FUNCTIONS
######

def train_with_generator(data_folder, network_weights_path, data_size = None,
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
    
    if network_weights_path is not None and os.path.exists(network_weights_path):
      with open(network_weights_path, 'rb') as fp:
        initial_weights = pickle.load(fp)
    else:
      initial_weights = None

    # replace_imgs_paths = general_utils.list_files_in_folder(backgrounds_folder, 'jpg') if augmentation and backgrounds_folder is not None else None
    replace_imgs_paths = general_utils.list_files_in_folder(backgrounds_folder, 'pickle') if augmentation and backgrounds_folder is not None else None

    model = network_utils.network_create(input_shape, regression, classification, initial_weights, retrain_from, view_summary=False)
    model, history = network_utils.network_train_generator(model, list_files, regression, classification, augmentation, replace_imgs_paths, batch_size, epochs)
    network_utils.network_stats(history, regression, classification, view_stats, save_stats, save_folder, model_name)

    if save_model:
        network_utils.network_save(model, save_folder, model_name)



################################
### MAIN
######

def get_args():
  import argparse

  def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

  def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)

  default_batch_size = 64
  default_epochs = 30
  debug_data_size = 64
  debug_batch_size = 4
  debug_epochs = 2

  parser = argparse.ArgumentParser(description='Train the network on the given dataset using a generator which performs dynamic loading and data augmentation.')
  parser.add_argument('data_folder', type=dir_path, help='path to the dataset') # required
  parser.add_argument('-r', '--regression', action='store_true', help='specify the argument if you want to perform regression')
  parser.add_argument('-c', '--classification', action='store_true', help='specify the argument if you want to perform classification')
  parser.add_argument('--data_size', type=int, default=None, metavar='DS', help='max number of samples in the dataset (default = entire dataset, debug = {})'.format(debug_data_size))
  parser.add_argument('--batch_size', type=int, default=default_batch_size, metavar='BS', help='training batch size (default = {}, debug = {})'.format(default_batch_size, debug_batch_size))
  parser.add_argument('--epochs', type=int, default=default_epochs, metavar='E', help='number of training epochs (default = {}, debug = {})'.format(default_epochs, debug_epochs))
  parser.add_argument('--weights_path', type=file_path, metavar='WP', help='path to the network initial weights dictionary {"layer_name": get_weights}') # required
  parser.add_argument('--retrain_from', type=int, default=None, metavar='RF', help='number of layer to retrain from (default = no training, 0 = complete training)')
  parser.add_argument('-a', '--augmentation', action='store_true', help='specify the argument if you want to perform data augmentation')
  parser.add_argument('--bgs_folder', type=dir_path, metavar='BGF', help='path to backgrounds folder for data augmentation')
  parser.add_argument('--bgs_name', type=str, metavar='BGN', help='name/identifier of the chosen backgrounds set')
  parser.add_argument('-s', '--save', action='store_true', help='specify the argument if you want to save the model and metrics')
  parser.add_argument('--save_folder', type=dir_path, metavar='SVF', help='path where to save the model and metrics')
  parser.add_argument('-d', '--debug', action='store_true', help='if the argument is specified, some parameters are set (overwritten) to debug values')

  parsed_args = parser.parse_args()
  if parsed_args.debug:
    parsed_args.data_size = debug_data_size
    parsed_args.batch_size = debug_batch_size
    parsed_args.epochs = debug_epochs

  return parsed_args



if __name__ == "__main__":
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')

  if args.debug:
    print("COUNT GPUs AVAILABLE: ", len(tf.config.experimental.list_physical_devices('GPU')), '\n')


  train_with_generator(
    args.data_folder, args.weights_path, args.data_size, args.regression, args.classification,
    args.retrain_from, args.batch_size, args.epochs,
    args.augmentation, args.bgs_folder, args.bgs_name,
    view_stats=False, save_stats=args.save, save_model=args.save, save_folder=args.save_folder
  )