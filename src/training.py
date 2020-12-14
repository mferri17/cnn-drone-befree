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

# When importing Tensorflow and Keras, please notice:
#   https://stackoverflow.com/a/57298275/10866825
#   https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable Tensorflow warnings https://stackoverflow.com/a/64448353/10866825

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# # -- for GradCAM
# from tf_keras_vis.utils import normalize
# from tf_keras_vis.gradcam import Gradcam
# from matplotlib import cm


### CUSTOM IMPORTS
## see https://stackoverflow.com/a/59703673/10866825 
## and https://stackoverflow.com/a/63523670/10866825 (for VS Code)

sys.path.append('.')
from functions import general_utils
from functions import network_utils



################################
### GLOBAL VARIABLES
######



################################
### FUNCTIONS
######

def train_with_generator(data_folder, network_weights_path, data_len,
                         regression, classification,
                         retrain_from, verbose, batch_size, epochs, oversampling,
                         use_lr_reducer, use_early_stop, 
                         use_profiler, profile_dir,
                         backgrounds_folder, backgrounds_len, backgrounds_name,
                         augmentation_prob, noise_folder,
                         view_stats, save_stats, save_model, save_folder):
    
    list_files = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
    list_files = list_files[:data_len]

    if network_weights_path is not None and os.path.exists(network_weights_path):
      with open(network_weights_path, 'rb') as fp:
        initial_weights = pickle.load(fp)
    else:
      initial_weights = None

    replace_imgs_paths = []
    if backgrounds_folder is not None:
      replace_imgs_paths = general_utils.list_files_in_folder(backgrounds_folder, 'pickle', recursive=True)
      if backgrounds_len is not None:
        np.random.seed(1) # TODO remove seed
        np.random.shuffle(replace_imgs_paths) # shuffling before reducing backgrounds
        replace_imgs_paths = replace_imgs_paths[:backgrounds_len] # reducing backgrounds cardinality

    noises_paths = []
    if noise_folder is not None:
      noises_paths = general_utils.list_files_in_folder(noise_folder, 'pickle', recursive=False)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    model_name = '{0} {1} - {2}{3}_len{4}_b{5}_{6}w_{7}{8}{9}_ep{10}'.format(
        timestr,                                    # 0
        socket.gethostname().replace('.','_'),      # 1
        'regr' if regression else '',               # 2
        'class' if classification else '',          # 3
        len(list_files),                            # 4
        batch_size,                                 # 5
        'r' if initial_weights is None else 'o',    # 6
        'trainfrom{}'.format(retrain_from) if retrain_from is not None else 'notrain',    # 7 optional
        '_{}(len{})'.format(backgrounds_name or 'bg', len(replace_imgs_paths)) if len(replace_imgs_paths) > 0 else '', # 8 optional
        '_augm{}'.format(str(augmentation_prob).replace('.','')) if augmentation_prob > 0 else '', # 9 optional
        epochs                                      # 10
    )

    with open(list_files[0], 'br') as first:
        input_shape = pickle.load(first)['image'].shape

    model = network_utils.network_create(
      input_shape, regression, classification, 
      initial_weights, retrain_from, view_summary=False
    )

    model, history = network_utils.network_train_generator(
      model, input_shape, list_files, 
      regression, classification, 
      replace_imgs_paths, augmentation_prob, noises_paths,
      batch_size, epochs, oversampling, verbose, 
      use_lr_reducer=use_lr_reducer, use_early_stop=use_early_stop, 
      use_profiler=use_profiler, profiler_dir=profile_dir
    )

    if view_stats or save_stats:
      network_utils.network_stats(
        history.history, regression, classification, 
        view_stats, save_stats, save_folder, model_name
      )

    if save_model:
        network_utils.network_save(save_folder, model_name, model, history.history)



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
        raise FileNotFoundError(string)

  default_batch_size = 64
  default_epochs = 30
  debug_data_len = 64
  debug_batch_size = 4
  debug_epochs = 2
  debug_oversampling = 1

  parser = argparse.ArgumentParser(description='Train the network on the given dataset using a generator which performs dynamic loading and data augmentation.')
  parser.add_argument('data_folder', type=dir_path, help='path to the dataset') # required
  parser.add_argument('gpu_number', type=int, help='number of the GPU to use') # required
  parser.add_argument('-r', '--regression', action='store_true', help='specify the argument if you want to perform regression')
  parser.add_argument('-c', '--classification', action='store_true', help='specify the argument if you want to perform classification')
  parser.add_argument('--data_len', type=int, default=None, metavar='DL', help='max number of samples in the dataset (default = entire dataset, debug = {})'.format(debug_data_len))
  parser.add_argument('--batch_size', type=int, default=default_batch_size, metavar='BS', help='training batch size (default = {}, debug = {})'.format(default_batch_size, debug_batch_size))
  parser.add_argument('--epochs', type=int, default=default_epochs, metavar='E', help='number of training epochs (default = {}, debug = {})'.format(default_epochs, debug_epochs))
  parser.add_argument('--oversampling', type=int, default=1, metavar='OS', help='number of times the dataset has to be repeated for each epoch (default = 1, debug = {})'.format(debug_oversampling))
  parser.add_argument('--weights_path', type=file_path, metavar='WP', help='path to the network initial weights dictionary {"layer_name": get_weights}') # required
  parser.add_argument('--retrain_from', type=int, default=None, metavar='RF', help='number of layer to retrain from (default = no training, 0 = complete training)')
  parser.add_argument('--verbose', type=int, default=2, metavar='VER', help='keras training verbosity (0: silent, 1: complete, 2: one line per epoch')
  parser.add_argument('--lr_reducer', action='store_true', help='specify the argument if you want to use learning rate reducer callback')
  parser.add_argument('--early_stop', action='store_true', help='specify the argument if you want to use early stop callback')
  parser.add_argument('--profiler', action='store_true', help='specify the argument if you want to use TensorBoard profiler callback')
  parser.add_argument('--profiler_dir', type=dir_path, metavar='PD', help='path in which to save TensorBoard logs')
  parser.add_argument('--bgs_folder', type=dir_path, metavar='BGF', help='path to backgrounds folder, treaten recursively (default = no background replacement)')
  parser.add_argument('--bgs_len', type=int, default=None, metavar='BL', help='max number of backgrounds to consider (default = entire bgs_folder content)')
  parser.add_argument('--bgs_name', type=str, default=None, metavar='BGN', help='name/identifier of the chosen backgrounds set, just used for naming purposes')
  # parser.add_argument('--augmentation', action='store_true', help='specify the argument if you want to perform standard image augmentation')
  parser.add_argument('--aug_prob', type=float, default=0, metavar='AP', help='probability of performing image augmentation on each sample')
  parser.add_argument('--noise_folder', type=dir_path, metavar='NF', help='path to noises for image augmentation (default = no noise augmentation)')
  parser.add_argument('--view_stats', action='store_true', help='specify the argument if you want to visualize model metrics')
  parser.add_argument('--save', action='store_true', help='specify the argument if you want to save the model and metrics')
  parser.add_argument('--save_folder', type=dir_path, metavar='SVF', help='path where to save the model and metrics')
  parser.add_argument('--debug', action='store_true', help='if the argument is specified, some parameters are set (overwritten) to debug values')

  parsed_args = parser.parse_args()
  if parsed_args.debug:
    parsed_args.data_len = debug_data_len
    parsed_args.batch_size = debug_batch_size
    parsed_args.epochs = debug_epochs
    parsed_args.oversampling = debug_oversampling

  return parsed_args



if __name__ == "__main__":

  ## --- Args
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')

  ## --- GPU settings

  cuda_visibile_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
  if cuda_visibile_devices is not None:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
  
  gpus = tf.config.experimental.list_physical_devices('GPU')

  if gpus:
    try:
      print('Selected GPU number', args.gpu_number)
      if args.gpu_number < 0:
        tf.config.set_visible_devices([], 'GPU')
      else:
        tf.config.experimental.set_visible_devices(gpus[args.gpu_number], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu_number], True) # not immediately allocating the full memory
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs \n')
    except RuntimeError as e:
      print(e) # visible devices must be set at program startup
  else:
      print('No available GPUs \n')

  ## --- Training

  train_with_generator(
    args.data_folder, args.weights_path, args.data_len, args.regression, args.classification,
    args.retrain_from, args.verbose, args.batch_size, args.epochs, args.oversampling,
    args.lr_reducer, args.early_stop, args.profiler, args.profiler_dir,
    args.bgs_folder, args.bgs_len, args.bgs_name, args.aug_prob, args.noise_folder,
    args.view_stats, save_stats=args.save, save_model=args.save, save_folder=args.save_folder
  )
