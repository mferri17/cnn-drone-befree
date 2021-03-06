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

def bgreplace_example(data_folder, bgs_folder):

    data_files = general_utils.list_files_in_folder(data_folder, 'pickle', recursive=True)
    bgs_files = general_utils.list_files_in_folder(bgs_folder, 'pickle', recursive=True)

    step = 2000
    ncols = 4
    nrows = 8
    size = (nrows * ncols)

    for start in range(0, len(data_files), step * size):
      fix, ax = plt.subplots(nrows, ncols, figsize=(30, 45),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.15))

      end = start + step * size
      for count, i in enumerate(range(start, end, step)):
        sample = general_utils.load_pickle(data_files[i])

        img = sample['image'].astype('uint8')
        mask = sample['mask'].astype('uint8')
        bg = (general_utils.load_pickle(bgs_files[np.random.randint(0, len(bgs_files))]) / 255).astype('float32')
        result = general_utils.image_augment_background(img, mask, bg, smooth=True)

        cell = ax[count//ncols, count%ncols]
        # cell.set_title('frame {:05}'.format(i), fontsize=3)
        cell.imshow(result)

      plt.savefig('C:/Temp/bgreplace_example.png', dpi=300, bbox_inches='tight')



def train_with_generator(data_folder, network_weights_path, data_len,
                         regression, classification,
                         retrain_from, verbose, batch_size, epochs, oversampling, val_not_shuffle,
                         use_lr_reducer, use_early_stop, compute_r2,
                         use_profiler, profile_dir,
                         backgrounds_folder, backgrounds_len, backgrounds_name, bg_smoothmask,
                         augmentation_prob, noise_folder,
                         view_stats, save_stats, save_model, save_folder):
    
    # --- Parameters

    list_files = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
    list_files = list_files[:data_len]

    if network_weights_path is not None and os.path.exists(network_weights_path):
      with open(network_weights_path, 'rb') as fp:
        initial_weights = pickle.load(fp)
    else:
      initial_weights = None

    with open(list_files[0], 'br') as first:
        input_shape = pickle.load(first)['image'].shape

    backgrounds = network_utils.load_backgrounds(backgrounds_folder, bg_smoothmask, backgrounds_len)
    noises = network_utils.load_noises(noise_folder)

    # --- Naming

    time_str = time.strftime("%Y%m%d_%H%M%S")
    backgrounds_str = '_{}(len{}{})'.format(backgrounds_name or 'bg', len(backgrounds), ',smooth' if bg_smoothmask else '')
    augmentation_str = '_augm{}{}'.format(str(augmentation_prob).replace('.',''), '(noise)' if len(noises) > 0 else '')

    model_name = '{0} {1} - {2}{3}_len{4}_b{5}_{6}w_{7}{8}{9}_ep{10}'.format(
        time_str,                                                                          # 0
        socket.gethostname().replace('.','_'),                                            # 1
        'regr' if regression else '',                                                     # 2 optional
        'class' if classification else '',                                                # 3 optional
        len(list_files),                                                                  # 4
        batch_size,                                                                       # 5
        'r' if initial_weights is None else 'o',                                          # 6
        'trainfrom{}'.format(retrain_from) if retrain_from is not None else 'notrain',    # 7
        backgrounds_str if len(backgrounds) > 0 else '',                           # 8 optional
        augmentation_str if augmentation_prob > 0 else '',                                # 9 optional
        epochs                                                                            # 10
    )

    # --- Computation

    model = network_utils.network_create(
      input_shape, regression, classification, 
      initial_weights, retrain_from, view_summary=False
    )

    model, history = network_utils.network_train_generator(
      model, input_shape, list_files, 
      regression, classification, 
      backgrounds, bg_smoothmask,
      augmentation_prob, noises,
      batch_size, epochs, oversampling, verbose, 
      0.3, (not val_not_shuffle),
      use_lr_reducer, use_early_stop, compute_r2,
      use_profiler, profile_dir
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
  debug_bgs_len = 100

  parser = argparse.ArgumentParser(description='Train the network on the given dataset using a generator which performs dynamic loading and data augmentation.')
  parser.add_argument('gpu_number', type=int, help='number of the GPU to use') # required
  parser.add_argument('data_folder', type=dir_path, help='path to the dataset') # required
  parser.add_argument('-r', '--regression', action='store_true', help='specify the argument if you want to perform regression')
  parser.add_argument('-c', '--classification', action='store_true', help='specify the argument if you want to perform classification')
  parser.add_argument('--data_len', type=int, default=None, metavar='DL', help='max number of samples in the dataset (default = entire dataset, debug = {})'.format(debug_data_len))
  parser.add_argument('--batch_size', type=int, default=default_batch_size, metavar='BS', help='training batch size (default = {}, debug = {})'.format(default_batch_size, debug_batch_size))
  parser.add_argument('--epochs', type=int, default=default_epochs, metavar='E', help='number of training epochs (default = {}, debug = {})'.format(default_epochs, debug_epochs))
  parser.add_argument('--oversampling', type=int, default=1, metavar='OS', help='number of times the dataset has to be repeated for each epoch (default = 1, debug = {})'.format(debug_oversampling))
  parser.add_argument('--val_not_shuffle', action='store_true', help='specify the argument if you want to NOT shuffle dataset for splitting between trainining and validation sets (default=shuffle)')
  parser.add_argument('--weights_path', type=file_path, metavar='WP', help='path to the network initial weights dictionary {"layer_name": get_weights}') # required
  parser.add_argument('--retrain_from', type=int, default=None, metavar='RF', help='number of layer to retrain from (default = no training, 0 = complete training)')
  parser.add_argument('--verbose', type=int, default=2, metavar='VER', help='keras training verbosity (0: silent, 1: complete, 2: one line per epoch')
  parser.add_argument('--lr_reducer', action='store_true', help='specify the argument if you want to use learning rate reducer callback')
  parser.add_argument('--early_stop', action='store_true', help='specify the argument if you want to use early stop callback')
  parser.add_argument('--compute_r2', action='store_true', help='specify the argument if you want to compute R2 (please note that it dramatically increases training time)')
  parser.add_argument('--profiler', action='store_true', help='specify the argument if you want to use TensorBoard profiler callback')
  parser.add_argument('--profiler_dir', type=dir_path, metavar='PD', help='path in which to save TensorBoard logs')
  parser.add_argument('--bgs_folder', type=dir_path, metavar='BGF', help='path to backgrounds folder, treaten recursively (default = no background replacement)')
  parser.add_argument('--bgs_len', type=int, default=None, metavar='BL', help='max number of backgrounds to consider (default = entire bgs_folder content, debug = {})'.format(debug_bgs_len))
  parser.add_argument('--bgs_name', type=str, default=None, metavar='BGN', help='name/identifier of the chosen backgrounds set, just used for naming purposes')
  parser.add_argument('--bg_smoothmask', action='store_true', help='specify the argument if you want to smooth the mask before replacing the background')
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
    parsed_args.bgs_len = debug_bgs_len

  return parsed_args



if __name__ == "__main__":

  ## --- Args
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')


  ## --- GPU settings

  network_utils.use_gpu_number(args.gpu_number)


  ## --- Training

  train_with_generator(
    args.data_folder, args.weights_path, args.data_len, args.regression, args.classification,
    args.retrain_from, args.verbose, args.batch_size, args.epochs, args.oversampling, args.val_not_shuffle,
    args.lr_reducer, args.early_stop, args.compute_r2, args.profiler, args.profiler_dir,
    args.bgs_folder, args.bgs_len, args.bgs_name, args.bg_smoothmask, args.aug_prob, args.noise_folder,
    args.view_stats, save_stats=args.save, save_model=args.save, save_folder=args.save_folder
  )

  # bgreplace_example(args.data_folder, args.bgs_folder)