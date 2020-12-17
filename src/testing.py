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

from sklearn.metrics import r2_score, mean_squared_error


def test_models_datasets(model_paths, data_folders, batch_size,
          bgs_folders, bgs_names, bg_smoothmask):

  # model_paths = [arena, replaced, replaced_augmented, replaced_augmented_noise, replaced_augmented_noise_smooth]
  # data_folders = [arena, test, test_green]
  # bgs_paths = [bg20, room11, room15, indoorCVPRtest]

  for mp in model_paths:
    print('\n\n------------------------------------------------------------------------------------------------\n')
    print(mp)
    model = tf.keras.models.load_model(mp)
    print('Model imported from', mp, '\n\n')
    continue

    for dp in data_folders:
      list_files = [os.path.join(dp, fn) for fn in os.listdir(dp)]
      input_shape = general_utils.load_pickle(list_files[0])['image'].shape

      for bp in bgs_folders:
        bgs = network_utils.load_backgrounds(bp, bg_smoothmask=bg_smoothmask)

        network_utils.network_evaluate(model, list_files, input_shape, batch_size,
                                        regression=True, classification=False,
                                        backgrounds=bgs, bg_smoothmask=bg_smoothmask)



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

  parser = argparse.ArgumentParser(description='Train the network on the given dataset using a generator which performs dynamic loading and data augmentation.')
  parser.add_argument('gpu_number', type=int, help='number of the GPU to use') # required
  parser.add_argument("--model_paths", nargs="+", type=file_path, default=[])
  parser.add_argument("--data_folders", nargs="+", type=dir_path, default=[])
  # parser.add_argument('-r', '--regression', action='store_true', help='specify the argument if you want to perform regression')
  # parser.add_argument('-c', '--classification', action='store_true', help='specify the argument if you want to perform classification')
  parser.add_argument('--batch_size', type=int, default=default_batch_size, metavar='BS', help='training batch size (default = {})'.format(default_batch_size))
  parser.add_argument('--bgs_folders', nargs="+", type=dir_path, default=[None], metavar='BGF', help='path to backgrounds folder, treaten recursively (default = no background replacement)')
  parser.add_argument('--bgs_names', nargs="+", type=str, default=[], metavar='BGN', help='name/identifier of the chosen backgrounds set, just used for naming purposes')
  parser.add_argument('--bg_smoothmask', action='store_true', help='specify the argument if you want to smooth the mask before replacing the background')

  parsed_args = parser.parse_args()
  return parsed_args



if __name__ == "__main__":

  ## --- Args
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')
  
  ## --- GPU settings

  network_utils.use_gpu_number(args.gpu_number)

  
  ## --- Testing

  test_models_datasets(
    args.model_paths, args.data_folders, args.batch_size, 
    args.bgs_folders, args.bgs_names, args.bg_smoothmask,
  )
