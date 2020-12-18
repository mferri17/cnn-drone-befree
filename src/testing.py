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

def test_models_datasets(model_paths, data_folders, batch_size, bgs_folders, bg_smoothmask, save, save_folder):

  # model_paths = [arena, replaced, replaced_augmented, replaced_augmented_noise, replaced_augmented_noise_smooth]
  # data_folders = [arena, test, test_green]
  # bgs_paths = [bg20, room11, room15, indoorCVPRtest]

  regression = True
  classification = False

  bgs_folders.insert(0, None) # so that we can also consider the dataset without any background replacement
  dependencies = { 'r2_keras': network_utils.r2_keras } # see https://stackoverflow.com/a/55652105/10866825

  if save:
    original_stdout = sys.stdout # Save a reference to the original standard output
    timestr = time.strftime("%Y%m%d_%H%M%S")
    save_name = '{} evaluation.txt'.format(timestr)
    save_path = os.path.join(save_folder, save_name)
    output_file = open(save_path, 'w')
    print('Printing on', save_path)
    sys.stdout = output_file # Change the standard output to the file we created.
  
  for dp in data_folders:
    list_files = [os.path.join(dp, fn) for fn in os.listdir(dp)]
    input_shape = general_utils.load_pickle(list_files[0])['image'].shape

    print('\n\n-------------------------------------------------------------------------------------------------------------------\n')
    print('Evaluation on DATASET from', dp)
    
    model = network_utils.network_create(input_shape, regression, classification, retrain_from_layer=0, view_summary=False)
    model = network_utils.network_compile(model, regression, classification)
    
    for mp in model_paths:
      # model = tf.keras.models.load_model(mp, custom_objects=dependencies)
      model.load_weights(mp, by_name=True)
      print('\n--------------------- MODEL imported from', mp)
      
      for bp in bgs_folders:
        print('\n----- BACKGROUND replacement with', bp, '\n')
        bgs = network_utils.load_backgrounds(bp, bg_smoothmask=bg_smoothmask)

        network_utils.network_evaluate(model, list_files, input_shape, batch_size,
                                        regression, classification, bgs, bg_smoothmask)

  if save:
    sys.stdout = original_stdout # Reset the standard output to its original value
    output_file.close()


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
  parser.add_argument('--bgs_folders', nargs="+", type=dir_path, default=[], metavar='BGF', help='path to backgrounds folder, treaten recursively (default = no background replacement)')
  # parser.add_argument('--bgs_names', nargs="+", type=str, default=[], metavar='BGN', help='name/identifier of the chosen backgrounds set, just used for naming purposes')
  parser.add_argument('--bg_smoothmask', action='store_true', help='specify the argument if you want to smooth the mask before replacing the background')
  parser.add_argument('--save', action='store_true', help='specify the argument if you want to save evaluation metrics')
  parser.add_argument('--save_folder', type=dir_path, metavar='SVF', help='path where to save evaluation metrics')

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
    args.bgs_folders, args.bg_smoothmask,
    args.save, args.save_folder,
  )
