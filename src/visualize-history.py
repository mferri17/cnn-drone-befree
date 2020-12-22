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

def visualize_like_training(histories_paths, regression, classification, loss_scale, r2_scale, acc_scale, dpi, pdf, save, save_folder):

  for hp in histories_paths:

    history = general_utils.load_pickle(hp)
    
    time.sleep(1)
    timestr = time.strftime("%Y%m%d_%H%M%S")
    _, filename = os.path.split(hp)
    name, ext = os.path.splitext(filename)
    name = ' - '.join(name.split(' - ')[:-1]) # remove the "history" word
    
    save_name = '{} - {}'.format(timestr, name)
    view = not save

    network_utils.network_stats(history, regression, classification, view, save, save_folder, save_name,
                                dpi, pdf, loss_scale, r2_scale, acc_scale)

    print('Model stats {}.'.format('saved as {}'.format(save_name) if save else 'shown'))


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

  parser = argparse.ArgumentParser(description='Visualize given models histories.')
  parser.add_argument('histories_paths', type=file_path, nargs='+', help='list of histories paths to be visualized')
  parser.add_argument('-r', '--regression', action='store_true', help='specify the argument if you want to perform regression')
  parser.add_argument('-c', '--classification', action='store_true', help='specify the argument if you want to perform classification')
  parser.add_argument('--loss_scale', type=float, nargs=2, metavar='LS', help='min and max loss chart scale')
  parser.add_argument('--r2_scale', type=float, nargs=2, metavar='RS', help='min and max loss chart scale')
  parser.add_argument('--acc_scale', type=float, nargs=2, metavar='AS', help='min and max loss chart scale')
  parser.add_argument('--dpi', type=int, metavar='DPI', help='image resolution when saved to PNG (100: low, medium: 500, high: 2000)')
  parser.add_argument('--pdf', action='store_true', help='specify the argument if you want to save the metrics as PDF rather than PNG')
  parser.add_argument('--save', action='store_true', help='specify the argument if you want to save evaluation metrics, otherwise they will just be shown')
  parser.add_argument('--save_folder', type=dir_path, metavar='SVF', help='path where to save evaluation metrics')

  parsed_args = parser.parse_args()
  return parsed_args



if __name__ == "__main__":

  ## --- Args
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')

  ## --- Testing

  visualize_like_training(
    args.histories_paths, args.regression, args.classification,
    args.loss_scale, args.r2_scale, args.acc_scale, 
    args.dpi, args.pdf, args.save, args.save_folder,
  )
