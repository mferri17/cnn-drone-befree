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

### CUSTOM IMPORTS
## see https://stackoverflow.com/a/59703673/10866825 
## and https://stackoverflow.com/a/63523670/10866825

sys.path.append('.')
from functions import general_utils



################################
### GLOBAL VARIABLES
######



################################
### FUNCTIONS
######

def img_processing(source_path, source_ext, dest_path, 
                   img_height, img_width, grayscale, cvt_float = False, 
                   data_size = None, find_recursive = False):

  images_paths = general_utils.list_files_in_folder(source_path, source_ext, find_recursive)
  images_paths = images_paths[:data_size]
  
  general_utils.create_folder_if_not_exist(dest_path)

  errors = []
  print('Preprocessing {} images to be ({},{},{}) ...'.format(len(images_paths), img_height, img_width, 1 if grayscale else 3))
  start_time = time.monotonic()

  for i, path in enumerate(images_paths):
    try:
      img = cv2.imread(path).astype('uint8')
    except:
      # cv2 does not read some images ERR properly, so the `astype` method fails
      # matplotlib.imread does not have this problem but, at the end, these ERR images still raise some issues
      # so we decide to directly skip problematic images
      errors.append(path)
      continue

    # img = img[..., ::-1]  # RGB --> BGR (for working with cv2 later)
    shape = img.shape

    # -- Convert images of any shape to shape (height,width,3)

    if len(shape) != 2 and len(shape) != 3:
      raise Exception('Images must have 2 (h,w) or 3 dimensions (h,w,c), error with {}'.format(path))
    elif len(shape) == 2 or shape[2] == 1: 
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # transforms grayscale image to have 3 channels
    elif shape[2] > 3:
      img = img[:,:,3] # removes alpha and other channels

    # -- Resize

    inter = cv2.INTER_AREA # best for image decimation, see https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    img = cv2.resize(img, (img_width, img_height), interpolation = inter) 
      
    # -- Convert grayscale images to shape (height,width,1) or BGR to RGB

    if grayscale:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis] # (height, width, 1)
    else:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -- Convert to float32

    if cvt_float:
      img = (img / 255).astype('float32')

    # -- Save and log progress

    img_folder_path, img_file = os.path.split(path)
    img_folder_name = os.path.basename(img_folder_path)
    img_name, img_ext = os.path.splitext(img_file)
    save_path = os.path.join(dest_path, img_folder_name + '/')
    general_utils.create_folder_if_not_exist(save_path)

    with open(os.path.join(save_path, img_name + '.pickle'), 'wb') as fp:
      pickle.dump(img, fp)

    if i % 1000 == 0:
      print('Progress {}/{}'.format(i, len(images_paths)))
  
    
  print('\n{} errors: {}'.format(len(errors), errors))
  print('\nProcess finished in {:.2f} minutes.'.format((time.monotonic() - start_time)/60))
  

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

  debug_data_size = 16

  parser = argparse.ArgumentParser(description='Take input images and saves them as float32 ndarrays of the given size in the pickle format')
  parser.add_argument('source_path', type=dir_path, help='path to the dataset') # required
  parser.add_argument('source_ext', type=str, help='image extension to be found') # required
  parser.add_argument('dest_path', type=dir_path, help='path where to store the result') # required
  parser.add_argument('dest_height', type=int, help='height of the resulting images')
  parser.add_argument('dest_width', type=int, help='height of the resulting images')
  parser.add_argument('-g', '--dest_gray', action='store_true', help='if the argument is specified, the result will be grayscale')
  parser.add_argument('--float32', action='store_true', help='if the argument is specified, the result will be float32 instead of uint8')
  parser.add_argument('--data_size', type=int, default=None, metavar='DS', help='max number of samples in the dataset (default = entire dataset, debug = {})'.format(debug_data_size))
  parser.add_argument('-r', '--recursive', action='store_true', help='if the argument is specified, the `source_path` content is treated recursively in sub-directories')
  parser.add_argument('-d', '--debug', action='store_true', help='if the argument is specified, some parameters are set (overwritten) to debug values')

  parsed_args = parser.parse_args()
  if parsed_args.debug:
    parsed_args.data_size = debug_data_size

  return parsed_args



if __name__ == "__main__":
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')

  img_processing(
    args.source_path, args.source_ext, args.dest_path,
    args.dest_height, args.dest_width, args.dest_gray, args.float32,
    args.data_size, args.recursive
  )