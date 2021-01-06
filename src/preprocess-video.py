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

def video_processing(video_path, dest_path, frames_count,
                     img_height, img_width, grayscale):

  time_str = time.strftime("%Y%m%d_%H%M%S")
  video_name = general_utils.get_name_file(video_path)
  save_path = os.path.join(dest_path, 'custom', '{} {}/'.format(time_str, video_name))
  general_utils.create_folder_if_not_exist(save_path)
  
  video_stream = cv2.VideoCapture(video_path) 
  if frames_count is None:
    frames_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

  print('Saving to', save_path)
  print('Preprocessing video to extract images of shape ({},{},{}) ...'.format(img_height, img_width, 1 if grayscale else 3))
  start_time = time.monotonic()
  i = 0

  while(i < frames_count): 
    ret, img = video_stream.read() 

    if not ret:
      break # video finished

    img = img.astype('uint8')
    shape = img.shape

    # -- Convert images of any shape to shape (height,width,3)

    if len(shape) != 2 and len(shape) != 3:
      raise Exception('Images must have 2 (h,w) or 3 dimensions (h,w,c), error with {}'.format(path))
    elif len(shape) == 2 or shape[2] == 1: 
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # transforms grayscale image to have 3 channels
    elif shape[2] > 3:
      img = img[:,:,3] # removes alpha and other channels

    # -- Crop

    # -- Resize
    
    inter = cv2.INTER_AREA # best for image decimation, see https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121
    img = cv2.resize(img, (img_width, img_height), interpolation = inter) 
      
    # -- Convert grayscale images to shape (height,width,1) or BGR to RGB

    if grayscale:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis] # (height, width, 1)
    else:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -- Save and log progress
    
    file_path = os.path.join(save_path, '{} - frame {:06}'.format(video_name, i))

    if i == 0: # saves the first image just to have a reference of it
      cv2.imwrite(file_path + '.jpg', img)

    with open(file_path + '.pickle', 'wb') as fp:
      # img = cv2.flip(img, 0)
      sample = {
        'image': img.astype('uint8'), 
        'centr': tuple([None]),
        'bbox': tuple([None]),
        'mask': np.array([0]).astype('uint8'),
        'gt': np.array([None, None, None, None]).astype('float64')
      }
      pickle.dump(sample, fp)

    if i % 1000 == 0:
      print('Progress {}/{}'.format(i, frames_count))
    
    i += 1

  video_stream.release() 
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
        raise FileNotFoundError(string)

  debug_data_size = 120

  parser = argparse.ArgumentParser(description='Take input images and saves them as float32 ndarrays of the given size in the pickle format')
  parser.add_argument('video_path', type=file_path, help='path to the dataset') # required
  parser.add_argument('dest_path', type=dir_path, help='path where to store the result') # required
  parser.add_argument('dest_height', type=int, help='height of the resulting images')
  parser.add_argument('dest_width', type=int, help='width of the resulting images')
  parser.add_argument('--data_size', type=int, default=None, metavar='DS', help='max number of frames of the video to extract (default = entire video, debug = {})'.format(debug_data_size))
  parser.add_argument('-g', '--dest_gray', action='store_true', help='if the argument is specified, the result will be grayscale')
  parser.add_argument('-d', '--debug', action='store_true', help='if the argument is specified, some parameters are set (overwritten) to debug values')

  parsed_args = parser.parse_args()
  if parsed_args.debug:
    parsed_args.data_size = debug_data_size

  return parsed_args



if __name__ == "__main__":
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')

  video_processing(
    args.video_path, args.dest_path, args.data_size,
    args.dest_height, args.dest_width, args.dest_gray
  )