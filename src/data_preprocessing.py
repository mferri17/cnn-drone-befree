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

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


### CUSTOM IMPORTS
## see https://stackoverflow.com/a/59703673/10866825 
## and https://stackoverflow.com/a/63523670/10866825 (for VS Code)

sys.path.append('.')
from functions import general_utils



################################
### GLOBAL VARIABLES
######



################################
### FUNCTIONS
######

def input_pipeline_optimization(source, destination, smooth_mask = True):
  pass


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

  parser = argparse.ArgumentParser(description='Performs different types of data preprocessing.')
  parser.add_argument('data_folder', type=dir_path, help='path to the dataset') # required
  parser.add_argument('save_folder', type=dir_path, metavar='SVF', help='path where to save the new dataset') # required
  parser.add_argument('--data_size', type=int, default=None, metavar='DS', help='max number of samples in the dataset (default = entire dataset)')

  return parsed_args



if __name__ == "__main__":

  ## --- Args
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')

  ## --- Preprocessing

  if args.mode == 1:
    input_pipeline_optimization()
