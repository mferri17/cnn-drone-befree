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
# from functions import network_utils



################################
### GLOBAL VARIABLES
######




################################
### FUNCTIONS
######

def uncompact_from_maskrcnn_data():

  current_path = os.path.dirname(os.path.realpath(__file__))
  
  compact_data_path = os.path.join(current_path, './../dev-visualization/maskRCNN/20201022_121833 detections orig_test total11035/maskrcnn final - detections orig_test total11035.npy')
  new_data_path = os.path.join(current_path, 'C:/Users/96mar/Desktop/meeting_dario/data/orig_test_11030/')
  data_name = 'orig_test 11030'

  compact_data = np.load(compact_data_path, allow_pickle=True)
  general_utils.create_folder_if_not_exist(new_data_path)
  
  print('Starting to uncompact...')
  
  for i, frame in enumerate(compact_data):
    if i % 5000 == 0:
      print('Progress {}/{}'.format(i+1, len(compact_data)))

    frame['image'] = frame['image'].astype('uint8') # recasting
    frame['mask'] = frame['mask'].astype('uint8') # recasting
    frame_name = '{} - frame {:06}.pickle'.format(data_name, i)
    with open(os.path.join(new_data_path, frame_name), 'wb') as fp:
      pickle.dump(frame, fp)

  print('Uncompact finished.')



################################
### MAIN
######

if __name__ == "__main__":

  uncompact_from_maskrcnn_data()



