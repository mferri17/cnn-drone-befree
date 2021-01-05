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


def triangle_pointer(img, p, rotation, color_):
    width = 640 # 520
    height = 480 # 330
    if p[0] >= width:
        p[0] = width - 40
    if p[1] >= height:
        p[1] = height - 40
    point_im = np.ones((40, 40, 3)) * 255
    side_len = 15
    p_1 = (20, 20)
    triangle = np.array([[int(p_1[0] - (math.sin(math.radians(60)) * side_len)), int(p_1[1] - (math.cos(math.radians(60)) * side_len))],
                         p_1,
                         [int(p_1[0] - (math.sin(math.radians(60)) * side_len)), int(p_1[1] + (math.cos(math.radians(60)) * side_len))]], np.int32)
    cv2.fillConvexPoly(point_im, triangle, color=color_, lineType=1)
    M = cv2.getRotationMatrix2D((20, 20), rotation, 1)
    rotated = cv2.warpAffine(point_im, M, (40, 40), cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    try:
      img[p[1] - 20:p[1] + 20, p[0] - 20:p[0] + 20] = rotated.astype(np.uint8)
    except:
      pass # some values for W (yaw) are out of scale, so they produces errors

def square_pointer(img, p, color_, w=640, h=480):
    point_im = np.ones((10, 10, 3)) * 255
    p_1 = (0, 0)
    p_2 = (10, 10)
    cv2.rectangle(point_im, p_1, p_2, color=color_, thickness=-1)
    p[0] = max(w - 10 if p[0] >= w-5 else p[0], 10)
    p[1] = max(h - 10 if p[1] >= h-5 else p[1], 10)
    img[p[1]-5 : p[1]+5, p[0]-5 : p[0]+5] = point_im.astype(np.uint8)


def draw_pointer(img, value, variable, model, frame_vert_dim, frame_horiz_dim, c_x=320, c_y=240, width=640, height=480):
  center_x = c_x
  center_y = c_y
  bar_width = 30

  # GT + models specs
  if model == -1: # GT
    color_ = (0, 0, 0)    # black
  elif model == 0:
    color_ = (0, 255, 0)  # green
  elif model == 1:
    color_ = (255, 0, 0)  # blue
  elif model == 2:
    color_ = (0, 0, 255)  # red
  else:
    raise ValueError('Parameter `model` is not valid.')

  # variables specs
  if variable == 'x': # left
    p_x = int(center_x - bar_width - frame_horiz_dim / 2)
    p_y = int(center_y + frame_vert_dim / 2 * ((-value+1.5) * 1.5))
    pt = [p_x, p_y]
    rot = 0
    pl = [(p_x, p_y), (p_x + 30, p_y)]
    ps = [p_x + 5 + (10 * model), p_y]
  elif variable == 'y': # bottom
    p_x = int(center_x + frame_horiz_dim / 2 * (-value * 1.5))
    p_y = int(center_y + bar_width + frame_vert_dim / 2)
    pt = [p_x, p_y]
    rot = 90
    pl = [(p_x, p_y), (p_x, p_y - 30)]
    ps = [p_x, p_y - 5 - (10 * model)]
  elif variable == 'z': # right
    p_x = int(center_x + bar_width + frame_horiz_dim / 2)
    p_y = int(center_y + frame_vert_dim / 2 * (-value * 1.5))
    pt = [p_x, p_y]
    rot = -180
    pl = [(p_x, p_y), (p_x - 30, p_y)]
    ps = [p_x - 5 - (10 * model), p_y]
  elif variable == 'w': # top
    p_x = int(center_x + frame_horiz_dim / 2 * (value * 1.5))
    p_y = int(center_y - bar_width - frame_vert_dim / 2)
    pt = [p_x, p_y]
    rot = -90
    pl = [(p_x, p_y), (p_x, p_y + 30)]
    ps = [p_x, p_y + 5 + (10 * model)]
  else:
    raise ValueError('Parameter `variable` is not valid.')
  
  if model == -1:
    triangle_pointer(img, pt, rot, color_)
  else:
    cv2.line(img, pl[0], pl[1], color_, thickness=2)
    square_pointer(img, ps, color_, w=width, h=height)

    
################################
 

def video_simple(path, images):
  video_width = 518
  video_height = 326
  video_fps = 30
  video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (video_width, video_height))
  
  for i, frame in enumerate(images):
    frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)

    border = (np.ones((480, 640, 3)) * 255).astype(np.uint8)
    center_point_x = 320
    center_point_y = 240
    x_offset = int(center_point_x - frame.shape[1] / 2)
    y_offset = int(center_point_y - frame.shape[0] / 2)
    border[y_offset:y_offset + frame.shape[0], x_offset:x_offset + frame.shape[1]] = frame
    im_final = border
    margin_h = int(center_point_y - 327 // 2)
    margin_w = int(center_point_x - 519 // 2)
    cv2.putText(im_final, 'Frame: {}'.format(i), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    crop_img = im_final[margin_h:-margin_h, margin_w:-margin_w]

    video_writer.write(crop_img)
  
  video_writer.release()


def video_multi_predictions(path, images, actuals, predictions, fps=25):
  legend_x, legend_y = (50, 50)
  legend = cv2.imread('C:/Users/96mar/Desktop/meeting_dario/videos/legend.png') # TODO fix

  # video writer

  details_height = 0
  video_width, video_height = 518, 326 + details_height
  video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, (video_width, video_height))

  # parameters

  color_main = (0, 0, 0)
  font = cv2.FONT_HERSHEY_DUPLEX
  area_height = 480
  area_width = 640
  center_point_x = area_width // 2
  center_point_y = area_height // 2
  frame_h, frame_w = 240, 432

  scale_x_orig = [(int(center_point_x - frame_w / 2), center_point_y), (int(center_point_x - frame_w / 2 - 30), center_point_y)]
  scale_y_orig = [(center_point_x, int(center_point_y + frame_h / 2)), (center_point_x, int(center_point_y + frame_h / 2 + 30))]
  scale_z_orig = [(int(center_point_x + frame_w / 2), center_point_y), (int(center_point_x + frame_w / 2 + 30), center_point_y)]
  scale_w_orig = [(center_point_x, int(center_point_y - frame_h / 2)), (center_point_x, int(center_point_y - frame_h / 2 - 30))]

  border_x = [(int(center_point_x - frame_w / 2), int(center_point_y - frame_h / 2)), (int(center_point_x - frame_w / 2 - 30), int(center_point_y + frame_h / 2))]
  border_y = [(int(center_point_x - frame_w / 2), int(center_point_y + frame_h / 2)), (int(center_point_x + frame_w / 2), int(center_point_y + frame_h / 2 + 30))]
  border_z = [(int(center_point_x + frame_w / 2), int(center_point_y - frame_h / 2)), (int(center_point_x + frame_w / 2 + 30), int(center_point_y + frame_h / 2))]
  border_w = [(int(center_point_x - frame_w / 2), int(center_point_y - frame_h / 2)), (int(center_point_x + frame_w / 2), int(center_point_y - frame_h / 2 - 30))]
  
  # creation

  for frame_idx, frame in enumerate(images):

    frame = cv2.cvtColor(cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
    im_final = (np.ones((area_height + details_height, area_width, 3)) * 255).astype(np.uint8) # bigger for allowing out of space predictions?

    # --- GT and predictions pointers for each model and variable

    for var_idx, var in enumerate(['x', 'y', 'z', 'w']): # for each variable
      draw_pointer(im_final, actuals[var_idx,frame_idx], var, -1, frame_h, frame_w) # GT
      for model_idx in range(3): # for each model
        draw_pointer(im_final, predictions[model_idx][var_idx,frame_idx], var, model_idx, frame_h, frame_w)

    # --- Common graphics 

    # Scales origins
    cv2.line(im_final, scale_x_orig[0], scale_x_orig[1], color_main, thickness=1)
    cv2.line(im_final, scale_y_orig[0], scale_y_orig[1], color_main, thickness=1)
    cv2.line(im_final, scale_z_orig[0], scale_z_orig[1], color_main, thickness=1)
    cv2.line(im_final, scale_w_orig[0], scale_w_orig[1], color_main, thickness=1)

    # Borders
    cv2.rectangle(im_final, border_x[0], border_x[1], color_main, thickness=1)
    cv2.rectangle(im_final, border_y[0], border_y[1], color_main, thickness=1)
    cv2.rectangle(im_final, border_z[0], border_z[1], color_main, thickness=1)
    cv2.rectangle(im_final, border_w[0], border_w[1], color_main, thickness=1)

    # Set image
    x_offset = int(center_point_x - frame_w / 2)
    y_offset = int(center_point_y - frame_h / 2)
    im_final[y_offset : y_offset+frame_h, x_offset : x_offset+frame_w] = frame
    
    # Crop
    margin_h = int(center_point_y - 327 // 2)
    margin_w = int(center_point_x - 519 // 2)
    crop_img = im_final[margin_h:-margin_h, margin_w:-margin_w]

    # Legend
    cv2.putText(crop_img, '{:05}'.format(frame_idx), (5, 10), font, 0.3, color_main, 1, cv2.LINE_AA) # frame number
    crop_img[legend_x:legend_x+legend.shape[0], legend_y:legend_y+legend.shape[1]] = legend # models legend
    cv2.putText(crop_img, 'X', (20, 65), font, 0.5, color_main, 1, cv2.LINE_AA)
    cv2.putText(crop_img, 'Y', (60, 305), font, 0.5, color_main, 1, cv2.LINE_AA)
    cv2.putText(crop_img, 'Z', (480, 270), font, 0.5, color_main, 1, cv2.LINE_AA)
    cv2.putText(crop_img, 'W', (450, 35), font, 0.5, color_main, 1, cv2.LINE_AA)
    
    # --- Other details

    if details_height > 0:
      cv2.putText(crop_img, 'ciao2', (5, 350), font, 0.5, (0,0,0), 1, cv2.LINE_AA)

    # --- Result
    # plt.imshow(crop_img)
    # plt.show()
    # exit()
    video_writer.write(crop_img)

  
  video_writer.release() # close file video


################################


def simulate_flight(models_paths, models_name,
                    data_folder, data_len, bgs_folder, save_folder):
    
  # --- Parameters

  regression = True
  classification = False
  bg_smoothmask = True
  batch_size = 64

  # Data

  start = 0
  end = start + data_len if data_len is not None else None
  list_files = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
  list_files = list_files[start:end]

  with open(list_files[0], 'br') as first:
      input_shape = pickle.load(first)['image'].shape

  backgrounds = network_utils.load_backgrounds(bgs_folder, bg_smoothmask)
  
  data_generator = network_utils.tfdata_generator(
    list_files, input_shape, batch_size,
    backgrounds, bg_smoothmask, 
    deterministic=True, cache=True, repeat=1
  )

  print('Retrieving ground truth from data...\n')
  data_x, data_y = network_utils.get_dataset_from_tfdata_gen(data_generator)
  
  # Models

  models = []
  models_names = []
  for mp in models_paths:
    _, filename = os.path.split(mp)
    model_name, _ = os.path.splitext(filename)
    model = network_utils.network_create(input_shape, regression, classification, retrain_from_layer=0, view_summary=False)
    model = network_utils.network_compile(model, regression, classification, compute_r2=True)
    model.load_weights(mp, by_name=True)
    print('Network weights restored from', model_name)
    models.append(model)
    models_names.append(model_name)
  print('Models loaded.\n')

  # Naming

  time_str = time.strftime("%Y%m%d_%H%M%S")
  data_name = os.path.split(os.path.dirname(data_folder))[1]
  data_str = '{}(len{})'.format(data_name.replace('_', ''), len(list_files))
  bgs_name = os.path.split(os.path.dirname(bgs_folder))[1] if bgs_folder is not None else 'bg'
  backgrounds_str = '_{}'.format(bgs_name or 'bg') if bgs_folder is not None else ''

  save_name = '{0} - {1}{2}'.format(
      time_str,                                             # 0
      data_str,                                             # 1
      backgrounds_str if len(backgrounds) > 0 else '',      # 2
  )

  # --- Computation

  # predictions = []
  # for i, model in enumerate(models):
  #   print('Computing predictions for model {} ...'.format(models_names[i]))
  #   pred = model.predict(data_generator)
  #   pred = np.squeeze(np.array(pred))
  #   predictions.append(pred)
  # print('Predictions completed.\n')

  # TODO fix making runtime predictions
  # # np.save('C:/Users/96mar/Desktop/meeting_dario/videos/preds.npy', predictions)
  # # exit()
  predictions = np.load('C:/Users/96mar/Desktop/meeting_dario/videos/preds.npy')
  print('Predictions loaded.\n')

  save_path = os.path.join(save_folder, '{}.avi'.format(save_name))
  images = (255 - data_x).astype(np.uint8)
  actuals = data_y

  print('Making video ...')
  video_multi_predictions(save_path, images, actuals, predictions)
  print('Video saved to', save_path)


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

  debug_data_len = 1024

  parser = argparse.ArgumentParser(description='Prediction with the given models on some sorted dataset for simulating and visualizing a flight.')
  parser.add_argument('gpu_number', type=int, help='number of the GPU to use') # required
  parser.add_argument("--models_paths", nargs="+", type=file_path, default=[], metavar='MP')
  parser.add_argument('--models_name', type=str, metavar='MN', help='name/identifier of the chosen models set')
  parser.add_argument("--data_folder", type=dir_path)
  parser.add_argument('--data_len', type=int, default=None, metavar='DL', help='max number of samples in the dataset (default = entire dataset, debug = {})'.format(debug_data_len))
  parser.add_argument('--bgs_folder', type=dir_path, metavar='BGF', help='path to backgrounds folder, treaten recursively (default = no background replacement)')
  # parser.add_argument('--bgs_name', type=str, metavar='BGN', help='name/identifier of the chosen backgrounds set, just used for naming purposes')
  # parser.add_argument('--save', action='store_true', help='specify the argument if you want to save evaluation metrics')
  parser.add_argument('--save_folder', type=dir_path, metavar='SVF', help='path where to save evaluation metrics')
  parser.add_argument('--debug', action='store_true', help='if the argument is specified, some parameters are set (overwritten) to debug values')

  parsed_args = parser.parse_args()
  if parsed_args.debug:
    parsed_args.data_len = debug_data_len

  return parsed_args



if __name__ == "__main__":

  ## --- Args
  
  args = get_args()
  print('\nGIVEN ARGUMENTS: ', args, '\n\n')


  ## --- GPU settings

  network_utils.use_gpu_number(args.gpu_number)


  ## --- Training

  simulate_flight(
    args.models_paths, args.models_name, 
    args.data_folder, args.data_len, args.bgs_folder,
    args.save_folder,
  )
