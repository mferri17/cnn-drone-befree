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

def simulate_flight(models_paths, models_name,
                    data_folder, data_len,
                    bgs_folder, bgs_name,
                    save, save_folder):
    
    # --- Parameters

    regression = True
    classification = False
    bg_smoothmask = True
    batch_size = 64

    # Data

    list_files = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
    list_files = list_files[:data_len]

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
    models_str = '{}(len{})'.format(models_name, len(models))
    data_name = os.path.split(os.path.dirname(data_folder))[1]
    data_str = '{}(len{})'.format(data_name.replace('_', ''), len(list_files))
    backgrounds_str = '_{}'.format(bgs_name or 'bg')

    save_name = '{0} {1} - {2}_{3}{4}'.format(
        time_str,                                             # 0
        socket.gethostname().replace('.','_'),                # 1
        models_str,                                           # 2
        data_str,                                             # 3
        backgrounds_str if len(backgrounds) > 0 else '',      # 4
    )

    # --- Computation

    # predictions = []
    # for i, model in enumerate(models):
    #   print('Computing predictions for model {} ...'.format(models_names[i]))
    #   pred = model.predict(data_generator)
    #   pred = np.squeeze(np.array(pred))
    #   predictions.append(pred)
    # print('Predictions completed.\n')
    # # np.save('C:/Users/96mar/Desktop/meeting_dario/videos/preds.npy', predictions)
    # # exit()
    
    predictions = np.load('C:/Users/96mar/Desktop/meeting_dario/videos/preds.npy')
    print('Predictions loaded.\n')

    path1 = "C:/Users/96mar/Desktop/meeting_dario/videos/prova1.avi"
    path2 = "C:/Users/96mar/Desktop/meeting_dario/videos/prova2.avi"
    images = (255 - data_x).astype(np.uint8) 
    actuals = data_y

    print('Making videos...')
    # video_simple(path1, images)
    frame_composer_multi_pred(path2, images, actuals, predictions)

    # --- Saving

    print('\n', save_name)
    if save:
      pass


def video_simple(path, images):
  video_width = 518
  video_height = 326
  video_fps = 30
  video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (video_width, video_height))
  
  for i, frame in enumerate(images[:200]):
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


def frame_composer_multi_pred(path, images, actuals, predictions):
  video_width = 518
  video_height = 326
  video_fps = 25
  legend_x, legend_y = (50, 50)
  legend = cv2.imread('C:/Users/96mar/Desktop/meeting_dario/videos/legend.png')
  video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), video_fps, (video_width, video_height))

  # x on the left 
  # y on the bottom
  # z on the right
  # w on the top
  for i, frame in enumerate(images[:200]):
    
    frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)

    v1_pred = predictions[0][:,i]
    v2_pred = predictions[1][:,i]
    v3_pred = predictions[2][:,i]
    y = actuals[:,i]
    # velocity = self.velocities[i]

    # data_area = (np.ones((330, 520, 3)) * 255).astype(np.uint8)
    data_area = (np.ones((480, 640, 3)) * 255).astype(np.uint8)
    black_color = (0, 0, 0)

    center_point_x = 320
    # center_point_x = int(520/2)
    center_point_y = 240
    # center_point_y = int(330/2)

    x_offset = int(center_point_x - frame.shape[1] / 2)
    y_offset = int(center_point_y - frame.shape[0] / 2)
    im_final = data_area

    draw_pointer(im_final, -y[0], 1, '0', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, -y[1], 2, '0', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, y[2], 3, '0', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, y[3], 4, '0', frame.shape[0], frame.shape[1])

    cv2.line(im_final, (int(center_point_x - frame.shape[1] / 2), center_point_y), (int(center_point_x - frame.shape[1] / 2 - 30), center_point_y), black_color, thickness=1)
    cv2.line(im_final, (int(center_point_x + frame.shape[1] / 2), center_point_y), (int(center_point_x + frame.shape[1] / 2 + 30), center_point_y), black_color, thickness=1)
    cv2.line(im_final, (center_point_x, int(center_point_y - frame.shape[0] / 2)), (center_point_x, int(center_point_y - frame.shape[0] / 2 - 30)), black_color, thickness=1)
    cv2.line(im_final, (center_point_x, int(center_point_y + frame.shape[0] / 2)), (center_point_x, int(center_point_y + frame.shape[0] / 2 + 30)), black_color, thickness=1)

    draw_line_datapoint(im_final, -v1_pred[0], 1, '1', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, -v2_pred[0], 1, '2', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, -v3_pred[0], 1, '3', frame.shape[0], frame.shape[1])

    draw_line_datapoint(im_final, -v1_pred[1], 2, '1', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, -v2_pred[1], 2, '2', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, -v3_pred[1], 2, '3', frame.shape[0], frame.shape[1])

    draw_line_datapoint(im_final, v1_pred[2], 3, '1', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, v2_pred[2], 3, '2', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, v3_pred[2], 3, '3', frame.shape[0], frame.shape[1])

    draw_line_datapoint(im_final, v1_pred[3], 4, '1', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, v2_pred[3], 4, '2', frame.shape[0], frame.shape[1])
    draw_line_datapoint(im_final, v3_pred[3], 4, '3', frame.shape[0], frame.shape[1])

    draw_pointer(im_final, -v1_pred[0], 1, '1', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, -v2_pred[0], 1, '2', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, -v3_pred[0], 1, '3', frame.shape[0], frame.shape[1])

    draw_pointer(im_final, -v1_pred[1], 2, '1', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, -v2_pred[1], 2, '2', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, -v3_pred[1], 2, '3', frame.shape[0], frame.shape[1])

    draw_pointer(im_final, v1_pred[2], 3, '1', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, v2_pred[2], 3, '2', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, v3_pred[2], 3, '3', frame.shape[0], frame.shape[1])

    draw_pointer(im_final, v1_pred[3], 4, '1', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, v2_pred[3], 4, '2', frame.shape[0], frame.shape[1])
    draw_pointer(im_final, v3_pred[3], 4, '3', frame.shape[0], frame.shape[1])

    cv2.rectangle(im_final,
                  (int(center_point_x - frame.shape[1] / 2), int(center_point_y - frame.shape[0] / 2)),
                  (int(center_point_x - frame.shape[1] / 2 - 30), int(center_point_y + frame.shape[0] / 2)),
                  black_color, thickness=1)
    cv2.rectangle(im_final,
                  (int(center_point_x + frame.shape[1] / 2), int(center_point_y - frame.shape[0] / 2)),
                  (int(center_point_x + frame.shape[1] / 2 + 30), int(center_point_y + frame.shape[0] / 2)),
                  black_color, thickness=1)

    cv2.rectangle(im_final,
                  (int(center_point_x - frame.shape[1] / 2), int(center_point_y + frame.shape[0] / 2)),
                  (int(center_point_x + frame.shape[1] / 2), int(center_point_y + frame.shape[0] / 2 + 30)),
                  black_color, thickness=1)
    cv2.rectangle(im_final,
                  (int(center_point_x - frame.shape[1] / 2), int(center_point_y - frame.shape[0] / 2)),
                  (int(center_point_x + frame.shape[1] / 2), int(center_point_y - frame.shape[0] / 2 - 30)),
                  black_color, thickness=1)
                  
    data_area[y_offset:y_offset + frame.shape[0], x_offset:x_offset + frame.shape[1]] = frame

    # draw_speed_square(im_final, velocity[0], velocity[1], frame.shape[0], frame.shape[1])

    margin_h = int(center_point_y - 327 // 2)
    margin_w = int(center_point_x - 519 // 2)

    crop_img = im_final[margin_h:-margin_h, margin_w:-margin_w]
    crop_img[legend_x:legend_x+legend.shape[0], legend_y:legend_y+legend.shape[1]] = legend
    
    video_writer.write(crop_img)

  video_writer.release()



# BGR
gt_color = (0, 0, 0)
v1_color = (0, 255, 0)
v2_color = (255, 0, 0)
v3_color = (0, 0, 255)


def triangle_pointer(img, p, rotation, color_):
    # width = 520
    width = 640
    # height = 330
    height = 480
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
    img[p[1] - 20:p[1] + 20, p[0] - 20:p[0] + 20] = rotated.astype(np.uint8)


def square_pointer(img, p, color_, w=640, h=480):
    width = w
    height = h
    p[0] = max(width - 10 if p[0] >= width-5 else p[0], 10)
    p[1] = max(height - 10 if p[1] >= height-5 else p[1], 10)
    point_im = np.ones((10, 10, 3)) * 255
    p_1 = (0, 0)
    p_2 = (10, 10)
    cv2.rectangle(point_im, p_1, p_2, color=color_, thickness=-1)
    img[p[1] - 5:p[1] + 5, p[0] - 5:p[0] + 5] = point_im.astype(np.uint8)


def draw_line_datapoint(img, value, graph, model, frame_vert_dim, frame_horiz_dim):
    center_x = 320
    center_y = 240
    bar_width = 30
    if model == '0':
        color_ = gt_color
    elif model == '1':  # lix_x SX
        color_ = v1_color
    elif model == '2':
        color_ = v2_color
    elif model == '3':
        color_ = v3_color
    else:
        color_ = (0, 0, 100)
    if graph == 1:  # lin x SX
        p_x = int(center_x - bar_width - frame_horiz_dim / 2)
        p_y = int(center_y + frame_vert_dim / 2 * (-value * 1.5))
        if model == '0':
            # triangle_pointer(img, (p_x, p_y), delta_rot, color_)
            pass
        else:
            cv2.line(img, (p_x, p_y), (p_x + 30, p_y), color_, thickness=2)
            # square_pointer(img, (p_x + 5 + (10 * (int(model) - 1)), p_y), color_)
    elif graph == 2:  # lin y
        p_x = int(center_x + frame_horiz_dim / 2 * (-value * 1.5))
        p_y = int(center_y + bar_width + frame_vert_dim / 2)
        if model == '0':
            # triangle_pointer(img, (p_x, p_y), 90 + delta_rot, color_)
            pass
        else:
            cv2.line(img, (p_x, p_y), (p_x, p_y - 30), color_, thickness=2)
            # square_pointer(img, (p_x, p_y - 5 - (10 * (int(model) - 1))), color_)
    elif graph == 3:  # lin z
        p_x = int(center_x + bar_width + frame_horiz_dim / 2)
        p_y = int(center_y + frame_vert_dim / 2 * (-value * 1.5))
        if model == '0':
            # triangle_pointer(img, (p_x, p_y), delta_rot - 180, color_)
            pass
        else:
            cv2.line(img, (p_x, p_y), (p_x - 30, p_y), color_, thickness=2)
            # square_pointer(img, (p_x - 5 - (10 * (int(model) - 1)), p_y), color_)
    elif graph == 4:  # ang z
        p_x = int(center_x + frame_horiz_dim / 2 * (-value * 1.5))  # TODO maybe revers direction
        p_y = int(center_y - bar_width - frame_vert_dim / 2)
        if model == '0':
            # triangle_pointer(img, (p_x, p_y), -90 + delta_rot, color_)
            pass
        else:
            cv2.line(img, (p_x, p_y), (p_x, p_y + 30), color_, thickness=2)
            # square_pointer(img, (p_x, p_y + 5 + (10 * (int(model) - 1))), color_)
    else:
        return None


def draw_pointer(img, value, graph, model, frame_vert_dim, frame_horiz_dim, c_x=320, c_y=240, width=640, height=480):
    center_x = c_x
    center_y = c_y
    bar_width = 30
    if model == '0':
        color_ = gt_color
        delta_rot = 0
    elif model == '1':  # lix_x SX
        color_ = v1_color
        delta_rot = 0
    elif model == '2':
        color_ = v2_color
        delta_rot = 0
    elif model == '3':
        color_ = v3_color
        delta_rot = 0
    else:
        color_ = (0, 0, 100)
        delta_rot = 0
    if graph == 1:  # lin x SX
        p_x = int(center_x - bar_width - frame_horiz_dim / 2)
        p_y = int(center_y + frame_vert_dim / 2 * ((-value-1.5) * 1.5))
        if model == '0':
            triangle_pointer(img, [p_x, p_y], delta_rot, color_)
        else:
            # cv2.line(img, (p_x, p_y), (p_x + 30, p_y), color_, thickness=2)
            square_pointer(img, [p_x + 5 + (10 * (int(model) - 1)), p_y], color_, w=width, h=height)
    elif graph == 2:  # lin y
        p_x = int(center_x + frame_horiz_dim / 2 * (-value * 1.5))
        p_y = int(center_y + bar_width + frame_vert_dim / 2)
        if model == '0':
            triangle_pointer(img, [p_x, p_y], 90 + delta_rot, color_)
        else:
            # cv2.line(img, (p_x, p_y), (p_x, p_y - 30), color_, thickness=2)
            square_pointer(img, [p_x, p_y - 5 - (10 * (int(model) - 1))], color_, w=width, h=height)
    elif graph == 3:  # lin z
        p_x = int(center_x + bar_width + frame_horiz_dim / 2)
        p_y = int(center_y + frame_vert_dim / 2 * (-value * 1.5))
        if model == '0':
            triangle_pointer(img, [p_x, p_y], delta_rot - 180, color_)
        else:
            # cv2.line(img, (p_x, p_y), (p_x - 30, p_y), color_, thickness=2)
            square_pointer(img, [p_x - 5 - (10 * (int(model) - 1)), p_y], color_, w=width, h=height)
    elif graph == 4:  # ang z
        p_x = int(center_x + frame_horiz_dim / 2 * (-value * 1.5))
        p_y = int(center_y - bar_width - frame_vert_dim / 2)
        if model == '0':
            triangle_pointer(img, [p_x, p_y], -90 + delta_rot, color_)
        else:
            # cv2.line(img, (p_x, p_y), (p_x, p_y + 30), color_, thickness=2)
            square_pointer(img, [p_x, p_y + 5 + (10 * (int(model) - 1))], color_, w=width, h=height)
    else:
        return None

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

  debug_data_len = 64
  debug_bgs_len = 100

  parser = argparse.ArgumentParser(description='Prediction with the given models on some sorted dataset for simulating and visualizing a flight.')
  parser.add_argument('gpu_number', type=int, help='number of the GPU to use') # required
  parser.add_argument("--models_paths", nargs="+", type=file_path, default=[], metavar='MP')
  parser.add_argument('--models_name', type=str, metavar='MN', help='name/identifier of the chosen models set')
  parser.add_argument("--data_folder", type=dir_path)
  parser.add_argument('--data_len', type=int, default=None, metavar='DL', help='max number of samples in the dataset (default = entire dataset, debug = {})'.format(debug_data_len))
  parser.add_argument('--bgs_folder', type=dir_path, metavar='BGF', help='path to backgrounds folder, treaten recursively (default = no background replacement)')
  parser.add_argument('--bgs_name', type=str, metavar='BGN', help='name/identifier of the chosen backgrounds set, just used for naming purposes')
  parser.add_argument('--save', action='store_true', help='specify the argument if you want to save evaluation metrics')
  parser.add_argument('--save_folder', type=dir_path, metavar='SVF', help='path where to save evaluation metrics')
  parser.add_argument('--debug', action='store_true', help='if the argument is specified, some parameters are set (overwritten) to debug values')

  parsed_args = parser.parse_args()
  if parsed_args.debug:
    parsed_args.data_len = debug_data_len
    parsed_args.bgs_len = debug_bgs_len

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
    args.data_folder, args.data_len,
    args.bgs_folder, args.bgs_name,
    args.save, args.save_folder,
  )
