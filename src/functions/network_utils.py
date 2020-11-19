
################################################################
############ IMPORTS
#################

import math
import os
import time
import errno
import random
import sys
import gc
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

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# # -- for GradCAM
# from tf_keras_vis.utils import normalize
# from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
# from matplotlib import cm


# -- import other files
from . import general_utils





################################################################
############ VARIABLES
#################






################################################################
############ FUNCTIONS
#################


### --------------------- STRUCTURE --------------------- ###

def network_create(input_size, regression, classification, initial_weights = None, retrain_from_layer = None, view_summary = True, view_plot = False):
  
  if not regression and not classification:
    raise ValueError("At least one between parameter `regression` and `classification` must be True.")

  # --- Network architecture

  # input
  input_img = Input(shape=(input_size[0], input_size[1], input_size[2]), name = 'input_1')

  # start resnet
  conv_1 = Conv2D(64, kernel_size=(7,7), strides=(2,2), padding='same', name = 'conv2d_1')(input_img)
  batch_1 = BatchNormalization(name = 'batch_normalization_1')(conv_1)
  activ_1 = Activation('relu', name = 'activation_1')(batch_1)
  pool_1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same', name = 'max_pooling2d_1')(activ_1)

  # block 1
  conv_2 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', name = 'conv2d_2')(pool_1)
  batch_2 = BatchNormalization(name = 'batch_normalization_2')(conv_2)
  activ_2 = Activation('relu', name = 'activation_2')(batch_2)
  conv_3 = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', name = 'conv2d_3')(activ_2)
  add_1 = Add(name = 'add_1')([conv_3, pool_1])

  # block 2
  batch_3 = BatchNormalization(name = 'batch_normalization_3')(add_1)
  activ_3 = Activation('relu', name = 'activation_3')(batch_3)
  conv_4 = Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', name = 'conv2d_4')(activ_3)
  batch_4 = BatchNormalization(name = 'batch_normalization_4')(conv_4)
  activ_4 = Activation('relu', name = 'activation_4')(batch_4)
  conv_5 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', name = 'conv2d_5')(activ_4)
  conv_6 = Conv2D(128, kernel_size=(1,1), strides=(2,2), padding='valid', name = 'conv2d_6')(add_1)
  add_2 = Add(name = 'add_2')([conv_5, conv_6])

  # block 3
  batch_5 = BatchNormalization(name = 'batch_normalization_5')(add_2)
  activ_5 = Activation('relu', name = 'activation_5')(batch_5)
  conv_7 = Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', name = 'conv2d_7')(activ_5)
  batch_6 = BatchNormalization(name = 'batch_normalization_6')(conv_7)
  activ_6 = Activation('relu', name = 'activation_6')(batch_6)
  conv_8 = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', name = 'conv2d_8')(activ_6)
  conv_9 = Conv2D(256, kernel_size=(1,1), strides=(2,2), padding='valid', name = 'conv2d_9')(add_2)
  add_3 = Add(name = 'add_3')([conv_8, conv_9])

  # end resnet
  batch_7 = BatchNormalization(name = 'batch_normalization_7')(add_3)
  activ_7 = Activation('relu', name = 'activation_7')(batch_7)
  pool_2 = AveragePooling2D(pool_size = (4, 7), strides = (1, 1), padding = 'valid', name = 'average_pooling2d_1')(activ_7)

  # dense
  flatten_1 = Flatten(name = 'flatten_1')(pool_2)
  dense_1 = (Dense(256, activation='relu', name="1_dense"))(flatten_1)
  dense_2 = (Dense(128, activation='relu', name="2_dense"))(dense_1)

  # targets
  y_0 = (Dense(1, activation='linear', name=general_utils.variables_names[0]))(dense_2)
  y_1 = (Dense(1, activation='linear', name=general_utils.variables_names[1]))(dense_2)
  y_2 = (Dense(1, activation='linear', name=general_utils.variables_names[2]))(dense_2)
  y_3 = (Dense(1, activation='linear', name=general_utils.variables_names[3]))(dense_2)
  y_4 = (Dense(3, activation='softmax', name=general_utils.variables_names[4]))(dense_2)
  y_5 = (Dense(3, activation='softmax', name=general_utils.variables_names[5]))(dense_2)
  y_6 = (Dense(3, activation='softmax', name=general_utils.variables_names[6]))(dense_2)
  y_7 = (Dense(3, activation='softmax', name=general_utils.variables_names[7]))(dense_2)

  outputs = []
  if regression:
    outputs.extend([y_0, y_1, y_2, y_3])
  if classification:
    outputs.extend([y_4, y_5, y_6, y_7])

  # model
  flat_model = Model(inputs = input_img, outputs = outputs) # MODEL

  # --- Restore weights from initial ones 

  if initial_weights is not None:
    for layer_name, weights in initial_weights.items(): # starts at 2 for skipping inputs and nested model
      try:
        flat_model.get_layer(layer_name).set_weights(weights)
      except ValueError: # get_layer raises ValueError is a layer does not exist
        # for each variable, the respective model only contains the associated variable (so the other outputs will be missing)
        print(layer_name, 'layer not found, skipping')
        continue
    print('Restored network initial weights')
  else:
    print('Network initialized with random weights')

  # --- Set trainable layers

  if retrain_from_layer is not None:
    non_trainable_until = retrain_from_layer # non-trainable until specified layer
  elif classification:
    non_trainable_until = -4 # classification layers always have to be trained
  else: # regression
    non_trainable_until = len(flat_model.layers) # nothing will be trainable

  for layer in flat_model.layers[:non_trainable_until]:
    layer.trainable =  False

  # --- Result 

  if view_plot: 
    plot_model(flat_model, show_shapes = True, expand_nested = True)
  if view_summary:
    flat_model.summary()
    print('Please note that the network is non-trainable until the {} layer.'.format(non_trainable_until))

  return flat_model


def network_export_weights(source_model, dest_folder, dest_name):
  weights = {}

  for layer in source_model.layers[1:]: # skip input layer
    if isinstance(layer, tf.keras.Model):
      for nest_layer in layer.layers:
        weights[nest_layer.name] = nest_layer.get_weights()
    else:
        weights[layer.name] = layer.get_weights()

  print(weights.keys())

  with open(os.path.join(dest_folder, dest_name + '.pickle'), 'wb') as fp:
    pickle.dump(weights, fp)


### ------------------ SIMPLE TRAINING ------------------ ###


def network_train(model, data_x, data_y, regression, classification, 
                  batch_size = 64, epochs = 30, verbose = 2,
                  validation_split = 0.3, validation_shuffle = True, 
                  use_lr_reducer = True, use_early_stop = False):

  # --- Model settings

  loss = []
  metrics = []

  if regression:
    loss.extend(['mean_absolute_error'] * 4)
    metrics.append('mse')
  if classification:
    loss.extend(['categorical_crossentropy'] * 4)
    metrics.append('accuracy')

  model.compile(loss=loss,
                metrics=metrics,
                optimizer='adam')

  callbacks = []

  if use_lr_reducer:
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=4, min_lr=0.1e-6)
    callbacks.append(lr_reducer)

  if use_early_stop:
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1)
    callbacks.append(early_stop)

  # --- Train/Validation split
  
  n_val = int(len(data_x) * validation_split)
  ix_val, ix_tr = np.split(np.random.permutation(len(data_x)), [n_val])
  
  x_valid = data_x[ix_val, :]
  x_train = data_x[ix_tr, :]
  y_valid = [var[ix_val] for var in data_y]
  y_train = [var[ix_tr] for var in data_y]

  # --- Training

  history = model.fit(
      x = x_train,
      y = y_train,
      batch_size = batch_size,
      epochs = epochs,
      validation_data = (x_valid, y_valid),
      # validation_split = validation_split,
      callbacks = callbacks,
      shuffle = True,
      verbose = verbose
  )

  return model, history


### ------------------ GENERATOR TRAINING ------------------ ###

def tf_parse_input(filename):
  return tf.numpy_function(
    map_parse_input, 
    [filename], 
    [tf.uint8, tf.uint8, tf.float64], 
    'map_parse_input')

def map_parse_input(filename):
  with open(filename, 'rb') as fp:
    sample = pickle.load(fp)
  return sample['image'].astype('uint8'), sample['mask'].astype('uint8'), sample['gt'].astype('float64')

def tf_augmentation(img, mask, gt, augmentation, backgrounds):
  if backgrounds is None:
    backgrounds = [] # None is not convertible to Tensor
  return tf.numpy_function(
    map_augmentation, 
    [img, mask, gt, augmentation, backgrounds], 
    [tf.uint8, tf.float64], 
    'map_augmentation')

def map_augmentation(img, mask, gt, augmentation, backgrounds):
  if augmentation:
    if len(backgrounds) > 0:
      bg = backgrounds[np.random.randint(0, len(backgrounds))]
      img = general_utils.image_augment_background_minimal(img, mask, bg)
  return img, gt

def tf_preprocessing(img, gt):
  x, y = tf.numpy_function(
    map_preprocessing, 
    [img, gt], 
    [tf.float32, tf.float64], 
    'map_preprocessing')
  # for multi-output networks, https://datascience.stackexchange.com/a/63937/107722 saved my life 
  y = {'x_pred': y[0], 'y_pred': y[1], 'z_pred': y[2], 'yaw_pred': y[3]}
  return x, y
  
def map_preprocessing(img, gt):
  x_data = (255 - img).astype(np.float32) # TODO this is not the same as Dario's one (?)
  y_data = gt[0:4]
  return x_data, y_data


def network_train_generator(model, input_size, data_files, 
                            regression, classification, augmentation, bgs_paths,
                            batch_size = 64, epochs = 30, verbose = 2,
                            validation_split = 0.3, validation_shuffle = True,
                            use_lr_reducer = True, use_early_stop = False, 
                            use_profiler = False, profiler_dir = '.\\logs', time_train = True):

  backgrounds = np.array([])
  if augmentation and bgs_paths is not None:
    # np.random.shuffle(bgs_paths) # shuffling before reducing backgrounds
    # bgs_paths = bgs_paths[:100] # reducing backgrounds cardinality
    print('Loading {} backgrounds in memory for data augmentation...'.format(len(bgs_paths)))
    backgrounds = np.array(list([general_utils.load_pickle(filepath) for filepath in bgs_paths]))
    
    if backgrounds.dtype == np.float32 or backgrounds.dtype == np.float64:
      backgrounds = (backgrounds * 255).astype('uint8')
      print('Backgrounds converted from float to uint8')
    elif backgrounds.dtype != np.uint8:
      backgrounds = backgrounds.astype('uint8')
      print('Backgrounds converted from unknown dtype to uint8')

    print('Loaded {} backgrounds of shape {}.\n'.format(len(backgrounds), backgrounds.shape[1:]))

  # --- Compilation

  loss = []
  metrics = []

  if regression:
    loss.extend(['mean_absolute_error'] * 4)
    metrics.append('mse')
  if classification:
    loss.extend(['categorical_crossentropy'] * 4)
    metrics.append('accuracy')

  model.compile(loss=loss,
                metrics=metrics,
                optimizer='adam')

  # --- Callbacks

  callbacks = []

  if use_lr_reducer:
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=4, min_lr=0.1e-6)
    callbacks.append(lr_reducer)

  if use_early_stop:
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1)
    callbacks.append(early_stop)

  if use_profiler:
    tensorboard = tf.keras.callbacks.TensorBoard(profiler_dir, histogram_freq=1, profile_batch = '5,20')
    callbacks.append(tensorboard)

  # --- Train/Validation split
    
  from sklearn.model_selection import train_test_split
  data_files_train, data_files_valid = train_test_split(data_files, test_size=validation_split, 
                                                        shuffle=validation_shuffle, random_state=1)

  # --- Generator

  prefetch = True
  prefetch_buffer = tf.data.experimental.AUTOTUNE
  map_parallel = tf.data.experimental.AUTOTUNE
  map_deter = False # map function parameter deterministic=False improves performances

  generator_train = tf.data.Dataset.from_tensor_slices(data_files_train)
  generator_train = generator_train.map(tf_parse_input, map_parallel, map_deter)
  generator_train = generator_train.map(lambda img, mask, gt : tf_augmentation(img, mask, gt, augmentation, backgrounds), map_parallel, map_deter)
  generator_train = generator_train.map(lambda img, gt : tf_preprocessing(img, gt), map_parallel, map_deter)
  generator_train = generator_train.batch(batch_size, drop_remainder=True)

  generator_valid = tf.data.Dataset.from_tensor_slices(data_files_valid)
  generator_valid = generator_valid.map(tf_parse_input, map_parallel, map_deter)
  generator_valid = generator_valid.map(lambda img, mask, gt : tf_augmentation(img, mask, gt, augmentation, backgrounds), map_parallel, map_deter)
  generator_valid = generator_valid.map(lambda img, gt : tf_preprocessing(img, gt), map_parallel, map_deter)
  generator_valid = generator_valid.batch(batch_size, drop_remainder=True)
  
  if prefetch:
    generator_train = generator_train.prefetch(prefetch_buffer)
    generator_valid = generator_valid.prefetch(prefetch_buffer)

  # --- Training

  print('Training started...')
  
  if time_train:
    start_time = time.monotonic()

  history = model.fit(
      x = generator_train,
      validation_data = generator_valid,
      epochs = epochs,
      callbacks = callbacks,
      verbose = verbose
  )

  if time_train:
    print('\nTraining time: {:.2f} minutes\n'.format((time.monotonic() - start_time)/60))

  return model, history


### --------------------- METRICS --------------------- ###


def network_stats(history, regression, classification, view, save, save_folder = '', save_name = ''):
  
  if classification:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
  else:
    fig, ax1 = plt.subplots(1, 1, figsize=(10,6))

  var_str = 'all_class'
  fig.suptitle(save_name)

  # - Loss

  ax1.plot(history.history['loss'], 'k--', label='Train Loss')
  ax1.plot(history.history['val_loss'], 'k', label='Valid Loss')
  ax1.legend(loc='upper right')
  ax1.set_xlabel('Epoch')
  ax1.set_title(var_str + ' training and validation Loss')

  # - Accuracy
  
  if classification:
    ax2.plot(history.history['x_class_accuracy'], 'r--', label='x_class train Accuracy')
    ax2.plot(history.history['val_x_class_accuracy'], 'r', label='x_class valid Accuracy')
    ax2.plot(history.history['y_class_accuracy'], 'g--', label='y_class train Accuracy')
    ax2.plot(history.history['val_y_class_accuracy'], 'g', label='y_class valid Accuracy')
    ax2.plot(history.history['z_class_accuracy'], 'b--', label='z_class train Accuracy')
    ax2.plot(history.history['val_z_class_accuracy'], 'b', label='z_class valid Accuracy')
    ax2.plot(history.history['w_class_accuracy'], 'y--', label='w_class train Accuracy')
    ax2.plot(history.history['val_w_class_accuracy'], 'y', label='w_class valid Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(var_str + ' training and validation Accuracy')
 
  if save:
    general_utils.create_folder_if_not_exist(save_folder)
    figname = os.path.join(save_folder, '{0} - v1_{1}_metrics.png'.format(save_name, var_str))
    fig.savefig(figname, bbox_inches='tight')

  if view:
    plt.show()
  else:
    plt.close()

  print('Model stats computed')


### --------------------- SAVE --------------------- ###


def network_save(model, folder, name, save_plot = False):
  '''
    Saves the model as .h5 file, accordingly to the input parameters.
    File name will be in the format '{name} - v1_{var_name}_model.h5'.

    Parameters:
        model (keras.engine.functional): Model to be saved
        folder (str): Path in which the model has to be saved
        name (str): Used for naming purposes, easy understandable identifier for the .h5 file name
        var_index (int): Used for naming purposes, must be coherent with the `network_create` function same parameter
        save_plot (bool): If True, also graphical representation (.png) of the model is saved together with the model itself

    Returns:
        model_path (str): Complete path of the saved .h5 file
  '''
  
  general_utils.create_folder_if_not_exist(folder)
  model_path = os.path.join(folder, '{} - v1_model'.format(name))

  if save_plot:
    plot_model(model, to_file = model_path + '.png', show_shapes = True, expand_nested = False)
  
  model_h5 = model_path + '.h5'
  model.save(model_h5)
  print('Model saved in', model_h5)

  return model_h5