
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

# ---- TF Data with generator

def data_generator(files_path, augmentation, backgrounds_paths):
  for fn in files_path:
    with open(fn, 'rb') as fp:
      sample = pickle.load(fp)
    sample_img = sample_augmentation(sample, backgrounds_paths) if augmentation and backgrounds_paths is not None else sample['image']
    sample_x, sample_y = sample_preprocessing(sample_img, sample['gt'])
    # sample_x = np.random.random((60, 108, 3))
    # sample_y = list(np.random.random((4)))
    yield sample_x, sample_y

def sample_augmentation(data, backgrounds_paths):
  with open(np.random.choice(backgrounds_paths), 'rb') as fp:
    bg = pickle.load(fp)
  return general_utils.image_augment_background(data['image'], data['mask'], background = bg)

def sample_preprocessing(img, gt):
  x_data = (255 - img).astype(np.float32)
  y_data = gt[0:4]
  return x_data, y_data

# ---- TF Data with map

def tf_parse_input(filename):
  return tf.numpy_function(
    map_parse_input, 
    [filename], 
    [tf.uint8, tf.uint8, tf.float32], 
    'map_parse_input')

def map_parse_input(filename):
  with open(filename, 'rb') as fp:
    sample = pickle.load(fp)
  # return np.random.random((60,108,3)), np.random.random((60,108)), np.random.random((4))
  return sample['image'].astype('uint8'), sample['mask'].astype('uint8'), sample['gt'].astype('float32')

def tf_augmentation(img, mask, gt, augmentation, backgrounds_paths):
  return tf.numpy_function(
    map_augmentation, 
    [img, mask, gt, augmentation, backgrounds_paths or []], 
    [tf.uint8, tf.float32], 
    'map_augmentation')

def map_augmentation(img, mask, gt, augmentation, backgrounds_paths):
  if augmentation:
    if len(backgrounds_paths) > 0:
      with open(np.random.choice(backgrounds_paths), 'rb') as fp:
        bg = pickle.load(fp)
        img = general_utils.image_augment_background_minimal(img, mask, bg.astype('uint8'))
  return img, gt

def tf_preprocessing(img, gt):
  return tf.numpy_function(
    map_preprocessing, 
    [img, gt], 
    [tf.float32, tf.float32], 
    'map_preprocessing')
  
def map_preprocessing(img, gt):
  x_data = (255 - img).astype(np.float32) # TODO this is not the same as Dario's one (?)
  y_data = gt[0:4].astype(np.float32)
  return x_data, y_data

# ---- end

def network_train_generator(model, input_size, data_files, 
                            regression, classification, augmentation, bgs_paths,
                            batch_size = 64, epochs = 30, verbose = 2,
                            validation_split = 0.3, validation_shuffle = True,
                            use_lr_reducer = True, use_early_stop = False, 
                            use_profiler = False, profiler_dir = '.\\logs', time_train = True):

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
  
  # generator_train = My_Batch_Generator(data_files_train, batch_size, regression, classification, augmentation, bgs_paths)
  # generator_valid = My_Batch_Generator(data_files_valid, batch_size, regression, classification, augmentation, bgs_paths)

  # generator_train = tf.data.Dataset.from_generator(
  #   lambda: data_generator(data_files_train, augmentation, bgs_paths), 
  #   output_types=(tf.float32, tf.float32), output_shapes=(input_size, (8 if regression and classification else 4))
  # )
  # generator_train = generator_train.batch(batch_size, drop_remainder=True)
  # generator_train = generator_train.prefetch(5)
  # # generator_train = generator_train.cache()

  # generator_valid = tf.data.Dataset.from_generator(
  #   lambda: data_generator(data_files_valid, augmentation, bgs_paths), 
  #   output_types=(tf.float32, tf.float32), output_shapes=(input_size, (8 if regression and classification else 4))
  # )
  # generator_valid = generator_valid.batch(batch_size, drop_remainder=True)
  # generator_valid = generator_valid.prefetch(5)
  # # generator_valid = generator_valid.cache()

  prefetch = True
  prefetch_buffer = tf.data.experimental.AUTOTUNE
  map_parallel = tf.data.experimental.AUTOTUNE
  map_deter = False

  generator_train = tf.data.Dataset.from_tensor_slices(data_files_train)
  generator_train = generator_train.map(tf_parse_input, map_parallel, map_deter)
  generator_train = generator_train.map(lambda img, mask, gt : tf_augmentation(img, mask, gt, augmentation, bgs_paths), map_parallel, map_deter)
  generator_train = generator_train.map(lambda img, gt : tf_preprocessing(img, gt), map_parallel, map_deter)
  generator_train = generator_train.batch(batch_size, drop_remainder=True)

  generator_valid = tf.data.Dataset.from_tensor_slices(data_files_valid)
  generator_valid = generator_valid.map(tf_parse_input, map_parallel, map_deter)
  generator_valid = generator_valid.map(lambda img, mask, gt : tf_augmentation(img, mask, gt, augmentation, bgs_paths), map_parallel, map_deter)
  generator_valid = generator_valid.map(lambda img, gt : tf_preprocessing(img, gt), map_parallel, map_deter)
  generator_valid = generator_valid.batch(batch_size, drop_remainder=True)
  
  if prefetch:
    generator_train = generator_train.prefetch(prefetch_buffer)
    generator_valid = generator_valid.prefetch(prefetch_buffer)
    # generator_train = generator_train.apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=2))
    # generator_valid = generator_valid.apply(tf.data.experimental.prefetch_to_device(device='/gpu:0', buffer_size=2))

  # --- Training

  if time_train:
    start_time = time.monotonic()

  history = model.fit(
      x = generator_train,
      # steps_per_epoch = int(len(data_files_train) // batch_size),
      validation_data = generator_valid,
      # validation_steps = int(len(data_files_valid) // batch_size),
      epochs = epochs,
      callbacks = callbacks,
      verbose = verbose
  )

  if time_train:
    print('\nTraining time: {:.2f} minutes\n'.format((time.monotonic() - start_time)/60))
  
  # # move before training
  # # for profiling non-tf operations, see https://www.tensorflow.org/guide/profiler#events and disable tensorboard callback
  # # this only works on tf version >= 2.3, so cannot be tested in windows (https://github.com/tensorflow/tensorflow/issues/38518#issuecomment-686790002)
  # if use_profiler:
  #   tf.profiler.experimental.start(profiler_dir, tf.profiler.experimental.ProfilerOptions(host_tracer_level=2, python_tracer_level=1, device_tracer_level=1))
  # # move after training
  # if use_profiler:
  #   tf.profiler.experimental.stop()

  return model, history


# ---- Old generator

def maskrcnn_transform_networkdata(images, actuals):
  image_size = images[0].shape
  x_data = 255 - images
  x_data = np.vstack(x_data[:]).astype(np.float32)
  x_data = np.reshape(x_data, (-1, image_size[0], image_size[1], image_size[2]))
  # x_data = images
  
  yr = np.transpose(actuals[:,0:4])     # shape (regr_variables, samples) (4, ?)
  cat = to_categorical(actuals[:,4:8])  # shape (samples, class_variables, categorical) (?, 4, 3)
  yc = np.transpose(cat, (1, 0, 2))     # shape (class_variables, samples, categorical) (4, ?, 3)
  y_data = [yr[0], yr[1], yr[2], yr[3], yc[0], yc[1], yc[2], yc[3]]
  # y_data = [yr[0], yr[1], yr[2], yr[3], 0, 0, 0, 0]

  return x_data, y_data

def data_loading(filenames):
  loaded = [np.load(fn, allow_pickle=True) for fn in filenames]
  return np.array(loaded)

def data_augmentation(data, backgrounds_paths):
  if backgrounds_paths is not None:
    for frame in data:
      with open(np.random.choice(backgrounds_paths), 'rb') as fp:
        bg = pickle.load(fp)
        frame['image'] = general_utils.image_augment_background(frame['image'], frame['mask'], background = bg)
        # frame['image'] = general_utils.image_augment_background_minimal(frame['image'], frame['mask'].astype('uint8'), background = bg)
  return data

def data_preprocessing(data, target_slicing):
  images = np.array([d['image'] for d in data])
  actuals = np.array([d['gt'] for d in data]) # https://stackoverflow.com/a/46317786/10866825
  data_x, data_y = maskrcnn_transform_networkdata(images, actuals)
  return data_x, data_y[target_slicing]

class My_Batch_Generator(tf.keras.utils.Sequence):
  
  def __init__(self, files, batch_size, regression, classification, augmentation, backgrounds_paths):
    if not regression and not classification:
      raise ValueError('At least one between parameter `regression` and `classification` must be True.')
    self.files = files
    self.batch_size = batch_size
    self.augmentation = augmentation
    self.backgrounds_paths = backgrounds_paths
    self.regression = regression
    self.classification = classification
    self.target_slicing = slice(0,8) if regression and classification else (slice(0,4) if regression else slice(4,8))
    
  def __len__(self):
    return (np.ceil(len(self.files) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx):
    batch_files = self.files[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_data = data_loading(batch_files)
    batch_augmented = data_augmentation(batch_data, self.backgrounds_paths) if self.augmentation else batch_data
    batch_x, batch_y = data_preprocessing(batch_augmented, self.target_slicing)
    # # synthetic data
    # batch_x = np.random.random((self.batch_size, 60, 108, 3))
    # batch_y = list(np.random.random((4, self.batch_size)))
    return batch_x, batch_y


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