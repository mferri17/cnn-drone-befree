
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

import tensorflow_probability as tfp
import tensorflow_addons as tfa

# # -- for GradCAM
# from tf_keras_vis.utils import normalize
# from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
# from matplotlib import cm

import albumentations as A

# -- import other files
from . import general_utils





################################################################
############ VARIABLES
#################




################################################################
############ FUNCTIONS
#################


def use_gpu_number(number):
  cuda_visibile_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
  if cuda_visibile_devices is not None:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
  
  gpus = tf.config.experimental.list_physical_devices('GPU')

  if gpus:
    try:
      print('Selected GPU number', number)
      if number < 0:
        tf.config.set_visible_devices([], 'GPU')
      else:
        tf.config.experimental.set_visible_devices(gpus[number], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[number], True) # not immediately allocating the full memory
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs \n')
    except RuntimeError as e:
      print(e) # visible devices must be set at program startup
  else:
      print('No available GPUs \n')


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
    print('Restored network initial weights.\n')
  else:
    print('Network initialized with random weights.\n')

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


def load_backgrounds(backgrounds_folder, bg_smoothmask=False, backgrounds_len=None):

  backgrounds = np.array([])

  if backgrounds_folder is not None:

    paths = general_utils.list_files_in_folder(backgrounds_folder, 'pickle', recursive=True)
    if backgrounds_len is not None:
      np.random.seed(1) # TODO remove seed
      np.random.shuffle(paths) # shuffling before reducing backgrounds
      paths = paths[:backgrounds_len] # reducing backgrounds cardinality
  
    if len(paths) > 0:  
      print('Loading {} backgrounds in memory for data augmentation...'.format(len(paths)))
      backgrounds = np.array(list([general_utils.load_pickle(filepath) for filepath in paths]))
      
      if bg_smoothmask: # need float32 background
        if issubclass(backgrounds.dtype.type, np.integer):
          backgrounds = backgrounds.astype('float32') # it's ok to have 0-255 float since input images for the network are like so

      else: # need uint8 background
        if backgrounds.dtype == np.float32 or backgrounds.dtype == np.float64:
          backgrounds = (backgrounds * 255).astype('uint8')
          print('Backgrounds converted from float to uint8')
        elif backgrounds.dtype != np.uint8:
          backgrounds = backgrounds.astype('uint8')
          print('Backgrounds converted from unknown dtype to uint8')

      print('Loaded {} backgrounds of shape {}.\n'.format(len(backgrounds), backgrounds.shape[1:]))
  
  return backgrounds


def load_noises(noise_folder):
  
  noises = np.array([])
  
  if noise_folder is not None:
    paths = general_utils.list_files_in_folder(noise_folder, 'pickle', recursive=False)

    if len(paths) > 0:  
      print('Loading {} noises in memory for data augmentation...'.format(len(paths)))
      noises = np.array(list([general_utils.load_pickle(filepath) for filepath in paths]))
      print('Loaded {} noises of shape {}.\n'.format(len(noises), noises.shape[1:]))

  return noises


img_counter = 0
def save_img(img):
  global img_counter
  img_counter += 1
  plt.imsave('C:/Temp/venv/images/{:05d}.jpg'.format(img_counter), cv2.resize(img, (600,400)))


def map_parse_input(filename, img_shape):
  img, mask, gt = tf.numpy_function(
    tf_parse_input, 
    [filename], 
    [tf.uint8, tf.uint8, tf.float64], 
    'tf_parse_input')

  img.set_shape(img_shape)
  gt.set_shape(8)
  return img, mask, gt


def tf_parse_input(filename):
  with open(filename, 'rb') as fp:
    sample = pickle.load(fp)
  return sample['image'].astype('uint8'), sample['mask'].astype('uint8'), sample['gt'].astype('float64')


def map_replace_background(img, mask, gt, backgrounds, smooth_mask):
  img_shape = img.shape
  
  if backgrounds.shape[0] > 0: # backgrounds shape is known at graph definition
    valid_shapes = tf.math.reduce_all(tf.math.equal(tf.shape(mask), tf.shape(img)[:-1]))
    
    if valid_shapes: # mask shape is dynamic
      ridx = tf.random.uniform(shape=[], minval=0, maxval=len(backgrounds), dtype=tf.dtypes.int64, seed=1) # TODO remove seed
      bg = backgrounds[ridx]

      if smooth_mask:
        # img, mask and bg must be float
        mask = tfa.image.gaussian_filter2d(tf.cast(mask, tf.float32))
        mask_stack = mask[:,:,np.newaxis] # add 3rd dimension for broadcasting
        img = tf.cast(img, tf.float32) # recasting for noise blending
        img = (mask_stack * img) + ((1-mask_stack) * bg) # Blend
        img = tf.cast(img, tf.uint8) # recasting back
      else:
        # img, mask and bg can be uint8
        mask_stack = mask[:,:,np.newaxis] # add 3rd dimension for broadcasting
        img = (mask_stack * img) + ((1-mask_stack) * bg) # Blend
  
  img.set_shape(img_shape)
  return img, gt


def map_augmentation(img, gt, aug_prob, noises):
  img_shape = img.shape

  # -- numpy augmentations
  if aug_prob > 0:
    img = tf.numpy_function(tf_albumentation, [img, aug_prob], tf.uint8, 'tf_albumentation') # albumentation
    img.set_shape(img_shape)

  # -- native tensorflow augmentations
  if aug_prob > 0:

    # horizontal flip
    if tf.random.uniform([]) < 0.5 * aug_prob:
      img = tf.image.flip_left_right(img)
      gt = tf.convert_to_tensor([gt[0], -gt[1], gt[2], -gt[3]]) # invert Y and YAW
      # TODO add classification handling for gt

    # additional noise
    if noises.shape[0] > 0: # known at graph definition
      if tf.random.uniform([]) < 0.2 * aug_prob:
        distr_tr = tfp.distributions.Triangular(low=0, high=0.75, peak=0.5)
        multiplier = distr_tr.sample() # noise is probabilistically reduced
        by_channel = tf.random.uniform([]) < 0.5 # 50% of cases are applied by channel
        
        ridx = tf.random.uniform(shape=[], minval=0, maxval=len(noises), dtype=tf.dtypes.int64, seed=1) # TODO remove seed
        noise = noises[ridx]

        if by_channel:
          multiplier *= 0.8 # reducing the RGB noise or it is too strong
          noise = tf.image.random_crop(noise, size=img_shape) # crop over the 3 channels
        else:
          noise = tf.image.random_crop(noise, size=[img_shape[0], img_shape[1], 1]) # crop over just 1 channel

        noise = tf.image.flip_left_right(noise) # horizontal flip
        noise = noise * multiplier + 1  # rescaling in (1-multiplier, 1+multiplier)
        img = tf.cast(img, tf.float32) # recasting for noise blending
        img = tf.math.multiply(img, noise)
        img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=255) # multiplying, we may exceed the valid range
        img = tf.cast(img, tf.uint8) # recasting back
      
  # -- result
  return img, gt


def tf_albumentation(img, aug_prob):
  augmenter = A.Compose([
        A.RandomBrightnessContrast(brightness_by_max=True, p=0.75), # 0.77 sec
        A.RandomGamma(p=0.5), # 0.50 sec
        A.CLAHE(p=0.05), # 3.50 sec
        A.Solarize(threshold=(200,250), p=0.2), # 0.81 sec
        A.OneOf([
            A.Equalize(by_channels=False, p=0.5), # 0.97 sec
            A.Equalize(by_channels=True, p=0.5), # 0.97 sec
        ], p=0.1),
        A.RGBShift(p=0.3), # 1.53 sec
        A.OneOf([
            A.ChannelDropout(fill_value=128, p=0.2), # 0.52 sec
            A.ChannelShuffle(p=0.8), # 0.44 sec TODO only apply to the background
        ], p=0.1),
        A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=True, elementwise=True, p=0.05), # 6.89 sec
        A.CoarseDropout(min_holes=20, max_holes=70, min_height=1, max_height=4, min_width=1, max_width=4, p=0.2), # 3.78 sec
        A.ToGray(p=0.05), # 0.34 sec
        A.InvertImg(p=0.05), # 0.36 sec
        A.OneOf([
            A.Blur(blur_limit=4, p=0.5), # 0.58 sec
            A.MotionBlur(blur_limit=6, p=0.5), # 0.97 sec
        ], p=0.05),
    ], p=aug_prob)
  img = augmenter(image=img)['image']
  return img


def map_preprocessing(img, gt):
  # if tf.random.uniform([]) < 0.005:
  #   tf.numpy_function(save_img, [img], [], 'save_img')
  
  # for multi-output networks, https://datascience.stackexchange.com/a/63937/107722 saved my life 
  x = tf.cast((255 - img), tf.float32) # TODO remove inversion, and also cast if already present before
  y = gt[0:4]
  y = {'x_pred': y[0], 'y_pred': y[1], 'z_pred': y[2], 'yaw_pred': y[3]}
  # TODO add classification handling for gt
  return x, y


def tfdata_generator(files, input_size, batch_size, backgrounds, bg_smoothmask, aug_prob=0, noises=[],
                     prefetch=True, parallelize=True, deterministic=False, cache=False, repeat=1):

  map_parallel = tf.data.experimental.AUTOTUNE if parallelize else None

  gen = tf.data.Dataset.from_tensor_slices(files)
  gen = gen.map(lambda filename: map_parse_input(filename, input_size), map_parallel, deterministic)

  if cache:
    gen = gen.cache()
    if not deterministic: # shuffling would destroy determinism
      gen = gen.shuffle(len(files), reshuffle_each_iteration=True)

  gen = gen.map(lambda img, mask, gt: map_replace_background(img, mask, gt, backgrounds, bg_smoothmask), map_parallel, deterministic)
  gen = gen.map(lambda img, gt: map_augmentation(img, gt, aug_prob, noises), map_parallel, deterministic)
  gen = gen.map(lambda img, gt: map_preprocessing(img, gt), map_parallel, deterministic)
  gen = gen.batch(batch_size, drop_remainder=True)
  gen = gen.repeat(repeat)

  if prefetch:
    gen = gen.prefetch(tf.data.experimental.AUTOTUNE)
  
  return gen
    

class CustomMetricR2(tf.keras.metrics.Metric):
  def __init__(self, name="r2", **kwargs):
    super(CustomMetricR2, self).__init__(name=name, **kwargs)
    self.y_true = tf.convert_to_tensor([])
    self.y_pred = tf.convert_to_tensor([])

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.y_true = tf.concat([self.y_true, tf.squeeze(y_true)], axis=0)
    self.y_pred = tf.concat([self.y_pred, tf.squeeze(y_pred)], axis=0)

  def result(self):
    SS_res =  K.sum(K.square(self.y_true - self.y_pred)) 
    SS_tot = K.sum(K.square(self.y_true - K.mean(self.y_true))) 
    r2 = (1 - SS_res/(SS_tot + K.epsilon()))
    return r2

  def reset_states(self):
    # The state of the metric will be reset at the start of each epoch.
    self.y_true = tf.convert_to_tensor([])
    self.y_pred = tf.convert_to_tensor([])


def network_compile(model, regression, classification, compute_r2=False):

  loss = []
  metrics = []
  eager = False

  if regression:
    loss.extend(['mean_absolute_error'] * 4)
    metrics.append('mse')
    if compute_r2:
      metrics.append(CustomMetricR2())
      eager = True
      # Computing R2 without run_eagerly=True, I get the error: An op outside of the function building code is being passed a "Graph" tensor.
      #   best solution, but hard https://github.com/tensorflow/tensorflow/issues/32889
      #   workaround https://github.com/tensorflow/tensorflow/issues/27519#issuecomment-662096683
      #   however, running tf.executing_eagerly() before the compile, it seems that TF is already executing eagerly 
      #   PLEASE note that run_eagerly seriously decreases time performance
    
  if classification:
    loss.extend(['categorical_crossentropy'] * 4)
    metrics.append('accuracy')

  model.compile(loss=loss,
                metrics=metrics,
                optimizer='adam',
                run_eagerly=eager) 

  
  return model


def network_train_generator(model, input_size, data_files, 
                            regression, classification, backgrounds, bg_smoothmask, aug_prob, noises = [],
                            batch_size = 64, epochs = 30, oversampling = 1, verbose = 2,
                            validation_split = 0.3, validation_shuffle = True,
                            use_lr_reducer = True, use_early_stop = False, compute_r2 = False,
                            use_profiler = False, profiler_dir = '.\\logs', time_train = True):

  # --- Compilation
  model = network_compile(model, regression, classification, compute_r2)

  # --- Callbacks

  callbacks = []

  if use_lr_reducer:
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=1, patience=4, min_lr=0.1e-6)
    callbacks.append(lr_reducer)

  if use_early_stop:
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, verbose=1)
    callbacks.append(early_stop)

  if use_profiler:
    tensorboard = tf.keras.callbacks.TensorBoard(profiler_dir, histogram_freq=1, profile_batch = '5,100')
    callbacks.append(tensorboard)

  # --- Data
  
  from sklearn.model_selection import train_test_split
  data_files_train, data_files_valid = train_test_split(data_files, test_size=validation_split, 
                                                        shuffle=validation_shuffle, random_state=1) # TODO remove seed

  backgrounds = tf.convert_to_tensor(backgrounds) # saves time during training
  noises = tf.convert_to_tensor(noises) # saves time during training

  generator_train = tfdata_generator(data_files_train, input_size, batch_size,
                                     backgrounds, bg_smoothmask, aug_prob, noises, 
                                     deterministic=False, cache=True, repeat=oversampling)
                                     
  generator_valid = tfdata_generator(data_files_valid, input_size, batch_size,
                                     backgrounds, bg_smoothmask, aug_prob=0, noises=[], 
                                     deterministic=True, cache=True, repeat=1)

  # --- Training

  print('Training started...')
  
  if time_train:
    start_time = time.monotonic()

  np.set_printoptions(suppress=True)

  history = model.fit(
      x = generator_train,
      validation_data = generator_valid,
      epochs = epochs,
      callbacks = callbacks,
      verbose = verbose
  )

  if time_train:
    print('\nTraining time: {:.2f} minutes\n'.format((time.monotonic() - start_time)/60))

  # # --- Evaluation
  
  # if True:

  #   original_stdout = sys.stdout # Save a reference to the original standard output
  #   timestr = time.strftime("%Y%m%d_%H%M%S")
  #   save_name = '{} evaluation after training.txt'.format(timestr)
  #   save_path = os.path.join("./../../dev-models/training_tfdata_tests/", save_name)
  #   # save_path = os.path.join("/project/save/", save_name)
  #   output_file = open(save_path, 'w')
  #   print('Printing on', save_path)
  #   sys.stdout = output_file # Change the standard output to the file we created.

  #   print('\nEvaluation started...')

  #   print('\n----- on validation set with network_evaluate on original images')

  #   network_evaluate(model, data_files_valid, input_size, batch_size, regression, classification)

  #   print('\n----- on validation set with network_evaluate on INDOOR1 background')

  #   bgs_valid = load_backgrounds('C:/Users/96mar/Desktop/meeting_dario/data/aug/backgrounds_dario/indoor1/', bg_smoothmask)
  #   # bgs_valid = load_backgrounds('/project/backgrounds/indoor1/', bg_smoothmask)
  #   network_evaluate(model, data_files_valid, input_size, batch_size, regression, classification, bgs_valid, bg_smoothmask)

  #   print('\n----- on validation set with network_evaluate on INDOOR2 background')

  #   bgs_valid = load_backgrounds('C:/Users/96mar/Desktop/meeting_dario/data/aug/backgrounds_dario/indoor2/', bg_smoothmask)
  #   # bgs_valid = load_backgrounds('/project/backgrounds/indoor2/', bg_smoothmask)
  #   network_evaluate(model, data_files_valid, input_size, batch_size, regression, classification, bgs_valid, bg_smoothmask)
    
  #   print('\nEvaluation finished.')

  #   sys.stdout = original_stdout # Reset the standard output to its original value
  #   output_file.close()

  return model, history


### --------------------- SAVING --------------------- ###


def network_save(folder, name, model, history, save_plot = False):
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

  with open(model_path + '_history.pickle', 'wb') as fp:
    pickle.dump(history, fp)

  print('\nModel and history saved in', model_h5)

  return model_h5


### --------------------- METRICS --------------------- ###


def network_stats(history, regression, classification, view, save, save_folder = '', save_name = '',
                  dpi=300, save_pdf = False, scale_loss = None, scale_r2 = None, scale_acc = None):

  ncharts = 0
  if 'loss' in history:
    ncharts += 1
  if 'x_pred_r2' in history:
    ncharts += 1
  if 'x_class_accuracy' in history:
    ncharts += 1

  print()
  fig, axs = plt.subplots(1, ncharts, figsize=(8*ncharts,5))
  fig.suptitle(save_name)
  counter = 0

  if not isinstance(axs, np.ndarray):
    axs = [axs] # just for easy indexing below 

  # - Loss

  if 'loss' in history:
    axs[counter].plot(history['loss'], 'k--', label='Train Loss')
    axs[counter].plot(history['val_loss'], 'k', label='Valid Loss')
    axs[counter].legend(loc='upper right')
    axs[counter].set_xlabel('Epoch')
    axs[counter].set_ylim(scale_loss)
    final_train_loss = history['loss'][-1]
    final_valid_loss = history['val_loss'][-1]
    axs[counter].set_title('Loss (train {:.2f}, val {:.2f})'.format(final_train_loss, final_valid_loss))
    print('Train Loss: \t\t', final_train_loss)
    print('Valid Loss: \t\t', final_valid_loss)
    counter += 1

  # - R2
  
  if 'x_pred_r2' in history:
    axs[counter].plot(history['x_pred_r2'], 'r--', label='x_class train R2')
    axs[counter].plot(history['val_x_pred_r2'], 'r', label='x_class valid R2')
    axs[counter].plot(history['y_pred_r2'], 'g--', label='y_class train R2')
    axs[counter].plot(history['val_y_pred_r2'], 'g', label='y_class valid R2')
    axs[counter].plot(history['yaw_pred_r2'], 'b--', label='z_class train R2')
    axs[counter].plot(history['val_yaw_pred_r2'], 'b', label='z_class valid R2')
    axs[counter].plot(history['z_pred_r2'], 'y--', label='w_class train R2')
    axs[counter].plot(history['val_z_pred_r2'], 'y', label='w_class valid R2')
    axs[counter].legend(loc='lower right')
    axs[counter].set_xlabel('Epoch')
    axs[counter].set_ylabel('R2')
    axs[counter].set_ylim(scale_r2)
    finals_train_r2 = [history['x_pred_r2'][-1], history['y_pred_r2'][-1], history['yaw_pred_r2'][-1], history['z_pred_r2'][-1]]
    finals_valid_r2 = [history['val_x_pred_r2'][-1], history['val_y_pred_r2'][-1], history['val_yaw_pred_r2'][-1], history['val_z_pred_r2'][-1]]
    axs[counter].set_title('R2 (train {:.2f}, val {:.2f})'.format(np.mean(finals_train_r2), np.mean(finals_valid_r2)))
    print('Train R2 [x,y,z,w]: \t', finals_train_r2)
    print('Valid R2 [x,y,z,w]: \t', finals_valid_r2)
    counter += 1

  # - Accuracy
  
  if 'x_class_accuracy' in history:
    axs[counter].plot(history['x_class_accuracy'], 'r--', label='x_class train Accuracy')
    axs[counter].plot(history['val_x_class_accuracy'], 'r', label='x_class valid Accuracy')
    axs[counter].plot(history['y_class_accuracy'], 'g--', label='y_class train Accuracy')
    axs[counter].plot(history['val_y_class_accuracy'], 'g', label='y_class valid Accuracy')
    axs[counter].plot(history['z_class_accuracy'], 'b--', label='z_class train Accuracy')
    axs[counter].plot(history['val_z_class_accuracy'], 'b', label='z_class valid Accuracy')
    axs[counter].plot(history['w_class_accuracy'], 'y--', label='w_class train Accuracy')
    axs[counter].plot(history['val_w_class_accuracy'], 'y', label='w_class valid Accuracy')
    axs[counter].legend(loc='lower right')
    axs[counter].set_xlabel('Epoch')
    axs[counter].set_ylabel('Accuracy')
    axs[counter].set_ylim(scale_acc)
    finals_train_accur = [history['x_class_accuracy'][-1], history['y_class_accuracy'][-1], history['z_class_accuracy'][-1], history['w_class_accuracy'][-1]]
    finals_valid_accur = [history['val_x_class_accuracy'][-1], history['val_y_class_accuracy'][-1], history['val_z_class_accuracy'][-1], history['val_w_class_accuracy'][-1]]
    axs[counter].set_title('Accur (train {:.2f}, val {:.2f})'.format(np.mean(finals_train_accur), np.mean(finals_valid_accur)))
    print('Train Accur [x,y,z,w]: \t', finals_train_accur)
    print('Valid Accur [x,y,z,w]: \t', finals_valid_accur)
    counter += 1
 
  if save:
    general_utils.create_folder_if_not_exist(save_folder)
    figname = os.path.join(save_folder, '{} - v1_all_var_metrics.{}'.format(save_name, 'pdf' if save_pdf else 'png'))
    fig.savefig(figname, bbox_inches='tight', dpi=dpi) # https://stackoverflow.com/questions/39870642/matplotlib-how-to-plot-a-high-resolution-graph

  if view:
    plt.show()
  else:
    plt.close()

  print('Model stats {}.'.format('saved' if save else 'shown'))



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def network_evaluate(model, data_files, input_size, batch_size, regression, classification, 
                     backgrounds=[], bg_smoothmask=False, aug_prob=0, noises=[]):
  
  # --- Data

  backgrounds = tf.convert_to_tensor(backgrounds) # saves time during training

  # print([fn.split(' ')[-1] for fn in data_files[:5]])
  # data_files = data_files[:4] # if uncommented, keras and sklearn produce the same result... so keras is computing R2 just for the first batch

  generator_test = tfdata_generator(data_files, input_size, batch_size,
                                    backgrounds, bg_smoothmask, aug_prob, noises, 
                                    deterministic=True, cache=True, repeat=1)                
  
  data = np.array(list(generator_test.as_numpy_iterator()), dtype=object)
  data_x = np.vstack(data[:,0][:])
  y = np.vstack(data[:,1])
  yx = np.hstack([batch[0]['x_pred'] for batch in y])
  yy = np.hstack([batch[0]['y_pred'] for batch in y])
  yz = np.hstack([batch[0]['z_pred'] for batch in y])
  yw = np.hstack([batch[0]['yaw_pred'] for batch in y])
  data_y = np.array([yx, yy, yz, yw])
  # print('data_x shape', data_x.shape)
  # print('data_y shape', data_y.shape)

  # --- Built-in evaluation

  print('\n--KERAS EVALUATION ON GENERATOR\n')
  evaluate_metrics = model.evaluate(generator_test, verbose=0, return_dict=True)
  
  print('TOTAL LOSS \t\t', evaluate_metrics['loss'])
  
  for metric in ['loss', 'mse', 'r2']:
    current = []
    for var in general_utils.variables_names[:4]:
      metric_name = '{}_{}'.format(var, metric)
      if metric_name in evaluate_metrics:
        current.append(evaluate_metrics[metric_name])

    values = ' '.join(['{:.08f}'.format(value) for value in current])
    print('[x, y, z, w] {} \t [{}]'.format(metric.upper(), values))
  
  # --- Custom evaluation
  
  print('\n--SKLEARN EVALUATION\n')
  pred = model.predict(generator_test)
  pred = np.squeeze(np.array(pred))
  y_transposed = np.transpose(data_y) # required in the shape (n_samples, n_outputs)
  p_transposed = np.transpose(pred) # required in the shape (n_samples, n_outputs)

  print('MAE SUM (loss) \t\t', np.sum(mean_absolute_error(y_transposed, p_transposed, multioutput='raw_values')))
  print('R2 MEAN \t\t', r2_score(y_transposed, p_transposed))
  print('[x, y, z, w] MAE \t', mean_absolute_error(y_transposed, p_transposed, multioutput='raw_values')) # same as loss
  print('[x, y, z, w] MSE \t', mean_squared_error(y_transposed, p_transposed, multioutput='raw_values'))
  print('[x, y, z, w] RMSE \t', np.sqrt(mean_squared_error(y_transposed, p_transposed, multioutput='raw_values')))
  print('[x, y, z, w] R2 \t', r2_score(y_transposed, p_transposed, multioutput='raw_values'))

  