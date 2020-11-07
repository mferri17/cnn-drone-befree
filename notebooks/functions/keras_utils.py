
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

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.metrics

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# %tensorflow_version 1.x
import tensorflow as tf

# when importing keras, please notice:
#   https://stackoverflow.com/a/57298275/10866825
#   https://www.pyimagesearch.com/2019/10/21/keras-vs-tf-keras-whats-the-difference-in-tensorflow-2-0/

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# -- for GradCAM
from tf_keras_vis.utils import normalize
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from matplotlib import cm


# -- import other files
sys.path.append('./')
import general_utils
# import importlib
# importlib.import_module('general_utils')





################################################################
############ VARIABLES
#################






################################################################
############ FUNCTIONS
#################


### ------ NETWORK

def network_create(original_model_path, retrain_all, var_index, classification_only = True, view_plot = False, view_summary = True, retrain_from_layer = 0):
  '''
    Creates the network structure just as the original Dario's old model but flattened (without nested models). Weights are copied as they are from the old trained model.
    The function can create a network which has classification variables and also regression original ones, depending on the value of `classification_only`.
    If `classification_only == True` you can also specify a value for variable `var_index`, so that the network will only have one output - this is useful for better handling Network Interpretability techniques. 
    If `var_index is None`, then all classification variables are being used. If `classification_only == False`, `var_index` is ignored.

    Parameters:
        original_model_path (str): Original Dario's model .h5 file path
        retrain_all (bool): If True all layers are trainable, otherwise just the last prediction layers are while other weights are taken from the original model
        var_index (int): Only uable when `classification_only == False`. If not None, specifies the only output classification variable the network has to be composed of (4: x, 5: y, 6: z, 7: w)
        classification_only (bool): If True, only classification variables compose the output of the network
        view_plot (bool): If True, visualize graph of the network
        view_summary (bool): If True, visualize tensorflow summary of the network
        retrain_from_layer (int): Index of the layer after you want to retrain the network. Only used when `retrain_all` is true.

    Returns:
        keras.engine.functional (keras.engine.functional): Resulting model
  '''

  if not classification_only: # if classification + regression is selected ...
    var_index = None # ... then ALL variables are taken into consideration

  # --- Network architecture

  input_img = Input(shape=(general_utils.image_height, general_utils.image_width, 3), name = 'input_1')

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

  if classification_only:
    outputs = [y_4, y_5, y_6, y_7]
    if var_index is not None:
      outputs = outputs[var_index - 4]
  else:
    outputs = [y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7]

  flat_model = Model(inputs = input_img, outputs = outputs) # MODEL

  # --- Restore weights from Dario's model and set trainable layers

  old_model = tf.keras.models.load_model(original_model_path)

  layers_counter = 0

  if classification_only:
    copy_until = -4 # avoids to copy regression variables weights
    non_trainable_until = -len(outputs) if isinstance(outputs, list) else -1 # only outputs will be trainable
  else:
    copy_until = None # all weights are restored
    non_trainable_until = -4  # only classification variables (the last four) will be trainable


  for layer in old_model.layers[2:copy_until]: # starts at 2 for skipping inputs and nested model
    flat_model.get_layer(layer.name).set_weights(layer.get_weights())
    layers_counter += 1

  for layer in old_model.get_layer('model_1').layers: # nested model weights
    flat_model.get_layer(layer.name).set_weights(layer.get_weights())
    layers_counter += 1

  if retrain_all:
    for layer in flat_model.layers[:retrain_from_layer]:
      layer.trainable =  False
  else:
    # Freeze all the layers before the `non_trainable_until` layer
    for layer in flat_model.layers[:non_trainable_until]:
      layer.trainable =  False

  # --- Result 

  if view_plot: 
    plot_model(flat_model, show_shapes = True, expand_nested = True)
  if view_summary:
    flat_model.summary()
    print('Please note that only output layers are now trainable.')

  return flat_model

    
def network_train(model, data_x, data_y, var_index, classification_only = True, verbose = 2,
                  batch_size = 64, epochs = 30, validation_split = 0.3, 
                  use_lr_reducer = True, use_early_stop = False):
  '''
    Trains the network with specified inputs and parameters.

    Parameters:
        model (keras.engine.functional): Model to be trained
        data_x (ndarray): Input X of the network
        data_y (list): Input Y to train the network
        var_index (int): Must be coherent with the `network_create` function same parameter
        classification_only (bool): Must be coherent with the `network_create` function same parameter
        verbose (bool): Keras fit() function verbose level (0: none, 1: some, 2: all)
        batch_size (int): Batch size
        epochs (int): Number of epochs
        validation_split (float): Percentage of input samples for the validation set
        use_lr_reducer (bool): If True, uses LR Reducer
        use_early_stop (bool): If True, uses Early Stopping

    Returns:
        model (keras.engine.functional): Same as the input model but trained
        history (keras.callbacks.History): Keras history containing model training metrics
  '''

  # --- Model settings

  if classification_only:
    loss = 'categorical_crossentropy'
    metrics = 'accuracy'
    if var_index is None:
      feed_y = data_y[4:8]
    else:
      feed_y = data_y[var_index]
  else:
    loss = list(np.array([['mean_absolute_error'] * 4, ['categorical_crossentropy'] * 4]).flatten())
    metrics = ['mse', 'accuracy']
    feed_y = data_y

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
  
  train_shape = len(data_x)
  n_val = int(train_shape * validation_split)
  
  np.random.seed(general_utils.random_seed)
  ix_val, ix_tr = np.split(np.random.permutation(train_shape), [n_val])

  x_valid = data_x[ix_val, :]
  x_train = data_x[ix_tr, :]
  y_valid = [var[ix_val] for var in feed_y] if var_index is None or not classification_only else feed_y[ix_val]
  y_train = [var[ix_tr] for var in feed_y]  if var_index is None or not classification_only else feed_y[ix_tr]

  del feed_y
  gc.collect()

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


def network_stats(history, var_index, classification_only, view, save, save_folder, save_name):
  '''
    Compute network training and validation metrics from model history.
    File name will be in the format '{save_name} - v1_{var_name}_metrics.png'.

    Parameters:
        history (keras.callbacks.History): Model to be saved
        var_index (int): Used for naming purposes and chart selection, must be coherent with the `network_create` function same parameter
        view (bool): If True, visualize the metrics
        save (bool): If True, saves the metrics
        save_folder (str): Path in which the metrics have to be saved
        save_name (str): Used for naming purposes, easy understandable identifier for the .png file name
  '''

  var_str = 'all_class' if var_index is None else general_utils.variables_names[var_index]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
  fig.suptitle(save_name)

  # - Loss

  ax1.plot(history.history['loss'], 'k--', label='Train Loss')
  ax1.plot(history.history['val_loss'], 'k', label='Valid Loss')
  ax1.legend(loc='upper right')
  ax1.set_xlabel('Epoch')
  ax1.set_title(var_str + ' training and validation Loss')

  # - Accuracy
  
  if var_index is None or not classification_only: # all variables metrics
    ax2.plot(history.history['x_class_accuracy'], 'r--', label='x_class train Accuracy')
    ax2.plot(history.history['val_x_class_accuracy'], 'r', label='x_class valid Accuracy')
    ax2.plot(history.history['y_class_accuracy'], 'g--', label='y_class train Accuracy')
    ax2.plot(history.history['val_y_class_accuracy'], 'g', label='y_class valid Accuracy')
    ax2.plot(history.history['z_class_accuracy'], 'b--', label='z_class train Accuracy')
    ax2.plot(history.history['val_z_class_accuracy'], 'b', label='z_class valid Accuracy')
    ax2.plot(history.history['w_class_accuracy'], 'y--', label='w_class train Accuracy')
    ax2.plot(history.history['val_w_class_accuracy'], 'y', label='w_class valid Accuracy')
  else: # single variable metrics
    ax2.plot(history.history['accuracy'], 'm--', label=general_utils.variables_names[var_index] + ' train Accuracy')
    ax2.plot(history.history['val_accuracy'], 'm', label=general_utils.variables_names[var_index] + ' valid Accuracy')
  ax2.legend(loc='lower right')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.set_title(var_str + ' training and validation Accuracy')

  if save:
    general_utils.create_folder_if_not_exist(save_folder)
    figname = save_folder + '{0} - v1_{1}_metrics.png'.format(save_name, var_str)
    fig.savefig(figname, bbox_inches='tight')

  if view:
    plt.show()
  else:
    plt.close()


def network_save(model, folder, name, var_index, save_plot = False):
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

  var_name = ('all_class' if var_index is None else general_utils.variables_names[var_index])
  model_path = folder + '{0} - v1_{1}_model'.format(name, var_name)

  if save_plot:
    plot_model(model, to_file = model_path + '.png', show_shapes = True, expand_nested = False)
  
  model_h5 = model_path + '.h5'
  model.save(model_h5)
  print('Model saved in', model_h5)

  return model_h5


def network_demo(model, data_x, data_img, data_actual, var_index, classification_only = True):
  '''
    Performs a short demo of the model prediction capabilities.

    Parameters:
        model (keras.engine.functional): Model to use
        data_x (ndarray): Input X data of the network
        data_img (ndarray): Image data for the given dataset
        data_actual (ndarray): Readable input Y data for the given dataset (`y_data_for_network` return value of the get_dataset_from_pickle function)
        var_index (int): Must be coherent with the `network_create` function same parameter
        classification_only (bool): Must be coherent with the `network_create` function same parameter

    Returns:
        model_path (str): Complete path of the saved .h5 file
  '''

  np.random.seed(general_utils.random_seed)
  frame_idx = np.random.randint(0, data_x.shape[0])
  batch = np.array([data_x[frame_idx]]) # input image

  print('input shape', model.input_shape)
  print('batch shape', batch.shape)

  plt.imshow(data_img[frame_idx])

  print()
  print('actual regr', data_actual[frame_idx,0:4])
  print('actual class', data_actual[frame_idx,4:8])
  print('actual class', [general_utils.var_labels[int(i)] for i in data_actual[frame_idx,4:8]])

  prediction = model.predict(batch) # model prediction

  print()
  if classification_only:
    axis = 2 if var_index is None else 1
    prediction_class_argmax = np.argmax(prediction, axis=axis).flatten()
  else:
    prediction_regr = np.array(prediction[0:4]).flatten()
    print('pred regr', prediction_regr)
    prediction_class_argmax = np.argmax(prediction[4:8], axis=2).flatten()


  print('pred class', prediction_class_argmax)
  print('pred class', [general_utils.var_labels[int(i)] for i in prediction_class_argmax] if var_index is None else general_utils.var_labels[int(prediction_class_argmax)])


  print()
  print('network prediction')
  print(np.array(prediction)) # ndarray conversion for prettier print

  print('\n--------------------------------------------------------------------')

  offset = 0 if classification_only else 4

  if var_index is None:
    for variable_to_show in range(4):
      class_idxs_sorted = np.argsort(prediction[offset + variable_to_show].flatten())[::-1]
      print('\nPredictions for variable', general_utils.variables_names[4 + variable_to_show])
      for i, idx in enumerate(class_idxs_sorted[:3]):
        cur_pred = prediction[offset + variable_to_show][0, idx] if var_index is None else prediction[0, idx]
        print("Top {} predicted class:     Pr(Class={:6} [index={}]) = {:5.3f}".format(
                i + 1, general_utils.var_labels[idx], idx, cur_pred))
  else:
    class_idxs_sorted = np.argsort(prediction.flatten())[::-1]

    print('Predictions for variable', general_utils.variables_names[4 + var_index])
    for i, idx in enumerate(class_idxs_sorted[:3]):
      cur_pred = prediction[offset + var_index][0, idx] if var_index is None else prediction[0, idx]
      print("Top {} predicted class:     Pr(Class={:6} [index={}]) = {:5.3f}".format(
              i + 1, general_utils.var_labels[idx], idx, cur_pred))

  print()


def network_export_variants(original_model_path, save_folder, save_name, data_x, data_y, retrain_all, 
                            from_layer = 0, epochs = 30, use_lr_reducer = True, use_early_stop = False,
                            copy_weights_after_first = False, view_stats = True, save_stats = True):
  '''
    Given the original Dario's model creates five different models, one for each class (all_class, x_class, y_class, z_class, w_class).
    Trains and saves all the .h5 model variants in the specified folder.
    File names will be in the format '{save_name} - v1_{var}_class.h5' where {var} is a string in ['all', 'x', 'y', 'z', 'w'].

    Parameters:
        original_model_path (str): Original Dario's model path
        destination_folder (str): Destination folder for the saved models
        save_name (str): Name for saving the files
        data_x (ndarray): Input X of the network
        data_y (ndarray): Input Y to train the network
        retrain_all (bool): If True retrains from the `from_layer` attribute layer, otherwise just the last prediction layers are trainable (the others weights are taken from the original model)
        from_layer (int): Index of the layer after you want to retrain the network. Only used when `retrain_all` is true.
        epochs (int): Number of epochs to train on
        use_lr_reducer (bool): If True, uses LR Reducer
        use_early_stop (bool): If True, uses Early Stopping
        copy_weights_after_first (bool): If True only the "all_class" network is trained, while the others (x, y, z, w classes) just inherit the weights from the first
        view_stats (bool): If True, visualize loss/accuracy charts
        save_stats (bool): If True, visualize loss/accuracy charts and saves them
  '''

  general_utils.create_folder_if_not_exist(save_folder)
  verbose = 2 if view_stats else 0
  classification_only = True

  if not copy_weights_after_first:
    # Train each variant independently

    for vi in [None, 4, 5, 6, 7]: # (None: all, 4: x, 5: y, 6: z, 7: w)
      model = network_create(original_model_path, retrain_all, vi, view_summary=False, retrain_from_layer=from_layer)
      model, history = network_train(model, data_x, data_y, vi, classification_only, verbose, 
                                      use_lr_reducer=use_lr_reducer, use_early_stop=use_early_stop, epochs = epochs)
      network_stats(history, vi, classification_only, view_stats, save_stats, save_folder, save_name)
      network_save(model, save_folder, save_name, vi)
      gc.collect()

  else:
    # Train the first network and copy the weights to the others
    
    model_all = network_create(original_model_path, retrain_all, None, view_summary=False, retrain_from_layer=from_layer)
    model_all, history = network_train(model_all, data_x, data_y, None, classification_only, verbose, 
                                    use_lr_reducer=use_lr_reducer, use_early_stop=use_early_stop, epochs = epochs)
    network_stats(history, None, classification_only, view_stats, save_stats, save_folder, save_name)
    network_save(model_all, save_folder, save_name, None)
    gc.collect()

    for vi in [4, 5, 6, 7]: # (None: all, 4: x, 5: y, 6: z, 7: w)
      model = network_create(original_model_path, False, vi, view_summary=False)

      for layer in model_all.layers[1:]: # starts at 1 for skipping inputs
        try:
          model.get_layer(layer.name).set_weights(layer.get_weights())
        except ValueError: # get_layer raises ValueError is a layer does not exist
          # for each variable, the respective model only contains the associated variable (so the other outputs will be missing)
          # print(vi, layer.name, 'layer not found, skipping')
          continue 

      for layer in model.layers:
        layer.trainable =  False
      
      network_save(model, save_folder, save_name, vi)
      gc.collect()

  return save_folder, save_name


def network_import_variants(models_folder, models_name):
  '''
    Load the five variants of the model (all_class, x_class, y_class, z_class, w_class) as saved by the function 'network_export_variants'.
    Models file names have to be in the format '{models_name} - v1_{var}_class.h5' where {var} is a string in ['all', 'x', 'y', 'z', 'w'].

    Parameters:
        models_folder (str): Path where to find the models
        models_name (str): Name prefix for the .h5 files (in the format '{models_name} v1_{var}_class.h5')

    Returns:
        model_var_all (keras.engine.functional): Model containing all classification variables as output
        model_vars (ndarray): Array containing models for single classification variable each -> [model_var_x, model_var_y, model_var_z, model_var_w]
  '''

  base_path = models_folder + models_name + ' - v1_'
  model_var_all = tf.keras.models.load_model(base_path + 'all_class_model.h5')
  model_var_x = tf.keras.models.load_model(base_path + 'x_class_model.h5')
  model_var_y = tf.keras.models.load_model(base_path + 'y_class_model.h5')
  model_var_z = tf.keras.models.load_model(base_path + 'z_class_model.h5')
  model_var_w = tf.keras.models.load_model(base_path + 'w_class_model.h5')
  model_vars = [model_var_x, model_var_y, model_var_z, model_var_w]
  print('Models imported from', models_folder)
  
  return model_var_all, model_vars


### ------ GRADCAM

def gradcam_data_select_predict_transform(model_vars, data_img, data_x, data_actual, max_samples, index_start = None):
  np.random.seed(general_utils.random_seed)
  max_random = data_x.shape[0]
  vis_idx_lb = index_start if index_start is not None else np.random.randint(0, max_random - max_samples - 1) # random selection if `index_start` not specified
  vis_idx_ub = vis_idx_lb + max_samples
  print('selected indexes are from', vis_idx_lb, 'to', vis_idx_ub, '\n')

  selected_img = np.array(data_img[vis_idx_lb:vis_idx_ub].tolist()) # images
  selected_input = np.array(data_x[vis_idx_lb:vis_idx_ub].tolist()) # model inputs

  selected_actuals = []
  selected_predictions_best = []

  for i in range(4):
    selected_actuals.append([general_utils.var_labels[int(i)] for i in data_actual[vis_idx_lb:vis_idx_ub, 4 + i]])
    selected_predictions_best.append([general_utils.var_labels[int(i)] for i in np.argmax(model_vars[i].predict(selected_input), axis=1).T]) # rows=images, cols=probs
    print(general_utils.variables_names[4 + i], 'actual \t\t', selected_actuals[i])
    print(general_utils.variables_names[4 + i], 'prediction \t', selected_predictions_best[i], '\n')
  # selected_prediction__best_all = np.argmax(model_var_all.predict(vis_input), axis=2).T # rows=images, cols=variables, last=probs

  return selected_img, selected_input, selected_actuals, selected_predictions_best, vis_idx_lb


def gradcam_model_modifier_class(m):
  '''
    Define modifier to replace the softmax function of classification layers to a linear function.

    Parameters:
        m (keras.engine.functional): Keras model
  '''
  # if var_index is None:
  m.get_layer('x_class').activation = tf.keras.activations.linear
  m.get_layer('y_class').activation = tf.keras.activations.linear
  m.get_layer('z_class').activation = tf.keras.activations.linear
  m.get_layer('w_class').activation = tf.keras.activations.linear
  # else: # single variable
  #   m.get_layer(general_utils.variables_names[var_index]).activation = tf.keras.activations.linear


def gradcam_model_modifier_last(m):
  '''
    Define modifier to replace the softmax function of the last layer to a linear function.

    Parameters:
        m (keras.engine.functional): Keras model
  '''
  m.layers[-1].activation = tf.keras.activations.linear


# Define loss functions
gradcam_loss_low    = lambda output: K.mean(output[:,general_utils.var_labels.index('low')])    if output.shape[1] == 3 else K.mean(output)
gradcam_loss_medium = lambda output: K.mean(output[:,general_utils.var_labels.index('medium')]) if output.shape[1] == 3 else K.mean(output)
gradcam_loss_high   = lambda output: K.mean(output[:,general_utils.var_labels.index('high')])   if output.shape[1] == 3 else K.mean(output)
gradcam_loss_total  = lambda output: K.mean(output)

gradcam_losses = [
  gradcam_loss_low,
  gradcam_loss_medium,
  gradcam_loss_high,
  gradcam_loss_total,
]


def gradcam_plot_pred_vs_gt(vis_img, vis_input, vis_actuals, vis_predictions_best,
                            model_var_all, model_vars, img_index, gcamplusplus = False):

  f, ax = plt.subplots(nrows=2, ncols=5, figsize=(20, 5), subplot_kw={'xticks': [], 'yticks': []})

  # X, Y, Z, W
  for i in range(4):
    gradcam = Gradcam(model_vars[i], gradcam_model_modifier_last) if not gcamplusplus else GradcamPlusPlus(model_vars[i], gradcam_model_modifier_last)
    # gradcam on the ground truth class
    cam = normalize(gradcam(gradcam_losses[general_utils.var_labels.index(vis_actuals[i][img_index])], vis_input[img_index]))
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    ax[0, i].set_title(general_utils.variables_names[i + 4] + ' GT: ' + vis_actuals[i][img_index])
    ax[0, i].imshow(vis_img[img_index])
    ax[0, i].imshow(heatmap, cmap='jet', alpha=0.3)
    # gradcam on the predicted class
    cam = normalize(gradcam(gradcam_losses[general_utils.var_labels.index(vis_predictions_best[i][img_index])], vis_input[img_index]))
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
    ax[1, i].set_title(general_utils.variables_names[i + 4] + ' PR: ' + vis_predictions_best[i][img_index])
    ax[1, i].imshow(vis_img[img_index])
    ax[1, i].imshow(heatmap, cmap='jet', alpha=0.3)

  # ALL
  img = vis_img[img_index]
  gradcam = Gradcam(model_var_all, gradcam_model_modifier_class) if not gcamplusplus else GradcamPlusPlus(model_var_all, gradcam_model_modifier_class)
  cam = normalize(gradcam(gradcam_loss_total, vis_input[img_index]))
  heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
  ax[1, 4].set_title('all_classes')
  ax[1, 4].imshow(img)
  ax[1, 4].imshow(heatmap, cmap='jet', alpha=0.3)


def get_wave_distribution(nsamples):
  interval = np.pi/10
  exceed = 10
  distr = np.sin(np.arange(0, nsamples * interval + interval*exceed, interval)) + 1
  shift = np.random.randint(0, 10) # first example does not always start at maximum probability
  distr = distr[shift:(shift-exceed)]
  return (distr / np.sum(distr))


def gradcam_comparison_varloss(vis_img, vis_input, vis_actuals, vis_predictions_best, vis_idx_lb,
                                var_types, loss_types, model_vars, frames, path, notes = 'tabulated', 
                                title = True, save = True, gcamplusplus = False):
  '''
    Given the (x, y, z, w) model variants, plot GradCAM results tabulated by variable (rows) and class (columns).

    Parameters:
        vis_img (ndarray): Input images, as returned by the function `gradcam_data_select_predict_transform`
        vis_input (ndarray): Input X, as returned by the function `gradcam_data_select_predict_transform`
        vis_actuals (ndarray): Input actuals in user-friendly format, as returned by the function `gradcam_data_select_predict_transform`
        vis_predictions_best (ndarray): Top1 prediction for each input and variable, as returned by the function `gradcam_data_select_predict_transform`
        vis_idx_lb (int): Used for naming puroposes, must be choerent with the parameter `index_start` of the function `gradcam_data_select_predict_transform`
        var_types (list): List of int, variables to take into consideration for computing GradCAM (0:x, 1:y, 2:z, 3:w)
        loss_types (list): List of int, classes to take into consideration for computing GradCAM (0:low, 1:medium, 2:high, 3:total)
        model_vars (list): List of Keras five variants model to use for computing GradCAM
        frames (list): List of indexes to take into consideration for computing GradCAM 
        path (str): Path in which to save the result if the `save` parameter is True
        notes (str): Notes to be added when creating datetime folder for saving
        title (bool): If True, subplots have suptitle
        save (bool): If True, saves the result on disk creating a datetime folder into the specified path
        gcamplusplus (bool): If True, uses GradcamPlusPlus instead of Gradcam


    Returns:
        model_var_all (keras.engine.functional): Model containing all classification variables as output
        model_vars (ndarray): Array containing models for single classification variable each -> [model_var_x, model_var_y, model_var_z, model_var_w]
  '''

  ncols = len(loss_types)
  nrows = len(var_types)

  if save:
    folder = general_utils.create_datetime_folder(path, notes)
    print(folder)

  for count_f, current_img_index in enumerate(frames):

    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int(ncols*3), int(nrows*2)), subplot_kw={'xticks': [], 'yticks': []}, constrained_layout=True)
    if title:
      f.suptitle('Frame ' + str(vis_idx_lb+current_img_index) +' tabulated per variable (row) and loss (column)')

    for count_v, vt in enumerate(var_types):
      for count_l, lt in enumerate(loss_types):

        gradcam = Gradcam(model_vars[vt], gradcam_model_modifier_last) if not gcamplusplus else GradcamPlusPlus(model_vars[vt], gradcam_model_modifier_last)
        cam = gradcam(gradcam_losses[lt], vis_input[current_img_index])

        if nrows == 1 and ncols == 1:
          cell = ax
        elif nrows == 1:
          cell = ax[count_l]
        else:
          cell = ax[count_v, count_l]
        
        heatmap = np.uint8(cm.jet(normalize(cam[0]))[..., :3] * 255)
        if lt == 3:
          celltitle = 'Average (GT {0}, PR {1})'.format(vis_actuals[vt][current_img_index], vis_predictions_best[vt][current_img_index])
        else:
          celltitle = str.upper(general_utils.variables_names[vt]) + ', class ' + general_utils.var_labels[lt]
        
        cell.set_title(celltitle)
        cell.imshow(vis_img[current_img_index]) 
        cell.imshow(heatmap, cmap='jet', alpha=0.3)
    
    if save:
      imagename = 'Frame {:04d} - class {}'.format(vis_idx_lb+current_img_index, var_types) + '.jpg'
      cell.figure.savefig(folder + imagename, bbox_inches='tight')

    if count_f % 20 == 0:
      print('Progress frame {0}/{1}'.format(count_f, len(frames)))
    
    if len(frames) > 10:
      plt.close(f)
    else:
      plt.show()

    gc.collect()



