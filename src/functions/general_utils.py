
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
import socket
import glob
from datetime import datetime

import pickle
import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from matplotlib import cm
from tensorflow.keras.utils import to_categorical





################################################################
############ VARIABLES
#################

image_height = 60
image_width = 108
variables_names = ['x_pred', 'y_pred', 'z_pred', 'yaw_pred', 'x_class', 'y_class', 'z_class', 'w_class']
var_labels = ['low','medium','high']

random_seed = 100




################################################################
############ FUNCTIONS
#################


### ------ GENERAL

def create_folder_if_not_exist(path):
  '''
    Creates the folder specified in the given path. Raises an exception is it fails.

    Parameters:
        path (str): Path to the folder that has to be created
  '''
  if not os.path.exists(os.path.dirname(path)):
      try:
          os.makedirs(os.path.dirname(path))
      except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise


def create_datetime_folder(base_folder, notes = ''):
  '''
    Creates a folder, named with the current datetime, inside the given path.

    Parameters:
        base_folder (str): Base path in which to create the folder
        notes (str): Additional notes for the folder name, after the datetime

    Returns:
        path (str): Complete path to the created folder
  '''
  timestr = time.strftime("%Y%m%d_%H%M%S")
  path = base_folder + timestr + ((' ' + notes) if notes != '' else '') + '/'
  create_folder_if_not_exist(path)
  return path


def load_images_from_folder(folder, extension='jpg', recursive = True):
  '''
    Loads images from folder in RGB format.

    Parameters:
        folder (str): Path in which to find the images

    Returns:
        images (ndarray): List of images found
  '''

  images = []
  for filename in list_files_in_folder(folder, extension, recursive):
    img = plt.imread(os.path.join(folder,filename))
    if img is not None:
      images.append(img)

  return np.array(images, dtype='O')


def list_files_in_folder(folder, extension, recursive = True):
  '''
    Find file list from folder.

    Parameters:
        folder (str): Path in which to find the files
        extension (str): Extension of the files to be found

    Returns:
        imgs_paths (list): List of images file paths
  '''
  
  from pathlib import Path

  if recursive:
    glob = Path(folder).rglob('*.{}'.format(extension))
  else:
    glob = Path(folder).glob('*.{}'.format(extension))

  # for map info, see https://github.com/DeepRNN/image_captioning/issues/66#issuecomment-687896235
  return list(map(str, glob))


def load_pickle(filepath):
  with open(filepath, 'rb') as fp:
    return pickle.load(fp)


def subplots_get_cell(total, rows, cols, counter, ax):
  if(total == 1 and cols == 1):
    cell = ax
  elif(rows == 1):
    cell = ax[counter]
  else:
    cell = ax[counter // cols, counter % cols]
  return cell


### ------ DATASET
  
def get_dataset_from_pickle(pickle_path, dataset_start_index = 0, dataset_keep_ratio = 0.5, plot_actual_distribution = False):
  '''
    Reads a pickle which contains a dataset (Pandas dataframe format) and returns the same in a format which is usable for network training. 
     - dataset[:, 0] image data
     - dataset[:, 1] user pose data
     - dataset[:, 2] drone odom data (velocities)
    Regression variables are used to retrieve classification variables by discretizing continous values accordingly to arbitrary defined bin intervals.
    Train set size is reduced accordingly to a parameter in order to avoid overloads.

    Parameters:
        pickle_path (str): Path of the pickle dataset file.
        dataset_start_index (int): Start index for the dataset to read and keep
        dataset_keep_ratio (float): Percentage of the dataset to keep, the rest is discarded.
        plot_actual_distribution (bool): If True, plots actual values distribution after the discretization

    Returns:
        img_data (ndarray): Images as they come from the original dataset
        x_data (ndarray): Input X for the network
        y_data_for_network (list): Input Y for training the network (4 regression and 4 classification variables)
        y_data_pretty (ndarray): Readable Y data for the given X, same as `y_data_for_network` but in a different shape
        odom_data (ndarray): Input odom for the network
  '''

  # --- Reading and preprocessing

  print('Reading dataset from ' + pickle_path)
  
  dataset = pd.read_pickle(pickle_path).values
  print('dataset original shape: ' + str(dataset.shape))

  keep = dataset_start_index + int(len(dataset) * dataset_keep_ratio)
  dataset = dataset[dataset_start_index:keep] # reducing dataset size
  print('dataset keep shape: \t' + str(dataset.shape))
  gc.collect()

  img_data = dataset[:, 0]
  x_data = 255 - img_data
  x_data = np.vstack(x_data[:]).astype(np.float32)
  x_data = np.reshape(x_data, (-1, image_height, image_width, 3))

  y_data = dataset[:, 1]
  y_data = np.vstack(y_data[:]).astype(np.float32)

  odom_data = dataset[:, 2]
  odom_data = np.vstack(odom_data[:]).astype(np.float32)

  print('img_data shape: \t' + str(img_data.shape))
  print('x_data shape: \t\t' + str(x_data.shape))
  print('y_data shape: \t\t' + str(y_data.shape))
  print('odom_dataset shape: \t' + str(odom_data.shape))

  # --- Actuals discretization

  # - X
  var_x_values = y_data[:,0]
  var_x_bins = np.array([-np.inf, 1.4, 1.6, np.inf])
  var_x_assign = np.digitize(var_x_values, var_x_bins) - 1

  # - Y
  var_y_values = y_data[:,1]
  var_y_bins = np.array([-np.inf, -0.15, 0.15, np.inf])
  var_y_assign = np.digitize(var_y_values, var_y_bins) - 1

  # - Z
  var_z_values = y_data[:,2]
  var_z_bins = np.array([-np.inf, -0.05, 0.05, np.inf])
  var_z_assign = np.digitize(var_z_values, var_z_bins) - 1

  # - W
  var_w_values = y_data[:,3]
  var_w_bins = np.array([-np.inf, -0.2, 0.2, np.inf])
  var_w_assign = np.digitize(var_w_values, var_w_bins) - 1

  # --- Actuals distribution

  def plot_class_distribution(values, labels, var_name):
    plt.hist(values)
    plt.xticks(range(len(labels)), labels)
    plt.title(var_name + ' class distribution')
    plt.show()

  if plot_actual_distribution:
    plot_class_distribution(var_x_assign, var_labels, 'X')
    plot_class_distribution(var_y_assign, var_labels, 'Y')
    plot_class_distribution(var_z_assign, var_labels, 'Z')
    plot_class_distribution(var_w_assign, var_labels, 'W')

  # --- Actuals mapping into list of 4 regression and 4 classification variables
  #     -> (xr, yr, zr, wr, xc, yc, zc, wc) This shape is required for training

  stack = np.squeeze(np.dstack((var_x_assign, var_y_assign, var_z_assign, var_w_assign)))
  y_data_pretty = np.append(y_data, stack, axis=1) # shape (samples, all_variables) (?, 8)

  # regression variables
  ytr = np.transpose(y_data_pretty[:,0:4])          # shape (regr_variables, samples) (4, ?)

  # classification variables
  cat_train = to_categorical(y_data_pretty[:,4:8])  # shape (samples, class_variables, categorical) (?, 4, 3)
  ytc = np.transpose(cat_train, (1, 0, 2))          # shape (class_variables, samples, categorical) (4, ?, 3)

  y_data_for_network = [ytr[0], ytr[1], ytr[2], ytr[3], ytc[0], ytc[1], ytc[2], ytc[3]]
  print('y_data_for_network shape cannot be computed because elements in the list have different shapes:') 
  print('y_data_for_network number of variables \t\t\t\t', len(y_data_for_network))
  print('y_data_for_network single regression variable (0:4) \t\t', np.shape(y_data_for_network[0]))
  print('y_data_for_network single classification variable (4:8) \t', np.shape(y_data_for_network[4]))

  # --- Result

  return img_data, x_data, y_data_for_network, y_data_pretty, odom_data
    

def image_background_replace_canny(img, canny_min, canny_max, l2gradient = False, 
                                    blur = 9, dilate_iter = 5, erode_iter = 5, mask_safe_area = None,
                                    transparent = True, replace_bg_images = None, 
                                    debug = False, save = False, save_folder = '', save_name = 'removebg-canny'):
  '''
    Replaces or removes the background from a given image by applying a mask, generated by using computer vision techniques based on Canny Edge Detection.
    Taken and adapted from https://stackoverflow.com/questions/29313667/how-do-i-remove-the-background-from-this-kind-of-image.

    Parameters:
        img (ndarray): Input image
        canny_min (int): Min threshold for the cv2 Canny function, in edge detection defines minimum intensity gradient to take a pixel into consideration
        canny_max (int): Max threshold for the cv2 Canny function, in edge detection defines intensity gradient threshold over which a pixel is a sure edge
        l2gradient  (bool): If true, uses a more accurate equation for finding gradient magnitude
        blur  (int): Blur to apply on the mask (must be positive and odd)
        dilate_iter  (int): Number of iteration for mask dilatation
        erode_iter  (int): Number of iteration for mask erosion
        mask_safe_area  (tuple): Defines the image area to certainly keep, regardless any masking algorithm, in the rectangle format (x, y, width, height)
        transparent  (bool): Whether to use a blank or transparent mask for the background. Ignored if `replace_bg_images` is a valid array.
        replace_bg_images (ndarray): Array of images to set as background instead of blank/transparent mask. If multiple images are provided, an array is returned.
        debug  (bool): If True, shows intermediate results as images
        save  (bool): If True, resulting image is saved on disk
        save_folder  (str): Path to save the resulting image
        save_name (str): Name to save the resulting image

    Returns:
        images (ndarray): Resulting images with replaced backgrounds. It will have multiple elements only if `replace_bg_images` is a valid array and contains multiple images.
  '''

  # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # dataset images are RGB, but cv2 wants BGR

  # --- Edge detection

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  if debug:
    plt.imshow(gray)
    plt.show()

  edges = cv2.Canny(gray, canny_min, canny_max, L2gradient = l2gradient)
  if debug:
    plt.imshow(edges)
    plt.colorbar()
    plt.show()
  edges = cv2.dilate(edges, None)
  if debug:
    plt.imshow(edges)
    plt.colorbar()
    plt.show()
  edges = cv2.erode(edges, None)
  if debug:
    plt.imshow(edges)
    plt.colorbar()
    plt.show()

  # --- Find contours in edges, sort by area

  contour_info = []
  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

  for c in contours:
      contour_info.append((
          c,
          cv2.isContourConvex(c),
          cv2.contourArea(c),
      ))
  contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
  max_contour = contour_info[0]

  # --- Create empty mask, draw filled polygon on it corresponding to largest contour
  #     Mask is black, polygon is white
  
  mask = np.zeros(edges.shape)
  cv2.fillConvexPoly(mask, max_contour[0], (255))
  if debug:
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

  # --- Smooth mask, then exclude `mask_safe_area` and blur it
  
  mask = cv2.dilate(mask, None, iterations=dilate_iter)
  mask = cv2.erode(mask, None, iterations=erode_iter)

  if mask_safe_area is not None and len(mask_safe_area) == 4:
    x = mask_safe_area[0]
    y = mask_safe_area[1]
    w = mask_safe_area[2]
    h = mask_safe_area[3]
    mask[y:y+h, x:x+w] = 255

  mask = cv2.GaussianBlur(mask, (blur, blur), 0)
  if debug:
    plt.imshow(mask)
    plt.colorbar()
    plt.show()

  # --- Choose background between blank and images 

  replacing = (isinstance(replace_bg_images, list) or isinstance(replace_bg_images, np.ndarray)) and len(replace_bg_images) > 0
  if replacing:
    backgrounds = replace_bg_images / 255
  else:
    backgrounds = [np.ones((np.shape(img)[1], np.shape(img)[0], 3))] # In RGB format

  # --- Blend masked img into backgrounds

  mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
  mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
  img         = img.astype('float32') / 255.0                 #  for easy blending

  result = []
  for bg in backgrounds:
    bg_resized = cv2.resize(bg, (np.shape(img)[1], np.shape(img)[0]))[:,:,:3] # resize and rmeove alpha channel
    masked = (mask_stack * img) + ((1-mask_stack) * bg_resized)  # Blend
    masked = (masked * 255).astype('uint8')                           # Convert back to 8-bit 
    if debug:
      plt.imshow(masked)
      plt.show()
    result.append(masked)
    
  # --- Add alpha channel for transparent background

  if not replacing and transparent:
    # merge RGB channels with the B&W mask we got before, which is used for creating the alpha channel
    # even if working with a single image, result must be an array 
    c_red, c_green, c_blue = cv2.split(img) # split image into channels
    result = [(cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0)) * 255).astype('uint8')]

  # --- Saving

  if save:
    for count, im in enumerate(result):
      create_folder_if_not_exist(save_folder)
      imgname = save_folder + save_name + '_bg{:03d}.png'.format(count)
      if cv2.imwrite(imgname, cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)) and debug:
        print('Saved in', imgname)
  
  return np.array(result)


def image_background_replace_grabcut(img, mode, rect_person, rect_face, iterations = 2, smooth = True,
                                      transparent = True, replace_bg_images = None, 
                                      debug = False, save = False, save_folder = '', save_name = 'removebg-grabcut'):
  '''
    Replaces or removes the background from a given image by applying a mask, generated by using computer vision techniques based on the GrabCut algorithm.
    Taken and adapted from https://docs.opencv.org/4.1.2/d8/d83/tutorial_py_grabcut.html.

    Parameters:
        img (ndarray): Input image
        mode (cv2.GrabCutModes): If equal cv2.GC_INIT_WITH_RECT, it just uses the person pose. If equal cv2.GC_INIT_WITH_MASK, it uses both face and person pose (if specified)
        rect_person (tuple): Area of the image where the person has been detected, in the rectangle format (x, y, width, height)
        rect_face (tuple): Area of the image where the face of the person has been detected, in the rectangle format (x, y, width, height)
        smooth (bool): If True, smooth the generated mask by using cv2 dilate-erode-smooth sequence
        iterations (int): Number of iterations for the GrabCut algorithm
        transparent (bool): Whether to use a blank or transparent mask for the background. Ignored if `replace_bg_images` is a valid array.
        replace_bg_images (ndarray): Array of images to set as background instead of blank/transparent mask. If multiple images are provided, an array is returned.
        debug (bool): If True, shows intermediate results as images
        save (bool): If True, resulting image is saved on disk
        save_folder (str): Path to save the resulting image
        save_name (str): Name to save the resulting image

    Returns:
        images (ndarray): Resulting images with replaced backgrounds. It will have multiple elements only if `replace_bg_images` is a valid array and contains multiple images.
  '''
  
  bgdModel = np.zeros((1,65),np.float64) # used by the algorithm internally, just create it like this
  fgdModel = np.zeros((1,65),np.float64) # used by the algorithm internally, just create it like this
  
  valid_person = rect_person is not None and len(rect_person) == 4
  valid_face = rect_face is not None and len(rect_face) == 4

  if mode == cv2.GC_INIT_WITH_RECT and not valid_person:
    raise ValueError('Parameter `rect_person` is missing or not valid. Check selected `mode` and format (x, y, width, height).')
  elif mode == cv2.GC_INIT_WITH_MASK and not valid_face:
    raise ValueError('Parameter `rect_face` is missing or not valid. Check selected `mode` and format (x, y, width, height).')
  
  # --- Create a base mask similar to the image
  
  if valid_person:
    mask = np.zeros(img.shape[:2],np.uint8) # sure-background everywhere
    x, y, w, h = rect_person[0], rect_person[1], rect_person[2], rect_person[3]
    mask[y:y+h, x:x+w] = cv2.GC_PR_BGD # probable-background inside the rectangle
  else:
    mask = np.ones(img.shape[:2],np.uint8) * cv2.GC_PR_BGD # probable-background everywhere
    
  if mode == cv2.GC_INIT_WITH_MASK and valid_face:
    x, y, w, h = rect_face[0], rect_face[1], rect_face[2], rect_face[3]
    mask[y:y+h, x:x+w] = cv2.GC_FGD # sure-foreground inside the face area

  # --- Run the algorithm

  cv2.grabCut(img, mask, rect_person, bgdModel, fgdModel, iterations, mode)

  # --- Binarize and smooth the mask

  isbackground = (mask==cv2.GC_BGD) | (mask==cv2.GC_PR_BGD)
  mask_binarized = np.where(isbackground, 0, 1).astype('float32') # [0 if isbackground(px) else 1 for px in image]
  mask_smoothed = mask_binarized

  if smooth:
    mask_smoothed = cv2.dilate(mask_binarized, None, iterations=5)
    mask_smoothed = cv2.erode(mask_smoothed, None, iterations=5)
    mask_smoothed = cv2.GaussianBlur(mask_smoothed, (5, 5), 0)

  # --- Debug results

  if debug:
    plt.imshow(img)
    if valid_person:
      plt.gca().add_patch(Rectangle((rect_person[0], rect_person[1]), rect_person[2], rect_person[3], linewidth=2, edgecolor='r', facecolor='none'))
    if valid_face:
      plt.gca().add_patch(Rectangle((rect_face[0], rect_face[1]), rect_face[2], rect_face[3], linewidth=2, edgecolor='y', facecolor='none'))
    plt.show()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    plt.imshow(mask_smoothed)
    plt.colorbar()
    plt.show()

  # --- Choose background between blank and images 
  
  replacing = (isinstance(replace_bg_images, list) or isinstance(replace_bg_images, np.ndarray)) and len(replace_bg_images) > 0
  if replacing:
    backgrounds = replace_bg_images / 255
  else:
    backgrounds = [np.ones((np.shape(img)[1], np.shape(img)[0], 3))] # In RGB format

  # --- Blend masked img into backgrounds

  mask_stack = mask_smoothed[:,:,np.newaxis]    # Add 3rd dimension for broadcasting
  img        = img.astype('float32') / 255.0    # For easy blending

  result = []
  for bg in backgrounds:
    bg_resized = cv2.resize(bg, (np.shape(img)[1], np.shape(img)[0]))[:,:,:3] # resize and remove alpha channel
    masked = (mask_stack * img) + ((1-mask_stack) * bg_resized)  # Blend
    masked = (masked * 255).astype('uint8')                      # Convert back to 8-bit 
    result.append(masked)
    
  # --- Add alpha channel for transparent background

  if not replacing and transparent:
    # merge RGB channels with the B&W mask we got before, which is used for creating the alpha channel
    # even if working with a single image, result must be an array 
    c_red, c_green, c_blue = cv2.split(img) # split image into channels
    result = [(cv2.merge((c_red, c_green, c_blue, mask_stack)) * 255).astype('uint8')]

  # --- Saving

  if save:
    for count, im in enumerate(result):
      create_folder_if_not_exist(save_folder)
      imgname = save_folder + save_name + '_bg{:03d}.png'.format(count)
      if cv2.imwrite(imgname, cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)) and debug:
        print('Saved in', imgname)

  return np.array(result)


def image_background_replace_mask(img, mask, smooth = True,
                                  transparent = True, replace_bg_images = None, 
                                  debug = False, save = False, save_folder = '', save_name = 'removebg-mask'):
  '''
    Replaces or removes the background from a given image by applying the given mask.

    Parameters:
        img (ndarray): Input image
        img (ndarray): Input mask for the image
        smooth (bool): If True, smooth the generated mask by using cv2 dilate-erode-smooth sequence
        transparent (bool): Whether to use a blank or transparent mask for the background. Ignored if `replace_bg_images` is a valid array.
        replace_bg_images (ndarray): Array of images to set as background instead of blank/transparent mask. If multiple images are provided, an array is returned.
        debug (bool): If True, shows intermediate results as images
        save (bool): If True, resulting image is saved on disk
        save_folder (str): Path to save the resulting image
        save_name (str): Name to save the resulting image

    Returns:
        images (ndarray): Resulting images with replaced backgrounds. It will have multiple elements only if `replace_bg_images` is a valid array and contains multiple images.
  '''

  # --- Binarize and smooth the mask

  mask_smoothed = mask.astype('float32')

  if smooth:
    mask_smoothed = cv2.dilate(mask_smoothed, None, iterations=5)
    mask_smoothed = cv2.erode(mask_smoothed, None, iterations=5)
    mask_smoothed = cv2.GaussianBlur(mask_smoothed, (5, 5), 0)

  # --- Debug results

  if debug:
    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.colorbar()
    plt.show()
    if smooth:
      plt.imshow(mask_smoothed)
      plt.colorbar()
      plt.show()

  # --- Choose background between blank and images 
  
  replacing = isinstance(replace_bg_images, (list, np.ndarray)) and len(replace_bg_images) > 0
  if replacing:
    backgrounds = replace_bg_images / 255
  else:
    backgrounds = [np.ones((np.shape(img)[1], np.shape(img)[0], 3))] # In RGB format

  # --- Blend masked img into backgrounds

  mask_stack = mask_smoothed[:,:,np.newaxis]    # Add 3rd dimension for broadcasting
  img        = img.astype('float32') / 255.0    # For easy blending

  result = []
  for bg in backgrounds:
    if len(bg.shape) == 2: # in case 2D images are given
      bg = cv2.cvtColor((bg*255).astype('uint8'), cv2.COLOR_GRAY2BGR) / 255

    bg_resized = cv2.resize(bg, (np.shape(img)[1], np.shape(img)[0]))[:,:,:3] # resize and remove alpha channel
    masked = (mask_stack * img) + ((1-mask_stack) * bg_resized)  # Blend
    masked = (masked * 255).astype('uint8')                      # Convert back to 8-bit 
    result.append(masked)
    
  # --- Add alpha channel for transparent background

  if not replacing and transparent:
    # merge RGB channels with the B&W mask we got before, which is used for creating the alpha channel
    # even if working with a single image, result must be an array 
    c_red, c_green, c_blue = cv2.split(img) # split image into channels
    result = [(cv2.merge((c_red, c_green, c_blue, mask_stack)) * 255).astype('uint8')]

  # --- Saving

  if save:
    for count, im in enumerate(result):
      create_folder_if_not_exist(save_folder)
      imgname = save_folder + save_name + '_bg{:03d}.png'.format(count)
      if cv2.imwrite(imgname, cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)) and debug:
        print('Saved in', imgname)

  return np.array(result)



def image_augment_background(img, mask, background, smooth = True):
  '''
    Replaces background from a given image by using the given mask and background image, see below for dtypes.
    Also provides error catching. For productioning, use instead the function `image_augment_background_minimal` with proper dtypes.

    Parameters:
        img (ndarray): Input image (uint8)
        mask (ndarray): Input mask for the image (boolean or binary int/float)
        background (ndarray): Image to set as background (float32)
        smooth (bool): If True, smooth the generated mask by using cv2 dilate-erode-smooth sequence

    Returns:
        images (ndarray): Resulting image with replaced backgrounds (uint8)
  '''

  # --- Check if the mask is valid

  if mask.shape != img.shape[:-1]:
    return img # some images may not have a mask

  # --- Binarize and smooth the mask

  mask_smoothed = mask.astype('float32')

  if True:
    mask_smoothed = cv2.dilate(mask_smoothed, None, iterations=5)
    mask_smoothed = cv2.erode(mask_smoothed, None, iterations=5)
    mask_smoothed = cv2.GaussianBlur(mask_smoothed, (5, 5), 0)

  # --- Blend masked img into backgrounds

  mask_stack = mask_smoothed[:,:,np.newaxis]    # Add 3rd dimension for broadcasting
  img        = img.astype('float32') / 255.0    # For easy blending

  try:
    masked = (mask_stack * img) + ((1-mask_stack) * background)  # Blend
    masked = (masked * 255).astype('uint8')
  except:
    print('\n\nFATAL ERROR IN BACKGROUND BLENDING! See shapes details below together with imshow result.')
    print('mask_stack.shape', mask_stack.shape)
    print('img.shape', img.shape)
    print('background.shape', background.shape)
    plt.imshow(img)
    plt.show()
    plt.imshow(background)
    plt.show()
    exit()
  
  return masked


def image_augment_background_minimal(img, mask, background):
  '''
    Replaces background from a given image by using the given (non-smoothed) mask and background.
    All the parameters must be dtype 'uint8' for a correct blending (so the mask cannot be smoothed).

    Parameters:
        img (ndarray): Input image (uint8)
        mask (ndarray): Input mask for the image (uint8)
        background (ndarray): Image to set as background (uint8)
    Returns:
        images (ndarray): Resulting image with replaced backgrounds (uint8)
  '''

  if mask.shape != img.shape[:-1]:
    return img # some images may not have a mask

  mask_stack = mask[:,:,np.newaxis]                             # Add 3rd dimension for broadcasting
  masked = (mask_stack * img) + ((1-mask_stack) * background)   # Blend
    
  return masked