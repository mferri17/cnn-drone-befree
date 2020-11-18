################################################################
############ IMPORTS
#################

import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

### CUSTOM IMPORTS
sys.path.append('.')
from functions import general_utils
from functions import network_utils



################################################################
############ FUNCTIONS
#################

######### TEST 1 

def test1(data_folder, data_size, augmentation, bgs_folder, input_size):
  
  list_data = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
  list_data = list_data[:data_size]
  list_backgrounds = general_utils.list_files_in_folder(bgs_folder, 'pickle')

  tf_data = tf.data.Dataset.from_generator(data_generator, 
                                           args=[list_data, augmentation, list_backgrounds], 
                                           output_types=(tf.float32, tf.float32), output_shapes=(input_size, (4)))

  for xs, ys in tf_data.shuffle(buffer_size=len(list_data)).batch(8).take(3):
    print(ys.numpy())

def data_generator(files_path, augmentation, backgrounds_paths):
  for fn in files_path:
    with open(fn, 'rb') as fp:
      sample = pickle.load(fp)

    sample_img = sample_augmentation(sample, backgrounds_paths) if augmentation and backgrounds_paths is not None else sample['image']
    sample_x, sample_y = sample_preprocessing(sample_img, sample['gt'])
    yield sample_x, sample_y

def sample_augmentation(data, backgrounds_paths):
  with open(np.random.choice(backgrounds_paths), 'rb') as fp:
    bg = pickle.load(fp)
  return general_utils.image_augment_background(data['image'], data['mask'], background = bg)

def sample_preprocessing(img, gt):
  x_data = (255 - img).astype(np.float32)
  y_data = gt[0:4]
  return x_data, y_data

######### TEST 2

def test2(data_folder, data_size, augmentation, bgs_folder, input_size):

  list_data = [os.path.join(data_folder, fn) for fn in os.listdir(data_folder)]
  list_data = list_data[:data_size]
  # tf_data = tf.data.Dataset.list_files(data_folder + '*')

  list_backgrounds = general_utils.list_files_in_folder(bgs_folder, 'pickle')

  tf_data = tf.data.Dataset.from_tensor_slices(list_data)
  tf_data = tf_data.map(tf_parse_input)
  tf_data = tf_data.map(lambda i, m, g : tf_preprocessing(i, m, g, list_backgrounds))
  for xs, ys in tf_data.take(2):
    plt.imshow(xs.numpy().astype('uint8'))
    plt.title(ys.numpy())
    plt.show()
  print('--------------')

  # tf_data = tf_data.map(lambda fn : tf.py_function(parse_input_and_preprocess, [fn, list_backgrounds], [tf.float32, tf.float32], 'parse_input_and_preprocess'))
  # for xs, ys in tf_data.take(3):
  #   plt.imshow(xs.numpy().astype('uint8'))
  #   plt.title(ys.numpy())
  #   plt.show()
  # print('--------------')

def tf_read_file(file_path):
  return tf.io.read_file(file_path)

def tf_parse_input(filename):
  return tf.py_function(parse_input, [filename], [tf.float32, tf.float32, tf.float32], 'parse_input')

def parse_input(filename):
  with open(filename.numpy(), 'rb') as fp:
    sample = pickle.load(fp)
  return sample['image'], sample['mask'], sample['gt']

def tf_preprocessing(img, mask, gt, backgrounds_paths):
  return tf.py_function(preprocessing, [img, mask, gt, backgrounds_paths], [tf.float32, tf.float32], 'preprocessing')
  
def preprocessing(img, mask, gt, backgrounds_paths):
  sample_img = sample_augmentation({'image': img.numpy(), 'mask': mask.numpy()}, backgrounds_paths)
  sample_x, sample_y = sample_preprocessing(sample_img, gt.numpy())
  return sample_x, sample_y

def get_shapes(img, mask, gt):
  return img.numpy().shape, mask.numpy().shape, gt.numpy().shape
  
def parse_input_and_preprocess(filename, backgrounds_paths):
  with open(filename.numpy(), 'rb') as fp:
    sample = pickle.load(fp)
  sample_img = sample_augmentation(sample, backgrounds_paths)
  sample_x, sample_y = sample_preprocessing(sample_img, sample['gt'])
  return sample_x, sample_y

######### TEST 3

def test3():

  ### 1

  # def double(num):
  #   return tf.data.Dataset.from_tensors(num*2)

  # dataset = tf.data.Dataset.range(5).interleave(double)
  # print(list(dataset.as_numpy_iterator()))
  
  ### 2
  
  # dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
  # dataset = dataset.interleave(
  #   lambda x: tf.data.Dataset.from_tensors(x).repeat(6),
  #   cycle_length=2, block_length=4)

  # print(list(dataset.as_numpy_iterator()))

  ### 3

  # import string

  # samples = list(range(10))
  # augmenters = list(string.ascii_lowercase)

  # ds_sam = tf.data.Dataset.from_tensor_slices(samples)

  # ds_aug = tf.data.Dataset.from_tensor_slices(augmenters)
  # ds_aug = ds_aug.shuffle(len(augmenters)).take(len(samples)).repeat(2)
  
  # dataset = tf.data.Dataset.range(2).interleave(lambda x: tf.data.Dataset.from_tensors(ds_sam))
  # # dataset = ds_sam.interleave(ds_aug)

  # for i in dataset.as_numpy_iterator():
  #   print(i)

################################
### MAIN
######

if __name__ == "__main__":

  data_folder = 'C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/'
  data_size = 512
  augmentation = True
  bgs_folder = 'C:/Users/96mar/Desktop/meeting_dario/data/indoorCVPR_09_PPDario/'
  input_size = (60, 108, 3)

  print('\nPROGRAM STARTED\n')

  # test1(data_folder, data_size, augmentation, bgs_folder, input_size)
  # test2(data_folder, data_size, augmentation, bgs_folder, input_size)
  # test3()

  print('\nPROGRAM FINISHED')

