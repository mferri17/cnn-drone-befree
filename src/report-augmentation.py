import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb

import albumentations as A
import random

sys.path.append('.')
from functions import general_utils


def replace_background(img, mask, background):
  if mask.shape != img.shape[:-1]:
    return img # some images may not have a mask

  mask_stack = mask[:,:,np.newaxis]                             # Add 3rd dimension for broadcasting
  masked = (mask_stack * img) + ((1-mask_stack) * background)   # Blend
    
  return masked



if __name__ == "__main__":

    
    augmentations = [
        {
            'name': 'Blur',
            'rules': [
                ('limit 3', A.Blur(blur_limit=3, p=1)),
                ('limit 6', A.Blur(blur_limit=6, p=1)),
            ]
        },
        {
            'name': 'ChannelDropout',
            'rules': [
                ('fill 0', A.ChannelDropout(fill_value=0, p=1)),
                ('fill 64', A.ChannelDropout(fill_value=64, p=1)),
                ('fill 128', A.ChannelDropout(fill_value=128, p=1)),
                ('fill 192', A.ChannelDropout(fill_value=192, p=1)),
                ('fill 255', A.ChannelDropout(fill_value=255, p=1)),
            ]
        },
        {
            'name': 'ChannelShuffle',
            'rules': [
                ('random', A.ChannelShuffle(p=1)),
                ('random', A.ChannelShuffle(p=1)),
                ('random', A.ChannelShuffle(p=1)),
            ]
        },
        {
            'name': 'CLAHE',
            'rules': [
                ('clip 4', A.CLAHE(clip_limit=(4, 4), p=1)),
                ('clip 8', A.CLAHE(clip_limit=(8, 8), p=1)),
                ('clip 16', A.CLAHE(clip_limit=(16, 16), p=1)),
                ('clip 32', A.CLAHE(clip_limit=(32, 32), p=1)),
            ]
        },
        {
            'name': 'Equalize',
            'rules': [
                ('classic', A.Equalize(by_channels=False, p=1)),
                ('by channel', A.Equalize(by_channels=True, p=1)),
            ]
        },
        {
            'name': 'GaussNoise',
            'rules': [
                ('limit 10', A.GaussNoise(var_limit=(10, 10), p=1)),
                ('limit 30', A.GaussNoise(var_limit=(30, 30), p=1)),
                ('limit 60', A.GaussNoise(var_limit=(60, 60), p=1)),
            ]
        },
        {
            'name': 'HueSaturationValue',
            'rules': [
                ('hue -20', A.HueSaturationValue(hue_shift_limit=(-20, -20), sat_shift_limit=0, val_shift_limit=0, p=1)),
                ('hue +20', A.HueSaturationValue(hue_shift_limit=(+20, +20), sat_shift_limit=0, val_shift_limit=0, p=1)),
                ('sat -30', A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(-30, -30), val_shift_limit=0, p=1)),
                ('sat +30', A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=(+30, +30), val_shift_limit=0, p=1)),
                ('val -20', A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(-20,-20), p=1)),
                ('val +20', A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=(+20,+20), p=1)),
                ('random', A.HueSaturationValue(p=1)),
            ]
        },
        {
            'name': 'InvertImg',
            'rules': [
                ('', A.InvertImg(p=1)),
            ]
        },
        {
            'name': 'ISONoise',
            'rules': [
                ('shift 0.04, intensity 0.2', A.ISONoise(color_shift=(0.04, 0.04), intensity=(0.2, 0.2), p=1)),
                ('shift 0.08, intensity 0.2', A.ISONoise(color_shift=(0.08, 0.08), intensity=(0.2, 0.2), p=1)),
                ('shift 0.06, intensity 0.4', A.ISONoise(color_shift=(0.06, 0.06), intensity=(0.4, 0.4), p=1)),
                ('shift 0.06, intensity 0.8', A.ISONoise(color_shift=(0.06, 0.06), intensity=(0.8, 0.8), p=1)),
            ]
        },
        {
            'name': 'MotionBlur',
            'rules': [
                ('limit 4', A.MotionBlur(blur_limit=(4,4), p=1)),
                ('limit 8', A.MotionBlur(blur_limit=(8,8), p=1)),
            ]
        },
        {
            'name': 'MultiplicativeNoise',
            'rules': [
                ('default', A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1)),
                ('per channel', A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, p=1)),
                ('elementwise', A.MultiplicativeNoise(multiplier=(0.8, 1.2), elementwise=True, p=1)),
                ('both', A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, elementwise=True, p=1)),
            ]
        },
        {
            'name': 'RandomBrightnessContrast',
            'rules': [
                ('brigh -0.2', A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2), contrast_limit=0, p=1)),
                ('brigh +0.2', A.RandomBrightnessContrast(brightness_limit=(+0.2, +0.2), contrast_limit=0, p=1)),
                ('contr -0.2', A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.2, -0.2), p=1)),
                ('contr +0.2', A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(+0.2, +0.2), p=1)),
            ]
        },
        {
            'name': 'RandomGamma',
            'rules': [
                ('gamma 40', A.RandomGamma(gamma_limit=(40, 40), p=1)),
                ('gamma 80', A.RandomGamma(gamma_limit=(80, 80), p=1)),
                ('gamma 120', A.RandomGamma(gamma_limit=(120, 120), p=1)),
                ('gamma 160', A.RandomGamma(gamma_limit=(160, 160), p=1)),
            ]
        },
        {
            'name': 'RGBShift',
            'rules': [
                ('range 20', A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1)),
                ('range 40', A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=1)),
                ('range 80', A.RGBShift(r_shift_limit=80, g_shift_limit=80, b_shift_limit=80, p=1)),
            ]
        },
        {
            'name': 'Solarize',
            'rules': [
                ('threshold 50', A.Solarize(threshold=(50, 50), p=1)),
                ('threshold 100', A.Solarize(threshold=(100, 100), p=1)),
                ('threshold 150', A.Solarize(threshold=(150, 150), p=1)),
                ('threshold 200', A.Solarize(threshold=(200, 200), p=1)),
            ]
        },
        {
            'name': 'ToGray',
            'rules': [
                ('', A.ToGray(p=1)),
            ]
        },
        {
            'name': 'ToSepia',
            'rules': [
                ('', A.ToSepia(p=1)),
            ]
        },
    ]

    # random.seed(42)
    # np.random.seed(42)

    img_test = plt.imread('C:/Users/96mar/Desktop/meeting_dario/data/women-gym.jpg')
    
    sp = 'C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/'
    sam_paths = os.listdir(sp)
    sample = np.load(os.path.join(sp, np.random.choice(sam_paths)), allow_pickle=True)

    bp = 'C:/Users/96mar/Desktop/meeting_dario/data/indoorCVPR_09_PPDario_uint8'
    bg_paths = list(map(str, Path(bp).rglob('*.pickle')))
    background = np.load(os.path.join(bp, np.random.choice(bg_paths)), allow_pickle=True)

    img = sample['image']
    replaced = general_utils.image_augment_background_minimal(sample['image'], sample['mask'].astype('uint8'), background)

    save_path = 'C:/Users/96mar/Desktop/meeting_dario/data/report-augmentations/'
    save_path = general_utils.create_datetime_folder(save_path)


    for group in augmentations:

        name = group['name']
        rules = group['rules']
        print('Computing', name, '...')

        # -- Performance evaluation
        
        time_iterations = 10000
        time_start = time.monotonic()
        time_augmenter = A.Compose([rules[-1][1]], p=1) # last augmentation rule
        for i in range(time_iterations):
            tmp = time_augmenter(image=replaced)['image']
        time_elaps = time.monotonic() - time_start

        # -- Example
        
        ncols = len(rules) + 1
        nrows = 2
        fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*3,nrows*2), subplot_kw={'xticks': [], 'yticks': []})

        fig.suptitle('{} ({:.2f}sec per {} iterations)'.format(name, time_elaps, time_iterations))
        ax[0, 0].imshow(img)
        ax[1, 0].imshow(replaced)

        for i, named_aug in enumerate(rules):
            title = named_aug[0]
            augmenter = A.Compose([named_aug[1]], p=1)

            ax[0, i+1].imshow(augmenter(image=img)['image'])
            ax[0, i+1].set_title(title)
            ax[1, i+1].imshow(augmenter(image=replaced)['image'])
            ax[1, i+1].set_title(title)

        plt.savefig(os.path.join(save_path, 'dataset-{}.jpg'.format(name)))
        plt.close()