import os
from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb

import albumentations as A
import random

def replace_background(img, mask, background):
  if mask.shape != img.shape[:-1]:
    return img # some images may not have a mask

  mask_stack = mask[:,:,np.newaxis]                             # Add 3rd dimension for broadcasting
  masked = (mask_stack * img) + ((1-mask_stack) * background)   # Blend
    
  return masked


def augment_and_show(augmentation, samples, backgrounds):
    nplotted = len(samples)
    ncols = len(samples)
    nrows = 4

    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*3,nrows*2), subplot_kw={'xticks': [], 'yticks': []})
    fig.tight_layout()

    for counter, element in enumerate(samples):
        if len(backgrounds) > 0:
            img = replace_background(element['image'], element['mask'].astype('uint8'), backgrounds[np.random.randint(0, len(backgrounds))])
        else:
            img = element['image']
        # ax[0, counter].imshow(element['image'])
        # ax[1, counter].imshow(img)
        ax[0, counter].imshow(img)
        ax[1, counter].imshow(augmentation(image=img)['image'])
        ax[2, counter].imshow(augmentation(image=img)['image'])
        ax[3, counter].imshow(augmentation(image=img)['image'])

    plt.show()


nimages = 4
nbackgrounds = 100

# random.seed(42)
# np.random.seed(42)

samples_path = 'C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/'
sam_paths = os.listdir(samples_path)
np.random.shuffle(sam_paths)
sam_paths = sam_paths[:nimages]
samples = [np.load(os.path.join(samples_path, fn), allow_pickle=True) for fn in sam_paths]

if True:
    backgrounds_path = 'C:/Users/96mar/Desktop/meeting_dario/data/indoorCVPR_09_PPDario_uint8'
    bg_paths = list(map(str, Path(backgrounds_path).rglob('*.pickle')))
    np.random.shuffle(bg_paths)
    bg_paths = bg_paths[:nbackgrounds]
    backgrounds = [np.load(os.path.join(backgrounds_path, fn), allow_pickle=True) for fn in bg_paths]
else:
    backgrounds = []

my = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), p=1),    
    A.RandomGamma(p=1), 
    A.CLAHE(p=0.5), 
    A.HueSaturationValue(p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=100, p=1),
        A.ISONoise(intensity=(0,1), p=1),
    ], p=0.5),
], p=1)

print(my)
print(type(my) == A.core.composition.Compose)
print(my is not None)

# augment_and_show(my, samples, backgrounds)