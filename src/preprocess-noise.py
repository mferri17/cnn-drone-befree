import os
import sys
import time

import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from functions import general_utils


# from https://stackoverflow.com/a/42154921/10866825
def perlin(width, height, thickness=10):
    lin_w = np.linspace(0, thickness, width, endpoint=False)
    lin_h = np.linspace(0, thickness, height, endpoint=False)
    x,y = np.meshgrid(lin_w,lin_h) # FIX3: I thought I had to invert x and y here but it was a mistake
    
    # permutation table
    p = np.arange(256,dtype=int)
    np.random.shuffle(p)
    p = np.stack([p,p]).flatten()
    # coordinates of the top-left
    xi = x.astype(int)
    yi = y.astype(int)
    # internal coordinates
    xf = x - xi
    yf = y - yi
    # fade factors
    u = fade(xf)
    v = fade(yf)
    # noise components
    n00 = gradient(p[p[xi]+yi],xf,yf)
    n01 = gradient(p[p[xi]+yi+1],xf,yf-1)
    n11 = gradient(p[p[xi+1]+yi+1],xf-1,yf-1)
    n10 = gradient(p[p[xi+1]+yi],xf-1,yf)
    # combine noises
    x1 = lerp(n00,n10,u)
    x2 = lerp(n01,n11,u)
    return lerp(x1,x2,v)

def lerp(a,b,x):
    "linear interpolation"
    return a + x * (b-a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h,x,y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    g = vectors[h%4]
    return g[:,:,0] * x + g[:,:,1] * y

# ---------------------------------

base_path = 'C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/'
image = np.load(os.path.join(base_path, np.random.choice(os.listdir(base_path))), allow_pickle=True)['image']
image = (image / 255).astype('float32')

save = False
save_path = 'C:/Users/96mar/Desktop/meeting_dario/data/perlin-noise/'

if save:
    general_utils.create_folder_if_not_exist(save_path)
    width = 300
    height = 200
    thickness= 30
    nimages = 100
else:
    width = 108
    height = 60
    thickness = 10
    nimages = 1

for i in range(nimages):
    
    p1 = perlin(width, height, thickness)
    p2 = perlin(width, height, thickness)
    p3 = perlin(width, height, thickness)
    noise = np.dstack([p1, p2, p3])
    noise = noise / np.max(noise)   # rescaling in (-1, +1)
    noise = noise.astype(np.float32)

    if save:
        filename = os.path.join(save_path, 'perlin-{:03}.pickle'.format(i))
        with open(filename, 'wb') as fp:
            pickle.dump(noise, fp)

    else:
        fig, ax = plt.subplots(1, 3, figsize=(6, 2), subplot_kw={'xticks': [], 'yticks': []})
        fig.tight_layout()
        
        multiplier = np.random.triangular(left=0, mode=0.5, right=1)
        noise = noise * multiplier + 1  # rescaling in (1-multiplier, 1+multiplier)
        augmented_bw = np.multiply(image, noise[:,:,0,np.newaxis])
        augmented_rgb = np.multiply(image, noise)

        ax[0].imshow(noise)
        ax[1].imshow(augmented_bw)
        ax[2].imshow(augmented_rgb)
        plt.show()