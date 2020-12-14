import os
import time
import numpy as np
import matplotlib.pyplot as plt


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



def perlin_noise(width, height, multiplier):
    noise = perlin(width, height)
    noise = noise * multiplier / np.max(noise) + 1 # rescaling in (1-multiplier, 1+multiplier)
    noise = noise[:,:,np.newaxis]
    return noise

# ---------------------------------

base_path = 'C:/Users/96mar/Desktop/meeting_dario/data/orig_train_63720/'
image = np.load(os.path.join(base_path, np.random.choice(os.listdir(base_path))), allow_pickle=True)['image']
image = (image / 255).astype('float32')

width = image.shape[1]
height = image.shape[0]
multiplier = 0.5

measure = False

if measure:
    time_iterations = 100
    time_start = time.monotonic()
    for i in range(time_iterations):
        noise = perlin(width, height)
        noise = noise * multiplier / np.max(noise) + 1
        noise = noise[:,:,np.newaxis]
    time_elaps = time.monotonic() - time_start
    print('Time: {:.2f}sec per {} iterations'.format(time_elaps, time_iterations))

else:
    noise = perlin_noise(width, height, multiplier)
    fig, ax = plt.subplots(1, 2, figsize=(6, 2), subplot_kw={'xticks': [], 'yticks': []})
    fig.tight_layout()
    
    ax[0].imshow(noise)
    ax[1].imshow(np.multiply(image, noise))
    plt.show()