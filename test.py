from musica import musica
import numpy as np
import matplotlib.pyplot as plt


# defining parameters:
L = 11
a = np.full(L, 11)
params = {
        'M': 255.0,
        'a': a,
        'p': 0.4 
        }
# reading grayscale image
img = plt.imread('sample.jpg') # gray scale image
# run MUSICA:
rs = musica(img, L, params, plot=True)
