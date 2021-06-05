from musica import read_tiff
from musica import musica 
import numpy as np


L = 6
a = np.full(L, 1.8)
params = {
        'M': 255.0,
        'a': a,
        'p': 0.8
        }

path = '../images/01.TIFF'
img = read_tiff(path)
rs = musica(img, L, params, plot=True)

