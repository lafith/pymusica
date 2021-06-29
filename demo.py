from musica import musica
import numpy as np
import matplotlib.pyplot as plt


# defining parameters:
# max level of pyramid
L = 11
a = np.full(L, 11)
params = {
        'M': 255.0,
        'a': a,
        'p': 0.4
        }
# reading grayscale image
img = plt.imread('sample.jpg')  # gray scale image
# run MUSICA:
img_enhanced = musica(img, L, params)

# show and save result
plt.figure()
plt.subplot(121)
plt.title("Original Image")
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.title("After Enhancement")
plt.imshow(img_enhanced, cmap="gray")
plt.show()
plt.imsave("contrast_enhanced.jpg", img_enhanced, cmap="gray")
