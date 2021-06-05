from tifffile import TiffFile, imread
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.exposure as exposure
import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand

def display_tiff(path):
    tif = TiffFile(path)
    info = {}
    info['pages'] = len(tif.pages)
    page0 = tif.pages[0]
    info['dtype'] = page0.dtype
    info['page0_dim'] = page0.shape 
    img = imread(path)
    info['img_shape'] = img.shape
    info['range'] = (img.min(), img.max())

    print(info)
    img_norm = exposure.rescale_intensity(
            img, in_range='image',
            out_range=(0,255)).astype(np.uint16)
    plt.imshow(
            img_norm,
            cmap='gray',interpolation='none')
    plt.show()


def display_pyramid(pyramid):
 rows, cols = pyramid[0].shape
 composite_image = np.zeros((rows, cols + (cols // 2)), dtype=np.double)
 composite_image[:rows, :cols] = pyramid[0]
 i_row = 0
 for p in pyramid[1:]:
     n_rows, n_cols = p.shape[:2]
     composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
     i_row += n_rows
 fig, ax = plt.subplots()
 ax.imshow(composite_image, cmap='gray')
 plt.show()


def read_tiff(path):
    print('Reading TIFF file...')
    # read tiff image
    img = skio.imread(path, plugin='tifffile')
    # rescaling tiff into 0-255
    img = exposure.rescale_intensity(
            img, in_range='image',
            out_range=(0,255)).astype(np.float64)
    return img


def resize_image(img):
    print('\nResizing image...')
    row, col=img.shape
    rowdiff = int(np.power(2,np.ceil(np.log2(row)))) - row
    coldiff = int(np.power(2,np.ceil(np.log2(col)))) - col
    img_=np.pad(img,((0,rowdiff),(0,coldiff)),'reflect')
    return img_


def gaussian_pyramid(img, L):
    print('\nCreating Gaussian pyramid...')
    # Gaussian Pyramid
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for l in range(L):
        print('creating Layer %d...'%(l+1))
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    return gp

def laplacian_pyramid(gauss, L):
    print('\nCreating Laplacian pyramid...')
    # Laplacian Pyramid:
    lp = []
    for l in range(L):
        print('Creating layer %d'%(l))
        tmp = pyramid_expand(gauss[l+1], preserve_range=True)
        tmp = gauss[l] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    return lp

def enhance_coefficients(laplacian, L, params):
    print('\nNon linear transformation of coefficients...')
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']
    for l in range(L):
        x = laplacian[l]
        G = a[l]*M
        print('Modifying Layer %d'%(l))
        laplacian[l] = G*np.multiply(
                    np.divide(
                        x,np.abs(x)),
                    np.power(
                        np.divide(
                            np.abs(x), M
                            ),p))     
    return laplacian

def reconstruct_image(laplacian, L):
    print('\nReconstructing image...')
    # Reconstructing original image from laplacian pyramid
    rs = laplacian[L]
    for i in range(L-1,-1,-1):
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
        print('Layer %d completed'%(i))
    return rs

def musica(img, L, params, plot=False):
    img_resized = resize_image(img)
    gp = gaussian_pyramid(img_resized,L)
    lp = laplacian_pyramid(gp, L)
    lp = enhance_coefficients(lp, L, params)
    rs = reconstruct_image(lp, L)
    rs = rs[:img.shape[0],:img.shape[1]]

    if plot == True:
        plt.subplot(1,2,1)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        plt.subplot(1,2,2)
        plt.imshow(rs, cmap='gray')
        plt.title('Reconstructed')
        plt.show()
    return rs

