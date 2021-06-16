# Script for running MUSICA algorithm on a grayscale image:
# Written by Lafith Mattara on 2021-06-05


from tifffile import TiffFile, imread
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.exposure as exposure
import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand


def display_tiff(path):
    """Function to read and display a TIFF file

    Parameters
    ----------
    path : str
        file path of TIFF image
    """
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
            out_range=(0, 255)).astype(np.uint16)
    plt.imshow(
            img_norm,
            cmap='gray', interpolation='none')
    plt.show()


def display_pyramid(pyramid):
    """Function for plotting all levels of an image pyramid

    Parameters
    ----------
    pyramid : list
        list containing all levels of the pyramid
    """
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
    """Function for reading a TIFF image

    Parameters
    ----------
    path : str
        file path to TIFF image

    Returns
    -------
    numpy.ndarray
        Image as a 2D numpy array
    """
    print('Reading TIFF file...')
    # read tiff image
    img = skio.imread(path, plugin='tifffile')
    # rescaling tiff into 0-255
    img = exposure.rescale_intensity(
            img, in_range='image',
            out_range=(0, 255)).astype(np.float64)
    return img


def isPowerofTwo(x):
    # check if number x is a power of two
    return x and (not(x & (x - 1)))


def findNextPowerOf2(n):
    # taken from https://www.techiedelight.com/round-next-highest-power-2/
    # Function will find next power of 2

    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1
    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1  # unset rightmost bit
    # `n` is now a power of two (less than `n`)
    # return next power of 2
    return n << 1


def resize_image(img):
    """MUSICA works for dimension like 2^N*2^M.
    Hence padding is required for arbitrary shapes

    Parameters
    ----------
    img : numpy.ndarray
        Original image

    Returns
    -------
    numpy.ndarray
        Resized image after padding
    """
    print('\nResizing image...')
    row, col = img.shape
    # check if dimensions are power of two
    # if not pad the image accordingly
    if isPowerofTwo(row):
        rowdiff = 0
    else:
        nextpower = findNextPowerOf2(row)
        rowdiff = nextpower - row

    if isPowerofTwo(col):
        coldiff = 0
    else:
        nextpower = findNextPowerOf2(col)
        coldiff = nextpower - col

    img_ = np.pad(
            img,
            ((0, rowdiff), (0, coldiff)),
            'reflect')
    return img_


def gaussian_pyramid(img, L):
    """Function for creating a Gaussian Pyramid

    Parameters
    ----------
    img : numpy.ndarray
        Input image or g0.
    L : Int
        Maximum level of decomposition.

    Returns
    -------
    list
        list containing images from g0 to gL in order
    """
    print('\nCreating Gaussian pyramid...')
    # Gaussian Pyramid
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for layer in range(L):
        print('creating Layer %d...' % (layer+1))
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    return gp


def laplacian_pyramid(img, L):
    """Function for creating Laplacian Pyramid

    Parameters
    ----------
    img : numpy.ndarray
        Input image or g0.
    L : Int
        Max layer of decomposition

    Returns
    -------
    list
        list containing laplacian layers from L_0 to L_L in order
    list
        list containing layers of gauss pyramid
    """
    gauss = gaussian_pyramid(img, L)
    print('\nCreating Laplacian pyramid...')
    # Laplacian Pyramid:
    lp = []
    for layer in range(L):
        print('Creating layer %d' % (layer))
        tmp = pyramid_expand(gauss[layer+1], preserve_range=True)
        tmp = gauss[layer] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    return lp, gauss


def enhance_coefficients(laplacian, L, params):
    """Non linear operation of pyramid coefficients

    Parameters
    ----------
    laplacian : list
        Laplacian pyramid of the image.
    L : Int
        Max layer of decomposition
    params : dict
        Store values of a, M and p.

    Returns
    -------
    list
        List of enhanced pyramid coeffiencts.
    """
    print('\nNon linear transformation of coefficients...')
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']
    for layer in range(L):
        x = laplacian[layer]
        x[x < 0] = 0.0  # removing all negative coefficients
        # x = np.abs(x)
        G = a[layer]*M
        print('Modifying Layer %d' % (layer))
        laplacian[layer] = G*np.multiply(
                    np.divide(
                        x, np.abs(x),
                        out=np.zeros_like(x),
                        where=x != 0),
                    np.power(
                        np.divide(
                            np.abs(x), M), p))
    return laplacian


def reconstruct_image(laplacian, L):
    """Function for reconstructing original image
    from a laplacian pyramid

    Parameters
    ----------
    laplacian : list
        Laplacian pyramid with enhanced coefficients
    L : int
        Max level of decomposition

    Returns
    -------
    numpy.ndarray
        Resultant image matrix after reconstruction.
    """
    print('\nReconstructing image...')
    # Reconstructing original image from laplacian pyramid
    rs = laplacian[L]
    for i in range(L-1, -1, -1):
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
        print('Layer %d completed' % (i))
    return rs


def musica(img, L, params, plot=False):
    """Function for running MUSICA algorithm

    Parameters
    ----------
    img : numpy.ndarray
        Input image
    L : int
        Max level of decomposition
    params : dict
        Contains parameter values required
        for non linear enhancement
    plot : bool, optional
        To plot the result, by default False

    Returns
    -------
    numpy.ndarray
        Final enhanced image with original dimensions
    """
    img_resized = resize_image(img)
    lp, _ = laplacian_pyramid(img_resized, L)
    lp = enhance_coefficients(lp, L, params)
    rs = reconstruct_image(lp, L)
    rs = rs[:img.shape[0], :img.shape[1]]

    if plot is True:
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(rs, cmap='gray')
        plt.title('After MUSICA')
        plt.show()
    return rs
