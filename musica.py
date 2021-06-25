# Script for running MUSICA algorithm on a grayscale image:
# Written by Lafith Mattara on 2021-06-05

import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand


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
    for _ in range(L):
        #print('creating Layer %d...' % (layer+1))
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
        #print('Creating layer %d' % (layer))
        tmp = pyramid_expand(gauss[layer+1], preserve_range=True)
        tmp = gauss[layer] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    return lp, gauss


# def enhance_coefficients_check(laplacian,L,a,p,M):
    # lp = [0]*L
    # for layer in range(L):
        # x = laplacian[layer]
        # a_ = a[layer]
        # p_ = p[layer]
        # new = np.zeros(laplacian[layer].shape)
        # for i in range(laplacian[layer].shape[0]):
            # for j in range(laplacian[layer].shape[1]):
                # xx = x[i,j]
                # if xx != 0:
                    # x_r = xx/abs(xx)
                    # new[i,j] = (a_*M)*x_r*((abs(xx)/M)**p_)
                # else:
                    # pass
        # lp[layer] = new
    # return lp

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
        G = a[layer]*M
        #print('Modifying Layer %d' % (layer))
        laplacian[layer] = G*np.multiply(
            np.divide(
                x, np.abs(x), out=np.zeros_like(x),where=x != 0),
            np.power(
                np.divide(
                    np.abs(x), M), p[layer]))
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
        #print('Layer %d completed' % (i))
    return rs


def musica(img, L, params):
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
    return rs
