# Copyright (C) 2021  Lafith Mattara

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# contact: lafithmattara@gmail.com

# Script for running MUSICA algorithm on a grayscale image:


import logging
import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand


logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
# change level here for console output
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('musica_py.log')
formatter = logging.Formatter(
        '%(asctime)s:%(levelname)s:%(name)s: %(message)s'
        )
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_format = logging.Formatter('%(levelname)s:\t%(message)s')
stream_handler.setFormatter(stream_format)
logger.addHandler(stream_handler)


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
    row, col = img.shape
    # check if dimensions are power of two
    # if not pad the image accordingly
    logger.debug("Calculating how much padding is required...")
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
    logger.info(
            'Image padded from [{},{}] to [{},{}]'.format(
                img.shape[0], img.shape[1],
                img_.shape[0], img_.shape[1]))
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
    logger.debug('Creating Gaussian pyramid...')
    # Gaussian Pyramid
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for layer in range(L):
        logger.debug('creating Layer %d...' % (layer+1))
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    logger.info('Finished creating Gaussian Pyramid')
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
    logger.debug('Creating Laplacian pyramid...')
    # Laplacian Pyramid:
    lp = []
    for layer in range(L):
        logger.debug('Creating layer %d' % (layer))
        tmp = pyramid_expand(gauss[layer+1], preserve_range=True)
        tmp = gauss[layer] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    logger.info("Finished creating Laplacian pyramid")
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
    logger.debug('Non linear transformation of coefficients...')
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']
    for layer in range(L):
        logger.info('Modifying Layer %d' % (layer))
        x = laplacian[layer]
        # removing all negative coefficients:
        # an attempt to reduce double edges
        x[x < 0] = 0.0
        G = a[layer]*M
        laplacian[layer] = G*np.multiply(
            np.divide(
                x, np.abs(x), out=np.zeros_like(x), where=x != 0),
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
    logger.debug('Reconstructing image...')
    # Reconstructing original image from laplacian pyramid
    rs = laplacian[L]
    for i in range(L-1, -1, -1):
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
        logger.debug('Layer %d completed' % (i))
    logger.info(
            'Finished reconstructing image from modified pyramid')
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
