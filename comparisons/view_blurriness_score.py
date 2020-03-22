"""
Image Processing Library.

Wow!
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from PIL import Image


def _3d_getting_blurriness_score(im):
    bs = np.zeros((im.shape[2]-1, 1))
    for i in range(im.shape[2]-1):
        print(str(i) + ' of ' + str(im.shape[2]-1))
        img = Image.fromarray(im[:, :, i])
        imlap = laplace(img)
        bs[i] = imlap.var()
    return bs


def _view_blurriness_score(folder, names_max, names_middle, names_min, names):
    # write the name of which row max
    im1 = imread(folder + '/' + names_max[names.index('Blurriness Level')] + '.tif')
    print(folder + '/' + names_max[names.index('Blurriness Level')] + '.tif')
    img1 = np.moveaxis(im1, 0, -1)  # [z,x,y] -> [x,y,z]
    bs_max = _3d_getting_blurriness_score(img1)
    # write the name of which row min
    im3 = imread(folder + '/' + names_min[names.index('Blurriness Level')] + '.tif')
    print(folder + '/' + names_min[names.index('Blurriness Level')] + '.tif')
    img3 = np.moveaxis(im3, 0, -1)  # [z,x,y] -> [x,y,z]
    bs_min = _3d_getting_blurriness_score(img3)
    # select the row which has max value and min
    bs_plane_max = np.where(bs_max == np.amax(bs_max))[0]
    bs_plane_min = np.where(bs_min == np.amin(bs_min))[0]
    # select and read the image's plane (also switch format for view!)
    im1 = img1[:, :, bs_plane_max[0]]
    im1 = Image.fromarray(im1)
    im1 = im1.resize((400, 400))
    im3 = img3[:, :, bs_plane_min[0]]
    im3 = Image.fromarray(im3)
    im3 = im3.resize((400, 400))
    # figure all!
    plt.figure(1, figsize=(20, 15))
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Max Blurriness Level', fontsize=10)
    ax.set_xlabel('That is ' + str(bs_plane_max[0]) + '. plane of\n'
                  + names_max[names.index('Blurriness Level')] + '.tif')
    plt.imshow(im1)
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Min Blurriness Level', fontsize=10)
    ax.set_xlabel('That is ' + str(bs_plane_min[0]) + '. plane of\n'
                  + names_min[names.index('Blurriness Level')] + '.tif')
    plt.imshow(im3)
    plt.savefig('Blurriness Differences.png', dpi=600)
    # plt.show()
    return
