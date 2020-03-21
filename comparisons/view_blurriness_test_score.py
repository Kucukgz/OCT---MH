"""
Image Processing Library.

Added filter for blurriness!

Wow!
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image
from scipy.ndimage import laplace


def _3d_getting_blurriness_score(im):
    bs = np.zeros((im.shape[2]-1, 1))
    for i in range(im.shape[2]-1):
        print(str(i) + ' of ' + str(im.shape[2]-1))
        img = Image.fromarray(im[:, :, i])
        imlap = laplace(img)
        bs[i] = imlap.var()
    return bs


def _3d_getting_blurriness_score_test(im):
    imfilter = gaussian_filter(im, sigma=3)
    bs_added_filter = _3d_getting_blurriness_score(imfilter)
    return bs_added_filter


def _view_blurriness_test_score(folder, names_max, names_middle, names_min, names):
    # write the name of which row max
    im1 = imread(folder + '/' + names_max[names.index('Blurriness Level Test')]
                 + '.tif')
    print(folder + '/' + names_max[names.index('Blurriness Level Test')]
                 + '.tif')
    img1 = np.moveaxis(im1, 0, -1)  # [z,x,y] -> [x,y,z]
    bs_added_filter_max = _3d_getting_blurriness_score_test(img1)
    # write the name of which row min
    im3 = imread(folder + '/' + names_min[names.index('Blurriness Level Test')]
                 + '.tif')
    print(folder + '/' + names_min[names.index('Blurriness Level Test')]
                 + '.tif')
    img3 = np.moveaxis(im3, 0, -1)  # [z,x,y] -> [x,y,z]
    bs_added_filter_min = _3d_getting_blurriness_score_test(img3)
    # select the row which has max value and min
    bs_added_filter_plane_max = np.where(bs_added_filter_max ==
                                         np.amax(bs_added_filter_max))[0]
    bs_added_filter_plane_min = np.where(bs_added_filter_min ==
                                         np.amin(bs_added_filter_min))[0]
    # select and read the image's plane (also switch format for view!)
    im1_filter = img1[:, :, bs_added_filter_plane_max[0]]
    im1_filter = Image.fromarray(im1_filter)
    im1_filter = im1_filter.resize((400, 400))
    im3_filter = img3[:, :, bs_added_filter_plane_min[0]]
    im3_filter = Image.fromarray(im3_filter)
    im3_filter = im3_filter.resize((400, 400))
    # figure all!
    plt.figure(1, figsize=(20, 15))
    ax = plt.subplot(1, 2, 1)
    ax.set_title('Max Blurriness Level Test', fontsize=10)
    ax.set_xlabel('That is ' + str(bs_added_filter_plane_max[0]) + '. plane of\n'
                  + names_max[names.index('Blurriness Level Test')] + '.tif')
    plt.imshow(im1_filter)
    ax = plt.subplot(1, 2, 2)
    ax.set_title('Min Blurriness Level Test', fontsize=10)
    ax.set_xlabel('That is ' + str(bs_added_filter_plane_min[0]) + '. plane of\n'
                  + names_min[names.index('Blurriness Level Test')] + '.tif')
    plt.imshow(im3_filter)
    plt.savefig('Blurriness Level Test Differences.png', dpi=600)
    # plt.show()
    return
