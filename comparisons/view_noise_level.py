"""
Image Processing Library.

Wow!
"""

import numpy as np
from skimage.io import imread
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
from PIL import Image


def _3d_get_noise_level(im):
    # 3d image
    nl = np.zeros((im.shape[2], 1))
    for i in range(im.shape[2]):
        nl[i] = estimate_sigma(
                  im[:, :, i], multichannel=False, average_sigmas=True)
    return nl


def _view_noise_level(folder, names_max, names_middle, names_min, names):
    # write the name of which row max
    img1 = imread(folder + '/' + names_max[names.index('Noise Level')] + '.tif')
    img1 = np.moveaxis(img1, 0, -1)  # [z,x,y] -> [x,y,z]
    print(folder + '/' + names_max[names.index('Noise Level')] + '.tif')
    nl_max = _3d_get_noise_level(img1)
    # write the name of which row min
    img2 = imread(folder + '/' + names_min[names.index('Noise Level')] + '.tif')
    img2 = np.moveaxis(img2, 0, -1)  # [z,x,y] -> [x,y,z]
    print(folder + '/' + names_min[names.index('Noise Level')] + '.tif')
    nl_min = _3d_get_noise_level(img2)
    # select the row which has max value and min
    nl_plane_max = np.where(nl_max == np.amax(nl_max))[0]
    nl_plane_max2 = np.where(nl_max == np.amin(nl_max))[0]
    nl_plane_min = np.where(nl_min == np.amin(nl_min))[0]
    nl_plane_min2 = np.where(nl_min == np.amax(nl_min))[0]
    # select and read the image's plane
    # also switch format for view!
    im1 = img1[:, :, nl_plane_max[0]]
    im1_2 = img1[:, :, nl_plane_max2[0]]
    im1 = Image.fromarray(im1)
    im1_2 = Image.fromarray(im1_2)
    im1 = im1.resize((400, 400))
    im1_2 = im1_2.resize((400, 400))
    im2 = img2[:, :, nl_plane_min[0]]
    im2_2 = img2[:, :, nl_plane_min2[0]]
    im2 = Image.fromarray(im2)
    im2_2 = Image.fromarray(im2_2)
    im2 = im2.resize((400, 400))
    im2_2 = im2_2.resize((400, 400))
    # figure all!
    plt.figure(1, figsize=(30, 30))
    plt.subplots_adjust(left=0.04, bottom=0.13, right=0.96,
                        top=0.87, wspace=0.47, hspace=0.71)
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Max Noise Level', fontsize=10)
    ax.set_xlabel('That is ' + str(nl_plane_max[0]) + '. plane of\n'
                  + names_max[names.index('Noise Level')] + '.tiff' +
                  '\n (That figure has max noise level in that tiff )',
                  fontsize=6)
    plt.imshow(im1)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Max Noise Level', fontsize=10)
    ax.set_xlabel('That is ' + str(nl_plane_max2[0]) + '. plane of\n'
                  + names_max[names.index('Noise Level')] + '.tif' +
                  '\n (That figure has min noise level in that tiff )',
                  fontsize=6)
    plt.imshow(im1_2)
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Min Noise Level', fontsize=10)
    ax.set_xlabel('That is ' + str(nl_plane_min2[0]) + '. plane of\n'
                  + names_min[names.index('Noise Level')] + '.tif' +
                  '\n (That figure has max noise level in that tiff )',
                  fontsize=6)
    plt.imshow(im2)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Min Noise Level', fontsize=10)
    ax.set_xlabel('That is ' + str(nl_plane_min[0]) + '. plane of\n'
                  + names_min[names.index('Noise Level')] + '.tif' +
                  '\n (That figure has min noise level in that tiff )',
                  fontsize=6)
    plt.imshow(im2_2)
    plt.savefig('Noise Level Differences.png', dpi=600)
    # plt.show()
    return
