"""
Image Processing Library.

Wow!
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image


def _3d_getting_contrast_level(im):
    cl = np.zeros((im.shape[2]-1, 1))
    for i in range(im.shape[2]-1):
        print(str(i) + ' of ' + str(im.shape[2]-1))
        imgx, imgy, imgz = np.gradient(im)
        img = np.sqrt(np.power(imgz[:, :, i], 2))
        cl[i] = np.sum(img) / (im.shape[0] * im.shape[1])
    return cl


def _view_contrast_level(folder, names_max, names_middle, names_min, names):
    # write the name of which row max
    img1 = imread(folder + '/' + names_max[names.index('Contrast Level')] + '.tif')
    img1 = np.moveaxis(img1, 0, -1)  # [z,x,y] -> [x,y,z]
    print(folder + '/' + names_max[names.index('Contrast Level')] + '.tif')
    cl_max = _3d_getting_contrast_level(img1)
    # write the name of which row min
    img2 = imread(folder + '/' + names_max[names.index('Contrast Level')] + '.tif')
    img2 = np.moveaxis(img2, 0, -1)  # [z,x,y] -> [x,y,z]
    print(folder + '/' + names_max[names.index('Contrast Level')] + '.tif')
    cl_min = _3d_getting_contrast_level(img2)
    # select the row which has max value and min
    cl_plane_max = np.where(cl_max == np.amax(cl_max))[0]
    cl_plane_min = np.where(cl_min == np.amin(cl_min))[0]
    # select and read the image's plane and next one
    # also switch format for view!
    im1 = img1[:, :, cl_plane_max[0]]
    im1_1 = img1[:, :, cl_plane_max[0]+1]
    im1 = Image.fromarray(im1)
    im1_1 = Image.fromarray(im1_1)
    im1 = im1.resize((400, 400))
    im1_1 = im1_1.resize((400, 400))
    im2 = img2[:, :, cl_plane_min[0]]
    im2_2 = img2[:, :, cl_plane_min[0]+1]
    im2 = Image.fromarray(im2)
    im2_2 = Image.fromarray(im2_2)
    im2 = im2.resize((400, 400))
    im2_2 = im2_2.resize((400, 400))
    # figure all!
    plt.figure(1, figsize=(30, 30))
    plt.subplots_adjust(left=0.04, bottom=0.13, right=0.96,
                        top=0.87, wspace=0.47, hspace=0.71)
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Max Contrast Level', fontsize=9)
    ax.set_xlabel('That is ' + str(cl_plane_max[0]) + '. plane of\n'
                  + names_max[names.index('Contrast Level')] + '.tif',
                  fontsize=7)
    plt.imshow(im1)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Max Contrast Level', fontsize=9)
    ax.set_xlabel('That is ' + str(cl_plane_max[0]+1) + '. plane of\n'
                  + names_max[names.index('Contrast Level')] + '.tif',
                  fontsize=7)
    plt.imshow(im1_1)
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Min Contrast Level', fontsize=9)
    ax.set_xlabel('That is ' + str(cl_plane_min[0]) + '. plane of\n'
                  + names_min[names.index('Contrast Level')] + '.tif',
                  fontsize=7)
    plt.imshow(im2)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Min Contrast Level', fontsize=9)
    ax.set_xlabel('That is ' + str(cl_plane_min[0]+1) + '. plane of\n'
                  + names_min[names.index('Contrast Level')] + '.tif',
                  fontsize=7)
    plt.imshow(im2_2)
    plt.savefig('Contrast Level Differences.png', dpi=600)
    # plt.show()
    return
