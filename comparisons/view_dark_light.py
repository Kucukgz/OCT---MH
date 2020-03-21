"""
Image Processing Library.

Wow!
"""
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from operator import itemgetter
from collections import defaultdict


def _3d_get_dark_light(im):
    dark_light_list = []
    light_percent_list = []
    for i in range(im.shape[2]-1):
        img = im[:, :, i]
        # im - grayscale [0-255]
        # intensity palette of the image
        palette = defaultdict(int)
        for pixel in np.nditer(img):
            palette[int(pixel)] += 1
        # sort the intensity present in the image
        sorted_x = sorted(palette.items(), key=itemgetter(1), reverse=True)
        light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
        for _, x in enumerate(sorted_x[:pixel_limit]):
            if x[0] <= 20:  # dull : too much darkness
                dark_shade += x[1]
            if x[0] >= 240:  # bright : too much whiteness
                light_shade += x[1]
            shade_count += x[1]
        light_percent_list.append(round((float(light_shade)/shade_count)*100, 2))
        dark_light_list.append(round((float(dark_shade)/shade_count)*100, 2))
    # print(dark_light_list)
    return dark_light_list


def _view_dark_light(folder, names_max, names_middle, names_min, names):
    # write the name of which row max
    im1 = imread(folder + '/' + names_max[names.index('Darkness Level')]
                 + '.tif')
    print(folder + '/' + names_max[names.index('Darkness Level')]
                 + '.tif')
    img1 = np.moveaxis(im1, 0, -1)  # [z,x,y] -> [x,y,z]
    darkness_level_max = _3d_get_dark_light(img1)
    # write the name of which row min
    im2 = imread(folder + '/' + names_min[names.index('Darkness Level')]
                 + '.tif')
    print(folder + '/' + names_min[names.index('Darkness Level')]
                 + '.tif')
    img2 = np.moveaxis(im2, 0, -1)  # [z,x,y] -> [x,y,z]
    darkness_level_min = _3d_get_dark_light(img2)
    # select the row which has max value and min
    darkness_level_plane_max = np.where(darkness_level_max ==
                                        np.amax(darkness_level_max))[0]
    darkness_level_plane_max2 = np.where(darkness_level_max ==
                                         np.amin(darkness_level_max))[0]
    darkness_level_plane_min = np.where(darkness_level_min ==
                                        np.amin(darkness_level_min))[0]
    darkness_level_plane_min2 = np.where(darkness_level_min ==
                                         np.amax(darkness_level_min))[0]
    # select and read the image's plane (also switch format for view!)
    img1_darkness_max = img1[:, :, darkness_level_plane_max[0]]
    img1_darkness_max = Image.fromarray(img1_darkness_max)
    img1_darkness_max = img1_darkness_max.resize((400, 400))
    img1_1_darkness_max = img1[:, :, darkness_level_plane_max2[0]]
    img1_1_darkness_max = Image.fromarray(img1_1_darkness_max)
    img1_1_darkness_max = img1_1_darkness_max.resize((400, 400))
    img2_darkness_min = img2[:, :, darkness_level_plane_min[0]]
    img2_darkness_min = Image.fromarray(img2_darkness_min)
    img2_darkness_min = img2_darkness_min.resize((400, 400))
    img2_2_darkness_min = img2[:, :, darkness_level_plane_min2[0]]
    img2_2_darkness_min = Image.fromarray(img2_2_darkness_min)
    img2_2_darkness_min = img2_2_darkness_min.resize((400, 400))
    # figure all!
    plt.figure(1, figsize=(30, 30))
    plt.subplots_adjust(left=0.04, bottom=0.13, right=0.96,
                        top=0.87, wspace=0.47, hspace=0.71)
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Max Darkness Level', fontsize=10)
    ax.set_xlabel('That is ' + str(darkness_level_plane_max[0]) +
                  '. plane of\n' + names_max[names.index('Darkness Level')]
                  + '.tif' + '\n (That figure has max darkness level in'
                  + ' that tiff )', fontsize=6)
    plt.imshow(img1_darkness_max)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Max Darkness Level', fontsize=10)
    ax.set_xlabel('That is ' + str(darkness_level_plane_max2[0]) +
                  '. plane of\n' + names_max[names.index('Darkness Level')]
                  + '.tif' + '\n (That figure has min darkness level in'
                  + ' that tiff )', fontsize=6)
    plt.imshow(img1_1_darkness_max)
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Min Darkness Level', fontsize=10)
    ax.set_xlabel('That is ' + str(darkness_level_plane_min[0]) +
                  '. plane of\n' + names_min[names.index('Darkness Level')]
                  + '.tif' + '\n (That figure has min darkness level in'
                  + ' that tiff )', fontsize=6)
    plt.imshow(img2_darkness_min)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Min Darkness Level', fontsize=10)
    ax.set_xlabel('That is ' + str(darkness_level_plane_min2[0]) +
                  '. plane of\n' + names_min[names.index('Darkness Level')]
                  + '.tif' + '\n (That figure has max darkness Level in'
                  + ' that tiff)', fontsize=6)
    plt.imshow(img2_2_darkness_min)
    plt.savefig('Darkness Level Differences.png', dpi=600)
    # plt.show()
    return
