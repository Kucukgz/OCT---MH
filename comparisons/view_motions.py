"""
Image Processing Library.

Wow!
"""

import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from pyoptflow import HornSchunck
import pandas as pd
from PIL import Image


def _motion_estimation(im1, im2, a=1.0, n=100):
    u, v = HornSchunck(im1, im2, alpha=a, Niter=n)
    m = np.sqrt(np.power(u, 2) + np.power(v, 2))  # magnitude
    a = np.arctan2(u, v)  # angle
    return m, v, u, a


def _get_optical_flow_z(im, window_size, tau=1e-2):
    # 3d image
    mz = np.zeros((im.shape[2]-1, 1))
    for i in range(im.shape[2]-1):
        print(str(i) + ' of ' + str(im.shape[2]-1))
        _, _, m, _ = _motion_estimation(
                        im[:, :, i], im[:, :, i+1])
        mz[i] = np.sum(m)
    mz.sort()
    mzn = mz / (im.shape[0] * im.shape[1] * im.shape[2])
    mzm = np.max(mz)
    mzi = np.min(mz)
    mznm = np.max(mzn)
    mzni = np.min(mzn)
    # print('max_optical_flow_z: ' + str(mzm) + ', ' + str(mznm))
    return mz, mzn, mzm, mznm, mzi, mzni


# get just MZM and MZNM features' min and max values
def _view_motions(report, folder):
    all_data = pd.read_excel(report)
    all_data = pd.DataFrame(all_data)
    names = ['Motion Est.', 'Normalised Motion Est.']
    partly_data = pd.DataFrame(all_data, columns=names)
    # get max,middle and min values
    for i in partly_data.columns:
        print(i)
        max_value = max(all_data[i])
        which_row_max = all_data.loc[all_data[i] == max_value].index[0]
        middle_value = all_data[i].median()
        which_row_middle = all_data.loc[all_data[i] == middle_value].index[0]
        min_value = min(all_data[i])
        which_row_min = all_data.loc[all_data[i] == min_value].index[0]
        # get the name of row in max,middle and min value
        which_row_name_max = all_data['Original Filename'][which_row_max]
        which_row_name_middle = all_data['Original Filename'][which_row_middle]
        which_row_name_min = all_data['Original Filename'][which_row_min]
        # get a plane number of max value of tiff file
        # the plane which has MAX mz value is shown
        im1 = imread(folder + '/' + which_row_name_max + '.tif')
        img1 = np.moveaxis(im1, 0, -1)  # [z,x,y] -> [x,y,z]
        mz, mzn, mzm, mznm, mzi, mzni = _get_optical_flow_z(img1, 20, tau=1e-2)
        if i == 'Motion Est.':
            which_plane_max = np.where(mz == mzm)[0]
        elif i == 'Normalised Motion Est.':
            which_plane_max = np.where(mzn == mznm)[0]
        im1 = img1[:, :, which_plane_max[0]]
        im1_2 = img1[:, :, which_plane_max[0]+1]            # NEXT ONE
        im1 = Image.fromarray(im1)                # change format to figure it
        im1_2 = Image.fromarray(im1_2)            # change format to figure it
        # Plane which has MIDDLE mz value is shown
        im2 = imread(folder + '/' + which_row_name_middle + '.tif')
        img2 = np.moveaxis(im2, 0, -1)  # [z,x,y] -> [x,y,z]
        mz, mzn, mzm, mznm, mzi, mzni = _get_optical_flow_z(img2, 15, tau=1e-2)
        which_plane_middle = int(len(mz)/2)
        im2 = img2[:, :, which_plane_middle]
        im2_2 = img2[:, :, which_plane_middle+4]            # NEXT ONE
        im2 = Image.fromarray(im2)                # change format to figure it
        im2_2 = Image.fromarray(im2_2)            # change format to figure it
        # Plane which has MIN mz value is shown
        im3 = imread(folder + '/' + which_row_name_min + '.tif')
        img3 = np.moveaxis(im3, 0, -1)  # [z,x,y] -> [x,y,z]
        mz, mzn, mzm, mznm, mzi, mzni = _get_optical_flow_z(img3, 15, tau=1e-2)
        if i == 'Motion Est.':
            which_plane_min = np.where(mz == mzi)[0]
        elif i == 'Normalised Motion Est.':
            which_plane_min = np.where(mzn == mzni)[0]
        im3 = img3[:, :, which_plane_min[0]]
        im3_2 = img3[:, :, which_plane_min[0]+1]            # NEXT ONE
        im3 = Image.fromarray(im3)                # change format to figure it
        im3_2 = Image.fromarray(im3_2)            # change format to figure it
        # overllapping two planes !
        im_max_result = Image.blend(im1, im1_2, 0.5)
        im_middle_result = Image.blend(im2, im2_2, 0.5)
        im_min_result = Image.blend(im3, im3_2, 0.5)
        # do the same size and figure all !
        im_max_result = im_max_result.resize((400, 400))
        im_middle_result = im_middle_result.resize((400, 400))
        im_min_result = im_min_result.resize((400, 400))
        # figure all !
        plt.figure(1, figsize=(10, 10))
        plt.subplots_adjust(wspace=1)
        # Max
        ax = plt.subplot(1, 3, 1)
        ax.set_title('Max ' + i, fontsize=10)
        ax.set_xlabel('That is ' + str(which_plane_max[0]) +
                      '. and next plane of\n' + which_row_name_max + '.tif')
        plt.imshow(im_max_result)
        # Middle
        ax = plt.subplot(1, 3, 2)
        ax.set_title('Middle ' + i, fontsize=10)
        ax.set_xlabel('That is ' + str(which_plane_middle) +
                      '. and next plane of\n' + which_row_name_middle + '.tif')
        plt.imshow(im_middle_result)
        # Min
        ax = plt.subplot(1, 3, 3)
        ax.set_title('Min ' + i, fontsize=10)
        ax.set_xlabel('That is ' + str(which_plane_min[0]) +
                      '. and next plane of\n' + which_row_name_min + '.tif')
        plt.imshow(im_min_result)
        plt.savefig(i + '.png', dpi=600)
        # plt.show()
