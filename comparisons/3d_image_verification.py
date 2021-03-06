"""
Image Processing Library.

After obtaining Report.xlsx file from previous script('3d_images_features'), use this script !!

This script ends up some png files. That show differences and correctness of every features

Wow!
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from view_blurriness_score import _view_blurriness_score
from view_motions import _view_motions
from view_blurriness_test_score import _view_blurriness_test_score
from view_noise_level import _view_noise_level
from view_contrast_level import _view_contrast_level
from view_dark_light import _view_dark_light
from view_dominant_intensity import _view_dominant_intensity


# get minimum, middle and maximum value of all features from report
def _get_all_minmidmax_values(report, folder):
    all_data = pd.read_excel(report)
    all_data = pd.DataFrame(all_data)
    names = ['Pre-op Vision', 'Contrast Level',
             'Darkness Level', 'Average Pixel',
             'Blurriness Level Test', 'Blurriness Level',
             'Noise Level', 'Dominant Intensity',
             'Average Intensity']
    partly_data = pd.DataFrame(all_data, columns=names)
    names_max = []
    names_middle = []
    names_min = []
    # get max,middle and min values
    for i in partly_data.columns:
        print(i)
        max_value = max(all_data[i])
        which_row_max = all_data.loc[all_data[i] == max_value].index[0]
        # middle_value = all_data[i].median()
        # which_row_middle = all_data.loc[all_data[i] == middle_value].index[0]
        middle_value = sorted(all_data[i])
        which_row_middle = int(len(middle_value)/2)
        min_value = min(all_data[i])
        which_row_min = all_data.loc[all_data[i] == min_value].index[0]
        # get the name of row in max,middle and min value
        which_row_name_max = all_data['Original Filename'][which_row_max]
        names_max.append(all_data['Original Filename'][which_row_max])
        which_row_name_middle = all_data['Original Filename'][which_row_middle]
        names_middle.append(all_data['Original Filename'][which_row_middle])
        which_row_name_min = all_data['Original Filename'][which_row_min]
        names_min.append(all_data['Original Filename'][which_row_min])
        im1 = Image.open(folder + '/' + which_row_name_max + '.tif')
        im2 = Image.open(folder + '/' + which_row_name_middle + '.tif')
        im3 = Image.open(folder + '/' + which_row_name_min + '.tif')
        new_image = im1.resize((400, 400))
        new_image2 = im2.resize((400, 400))
        new_image3 = im3.resize((400, 400))
        plt.figure(1)  #, figsize=(20, 20))
        # plt.subplot(131)
        # ax = plt.gca()
        ax = plt.subplot(1, 3, 1)
        ax.set_title('Max ' + i, fontsize=10)
        ax.set_xlabel(which_row_name_max + '.tif')
        plt.imshow(new_image)
        # plt.subplot(132)
        # ax = plt.gca()
        ax = plt.subplot(1, 3, 2)
        ax.set_title('Middle ' + i, fontsize=10)
        ax.set_xlabel(which_row_name_middle + '.tif')
        plt.imshow(new_image2)
        # plt.subplot(133)
        # ax = plt.gca()
        ax = plt.subplot(1, 3, 3)
        ax.set_title('Min ' + i, fontsize=10)
        ax.set_xlabel(which_row_name_min + '.tif')
        plt.imshow(new_image3)
        plt.savefig(i + '.png', dpi=600)
        # plt.show()
    return names_max, names_middle, names_min, names


if __name__ == '__main__':
    # load image from home - main
    folder = 'comparisons/images'
    file = 'comparisons/OCT - MH - Dataset.csv'
    # get report file! -- which is generated by '3d_images_features'
    report = 'Report.xlsx'
    # check movement between planes!
    _view_motions(report, folder)
    # get maximum, minimum and middle values of all features from report file
    names_max, names_middle, names_min, names = _get_all_minmidmax_values(report, folder)
    # check comparisons for every features!
    _view_blurriness_score(folder, names_max, names_middle, names_min, names)
    _view_contrast_level(folder, names_max, names_middle, names_min, names)
    _view_noise_level(folder, names_max, names_middle, names_min, names)
    _view_blurriness_test_score(folder, names_max, names_middle, names_min, names)
    _view_dark_light(folder, names_max, names_middle, names_min, names)
    _view_dominant_intensity(folder, names_max, names_middle, names_min, names)
