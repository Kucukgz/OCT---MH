"""
Image Processing Library.

Wow!
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread
from sklearn.cluster import KMeans


def _3d_get_dominant_intensity(im):
    dominant_intensity_list = []
    for i in range(im.shape[2]-1):
        # reshape
        img = im[:, :, i]
        x, y = img.shape
        imr = img.reshape(x, y)
        # k-means
        kmeans_cluster = KMeans(n_clusters=5)
        kmeans_cluster.fit(imr)
        cluster_centers = kmeans_cluster.cluster_centers_
        cluster_labels = kmeans_cluster.labels_
        # dominant intensity
        palette = np.uint8(cluster_centers)
        dominant_intensity = palette[np.argmax(
                                    np.unique(cluster_labels, return_counts=True)[1])]
        # from vector [1,...,z] - > 1 number
        dominant_intensity_list.append(np.median(dominant_intensity))
        # print(dominant_intensity_list)
    return dominant_intensity_list


def _view_dominant_intensity(folder, names_max, names_middle, names_min, names):
    # write the name of which row max
    im1 = imread(folder + '/' + names_max[names.index('Dominant Intensity')]
                 + '.tif')
    print(folder + '/' + names_max[names.index('Dominant Intensity')]
                 + '.tif')
    img1 = np.moveaxis(im1, 0, -1)  # [z,x,y] -> [x,y,z]
    dominant_intensity_max = _3d_get_dominant_intensity(img1)
    # write the name of which row min
    im2 = imread(folder + '/' + names_min[names.index('Dominant Intensity')]
                 + '.tif')
    print(folder + '/' + names_min[names.index('Dominant Intensity')]
                 + '.tif')
    img2 = np.moveaxis(im2, 0, -1)  # [z,x,y] -> [x,y,z]
    dominant_intensity_min = _3d_get_dominant_intensity(img2)
    # select the row which has max value and min
    dominant_intensity_plane_max = np.where(dominant_intensity_max ==
                                            np.amax(dominant_intensity_max))[0]
    dominant_intensity_plane_max2 = np.where(dominant_intensity_max ==
                                             np.amin(dominant_intensity_max))[0]
    dominant_intensity_plane_min = np.where(dominant_intensity_min ==
                                            np.amin(dominant_intensity_min))[0]
    dominant_intensity_plane_min2 = np.where(dominant_intensity_min ==
                                             np.amax(dominant_intensity_min))[0]
    # select and read the image's plane (also switch format for view!)
    img1_dominant_max = img1[:, :, dominant_intensity_plane_max[0]]
    img1_dominant_max = Image.fromarray(img1_dominant_max)
    img1_dominant_max = img1_dominant_max.resize((400, 400))
    img1_1_dominant_max = img1[:, :, dominant_intensity_plane_max2[0]]
    img1_1_dominant_max = Image.fromarray(img1_1_dominant_max)
    img1_1_dominant_max = img1_1_dominant_max.resize((400, 400))
    img2_dominant_min = img2[:, :, dominant_intensity_plane_min[0]]
    img2_dominant_min = Image.fromarray(img2_dominant_min)
    img2_dominant_min = img2_dominant_min.resize((400, 400))
    img2_2_dominant_min = img2[:, :, dominant_intensity_plane_min2[0]]
    img2_2_dominant_min = Image.fromarray(img2_2_dominant_min)
    img2_2_dominant_min = img2_2_dominant_min.resize((400, 400))
    # figure all!
    plt.figure(1, figsize=(30, 30))
    plt.subplots_adjust(left=0.04, bottom=0.13, right=0.96,
                        top=0.87, wspace=0.47, hspace=0.71)
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Max Dominant Intensity', fontsize=10)
    ax.set_xlabel('That is ' + str(dominant_intensity_plane_max[0]) +
                  '. plane of\n' + names_max[names.index('Dominant Intensity')]
                  + '.tif' + '\n (That figure has max dominant intensity in'
                  + ' that tiff )', fontsize=6)
    plt.imshow(img1_dominant_max)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Max Dominant Intensity', fontsize=10)
    ax.set_xlabel('That is ' + str(dominant_intensity_plane_max2[0]) +
                  '. plane of\n' + names_max[names.index('Dominant Intensity')]
                  + '.tif' + '\n (That figure has min dominant intensity in'
                  + ' that tiff )', fontsize=6)
    plt.imshow(img1_1_dominant_max)
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Min Dominant Intensity', fontsize=10)
    ax.set_xlabel('That is ' + str(dominant_intensity_plane_min[0]) +
                  '. plane of\n' + names_min[names.index('Dominant Intensity')]
                  + '.tif' + '\n (That figure has min dominant intensity in'
                  + ' that tiff )', fontsize=6)
    plt.imshow(img2_dominant_min)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Min Dominant Intensity', fontsize=10)
    ax.set_xlabel('That is ' + str(dominant_intensity_plane_min2[0]) +
                  '. plane of\n' + names_min[names.index('Dominant Intensity')]
                  + '.tif' + '\n (That figure has max dominant intensity in'
                  + ' that tiff)', fontsize=6)
    plt.imshow(img2_2_dominant_min)
    plt.savefig('Dominant Intensity Differences.png', dpi=600)
    # plt.show()
    return
