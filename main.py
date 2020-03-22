"""
Image Processing Library.

That script generated and reported all features and their values.

Firstly, it checks that whether the filenames in the given excel file matches the existing image names, or not.

Then, it controls the correctness with hashing. For example, if two different images can be named the same
but with hashing it enables is it the same image or not. Therefore, it can find dublicate images.

After that, it gets blurriness score, blurriness score by adding gaussiand filter, average pixel width, average intensity
darkness and whiteness, dominant intensity, noise level, contrast level, and motion estimation between every plane.

Lastly, it recorded as an excel file and do histogram.

Wow!
"""

import os
import xlsxwriter
from operator import itemgetter
from collections import defaultdict
import numpy as np
from skimage.io import imread
from skimage.restoration import estimate_sigma
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage import laplace
from scipy.ndimage import gaussian_filter
from pyoptflow import HornSchunck
from skimage import feature
from skimage.color import rgb2gray
import pandas as pd
import hashlib


# That func controls the filenames in the excel file match to the image names
# in the folder of images
def _files_control(folder, file):
    folderlist = os.listdir(folder)
    folderlist = [w.replace('.tif', '') for w in folderlist]
    filenames_list = np.genfromtxt(file, dtype=None, encoding=None,
                                   delimiter=',', skip_header=1, usecols=0)
    filenames_list = list(filenames_list)
    differences = list(set(folderlist) & set(filenames_list))
    if (len(folderlist) == len(differences) and
            len(filenames_list) == len(differences)):
        print('1')
    else:
        print('0 - completely matches! ')
    return folder, file


def _find_duplicate_file(folder):
    # dups in format {hash:[names]}
    dups = {}
    for dirname, subdirs, filelist in os.walk(folder):
        print('Scanning %s...' % dirname)
        for filename in filelist:
            # get the path to the file
            path = os.path.join(dirname, filename)
            # calculate hash
            file_hash = _hash_file(path)
            print(file_hash)
            # add or append the file path
            if file_hash in dups:
                dups[file_hash].append(path)
            else:
                dups[file_hash] = [path]
    return dups


def _hash_file_name(filename):
    file_hash = _hash_file(filename)
    return file_hash


def _hash_file(path, blocksize=65536):
    afile = open(path, 'rb')
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    # while len(buf) > 0:
    while buf:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def _print_duplicate_file(dict):
    results = list(filter(lambda x: len(x) > 1, dict.values()))
    # if len(results) > 0:
    if results:
        print('Duplicates Found:')
        print('The following files are identical. The name could differ,' +
              'but the content is identical')
        print('___________________')
        for result in results:
            for subresult in result:
                print('\t\t%s' % subresult)
        print('___________________')
    else:
        print('No duplicate files found. ✓ ')


def _display(im):
    # display max image
    im = np.amax(im, axis=2)
    plt.imshow(im)
    plt.show()


def _get_blurriness_score(im):
    # im - greyscale image
    imlap = laplace(im)
    bs = imlap.var()
    return bs


def _get_blurriness_score_test(im):
    im2 = gaussian_filter(im, sigma=3)
    bs2 = _get_blurriness_score(im2)
    return bs2


def _get_average_pixel_width(im):
    # im - grayscale
    if len(im.shape) == 3:
        im = rgb2gray(im)
    imedges = feature.canny(im, sigma=3)
    avg_width = (float(np.sum(imedges)) / (im.shape[0]*im.shape[1]))
    return avg_width*100


def _get_average_intensity(im):
    ai = im.mean()
    return ai


def _get_dark_light(im):
    # im - grayscale [0-255]
    # intensity palette of the image
    palette = defaultdict(int)
    for pixel in np.nditer(im):
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
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent


def _get_dominant_intensity(im):
    # reshape
    x, y, z = im.shape
    imr = im.reshape(x*y, z)
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
    dominant_intensity = np.median(dominant_intensity)
    # quantized image
    imq = palette[cluster_labels.flatten()]
    imq = imq.reshape(im.shape)
    return dominant_intensity, imq


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
    mzn = mz / (im.shape[0] * im.shape[1] * im.shape[2])
    mzm = np.max(mz)
    mznm = np.max(mzn)
    # print('max_optical_flow_z: ' + str(mzm) + ', ' + str(mznm))
    return mz, mzn, mzm, mznm


def _display_optical_flow(im1, im2, u, v, m, a):
    plt.subplot(3, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('im1')
    plt.imshow(im1)
    plt.subplot(3, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('im2')
    plt.imshow(im2)
    plt.subplot(3, 2, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title('u')
    plt.imshow(u)
    plt.subplot(3, 2, 4)
    plt.xticks([])
    plt.yticks([])
    plt.title('v')
    plt.imshow(v)
    plt.subplot(3, 2, 5)
    plt.xticks([])
    plt.yticks([])
    plt.title('m')
    plt.imshow(m)
    plt.subplot(3, 2, 6)
    plt.xticks([])
    plt.yticks([])
    plt.title('a')
    plt.imshow(a)
    plt.show()


def _display_optical_flow_z(mz, mzn):
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(mz)
    ax2.plot(mzn)
    ax1.set_title('mz')
    ax2.set_title('mzn')
    plt.show()


def _get_noise_level(im):
    # 3d image
    nl = np.zeros((im.shape[2], 1))
    for i in range(im.shape[2]):
        nl[i] = estimate_sigma(
                  im[:, :, i], multichannel=False, average_sigmas=True)
    nlm = np.mean(nl)
    return nl, nlm


def _get_contrast_level(im):
    imgx, imgy, imgz = np.gradient(im)
    img = np.sqrt(np.power(imgx, 2) + np.power(imgy, 2) + np.power(imgz, 2))
    cl = np.sum(img) / (im.shape[0] * im.shape[1] * im.shape[2])
    return cl


# get some features all together
def _image_features_3d(filename):
    im = imread(filename)
    im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]
    # check image quality
    bs = _get_blurriness_score(im)
    bs2 = _get_blurriness_score_test(im)
    ai = _get_average_intensity(im)
    lp, dp, = _get_dark_light(im)
    di, imq = _get_dominant_intensity(im)
    avg_width = _get_average_pixel_width(im)
    nl, nlm = _get_noise_level(im)
    cl = _get_contrast_level(im)
    return im, bs, bs2, ai, lp, dp, di, imq, avg_width, nl, nlm, cl


# read the first excel file for the next steps
def _get_main_info(file):
    data = pd.read_csv(file)
    data = data.loc[data["Original Filename"].str.lower().sort_values().index]
    org_name = pd.DataFrame(data, columns=['Original Filename'])
    new_name = pd.DataFrame(data, columns=['New Filename'])
    xpos = pd.DataFrame(data, columns=['X microns/pixels'])
    ypos = pd.DataFrame(data, columns=['Y microns/pixels'])
    zpos = pd.DataFrame(data, columns=['Z microns/pixels'])
    visscore = pd.DataFrame(data, columns=['Pre op Vision'])
    return org_name, new_name, xpos, ypos, zpos, visscore


# write all 3D images features to an excel file
def _writing_file(worksheet, row, col, org_name, new_name, bs, bs2,
                  lp, dp, cl, avg_width, mzm, mznm, xpos, ypos, zpos,
                  visscore, nlm, di, ai, file_hash):
    rowHeaders = ['Original Filename', 'New Filename', 'X Microns/Pixels',
                  'Y Microns/Pixels', 'Z Microns/Pixels', 'Pre-op Vision',
                  'Contrast Level', 'Whiteness Level', 'Darkness Level',
                  'Average Pixel', 'Blurriness Level Test', 'Blurriness Level',
                  'Motion Est.', 'Normalised Motion Est.', 'Noise Level',
                  'Dominant Intensity', 'Average Intensity',
                  'Hash Number']
    rowValues = [org_name.iloc[row-1]['Original Filename'],
                 new_name.iloc[row-1]['New Filename'],
                 xpos.iloc[row-1]['X microns/pixels'],
                 ypos.iloc[row-1]['Y microns/pixels'],
                 zpos.iloc[row-1]['Z microns/pixels'],
                 visscore.iloc[row-1]['Pre op Vision'],
                 cl, lp, dp, avg_width, bs2, bs, mzm, mznm,
                 nlm, di, ai, file_hash]
    worksheet.write_row(0, col,  tuple(rowHeaders))
    worksheet.write_row(row, col, tuple(rowValues))
    return worksheet, row, col, org_name, new_name, bs, bs2, lp, dp, cl, avg_width, mzm, mznm, xpos, ypos, zpos, visscore, nlm, di, ai, file_hash


# get and print all 3D images features to an excel file
def _get_all_image_features_3d(folder):
    # list image files
    filenames = os.listdir(folder)
    # sort the image filenames
    filenames = sorted(filenames, key=lambda v: v.upper())
    # get main info inside excel file
    org_name, new_name, xpos, ypos, zpos, visscore = _get_main_info(file)
    # create the excel file
    workbook = xlsxwriter.Workbook('Report.xlsx')
    worksheet = workbook.add_worksheet()
    row = 1
    col = 0
    window_size = 15
    # loop
    for filename in filenames:
        print(filename)
        filename = os.path.join(folder, filename)
        # get info
        file_hash = _hash_file_name(filename)
        im, bs, bs2, ai, lp, dp, di, imq, avg_width, nl, nlm, \
            cl = _image_features_3d(filename)
        # get info
        mz, mzn, mzm, mznm = _get_optical_flow_z(im, window_size, tau=1e-2)
        # write the info to the worksheet
        worksheet, row, col, org_name, new_name, bs, bs2, lp, dp, cl, \
            avg_width, mzm, mznm, xpos, ypos, zpos, visscore, nlm, di, ai, \
                file_hash = _writing_file(worksheet, row, col, org_name,
                                          new_name, bs, bs2, lp, dp, cl,
                                          avg_width, mzm, mznm, xpos,
                                          ypos, zpos, visscore, nlm,
                                          di, ai, file_hash)
        row += 1
        # print what you have
        _print_image_features_3d(bs, bs2, ai, lp,
                                 dp, cl, di, nlm, mzm, mznm)
    workbook.close()
    return im, bs, bs2, ai, lp, dp, di, avg_width, nl, nlm, cl, mz, mzn, mzm, mznm, file_hash


# print image features
def _print_image_features_3d(bs, bs2, ai, lp, dp, cl, di, nlm, mzm, mznm):
    # plot
    # display(im)
    print('blurrness_score: ' + str(bs))
    print('blurrness_score_test: ', str(bs2))
    print('average_intensity: ' + str(ai))
    print('dark_light: lp=' + str(lp) + ', dp=' + str(dp))
    print('dominant_intensity: ' + str(di))
    print('noise_level: ' + str(nlm))
    print('contrast_level: ' + str(cl))
    print('max_optical_flow_z: mzm = ' + str(mzm) + ', mznm = ' + str(mznm))


# draw histogram
def _doing_histogram(file):
    all_data = pd.read_excel('Report.xlsx')
    names = ['Pre-op Vision', 'Contrast Level',
             'Whiteness Level', 'Darkness Level',
             'Average Pixel', 'Blurriness Level Test', 'Blurriness Level',
             'Motion Est.', 'Normalised Motion Est.', 'Noise Level',
             'Dominant Intensity', 'Average Intensity']
    dataset = pd.DataFrame(all_data, columns=names)
    nrows = 3
    ncols = int(len(names)/nrows)
    fig, ax2d = plt.subplots(nrows, ncols)
    ax = np.ravel(ax2d)
    for count, p in enumerate(ax):
        p.hist(dataset.iloc[:, count], bins=50, color='red', alpha=0.7,
               edgecolor='black', lw=1.5)
        p.set_xlabel(dataset.columns[count], fontsize=13)
        p.set_ylabel('Frequency', fontsize=13)
    plt.tight_layout()
    fig.set_size_inches(20, 15)
    fig.subplots_adjust(left=0.09, bottom=0.13, right=0.96,
                        top=0.87, wspace=0.47, hspace=0.71)
    plt.show()
    fig.savefig('full_figure.png', dpi=600)


if __name__ == '__main__':
    # load images
    folder = 'comparisons/images'
    file = 'comparisons/OCT - MH - Dataset.csv'
    # try getting features!
    folder, file = _files_control(folder, file)
    dups = _find_duplicate_file(folder)
    _print_duplicate_file(dups)
    imf = _get_all_image_features_3d(folder)
    _doing_histogram(file)
