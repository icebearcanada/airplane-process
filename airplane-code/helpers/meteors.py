# Calculates the closure quantities (angle and amplitude) for each second that has
# a valid xspectra in the Level 1 data. Plots minute-long RTIs and the corresponding
# closure quantities.
#
# Author: Brian Pitzel
# Date created:  18-07-23
# Date modified: 18-07-23


import h5py
import numpy as np
import cupy as cp
#import digital_rf
import helpers.utils as utils
import datetime
import matplotlib.pyplot as plt
import scipy
from skimage import morphology

def detect_meteors(power, threshold1, threshold2, threshold3, res_incr=1):
    """
    Find the meteors in a given timeframe based on power and threshold values.

    Parameters:
    -----------
    power : numpy.ndarray, dtype(float), shape (n_time, n_rng). Nominally (600, 2000)
            An array of the power in each range gate for a given timeframe

    threshold1 : int or float
            The initial threshold value. Any power values above this value will be considered as meteors

    threshold2 : int or float
            The second threshold value. Any power values above this value, IN THE AREA OF THE METEORS FOUND BY
            THRESHOLD1, will also be considered meteors.

    threshold3 : int or float
            The final threshold value. Again, power values above this value will be considered meteors IF they
            are also in the area of the meteors found by the first two thresholds

    Returns:
    --------
    detection_binary_image : 

    meteors : numpy.ndarray, dtype(integer), shape (n_time, n_rng). Nominally (600, 2000)
            An array of integers  representing different meteors in the range-time space.
            Indices with value 0 are considered to have no meteor. Indices with a value
            greater than 0 are considered to have a meteor, shared with other indices with
            the same value.

    meteor_cnt : int
            The number of meteor labels used in meteors.
            The number of (specular, non-specular, ambiguous) meteors in the
            input array
    """
    image_list = []

    # start playing with the binary image
    detection_binary_image_0 = np.where(power < threshold1, 0, 1)
    image_list.append(detection_binary_image_0)

    # cushion
    time_size = (2*1)*res_incr + 1
    range_size = 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image_0, structure=structure)
    image_list.append(detection_binary_image)

    # search (threshold2)
    time_size = (2*25)*res_incr + 1
    range_size = 2*1 + 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image, structure=structure)
    image_list.append(detection_binary_image)

    detection_binary_image_1 = np.where((detection_binary_image) & (power >= threshold2), 1, 0)
    image_list.append(detection_binary_image_1 - detection_binary_image_0)

    # cushion
    time_size = (2*1)*res_incr + 1
    range_size = 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image_1, structure=structure)
    image_list.append(detection_binary_image)

    # remove small objects
    detection_binary_image = morphology.remove_small_objects(detection_binary_image, min_size=4, connectivity=1)
    image_list.append(detection_binary_image)

    # search (threshold2, round 2)
    time_size = (2*25)*res_incr + 1
    range_size = 2*1 + 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image, structure=structure)
    image_list.append(detection_binary_image)

    detection_binary_image_2 = np.where((detection_binary_image) & (power >= threshold2), 1, 0)
    image_list.append(detection_binary_image_2 - detection_binary_image_1)

    # cushion
    time_size = (2*1)*res_incr + 1
    range_size = 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image_2, structure=structure)
    image_list.append(detection_binary_image)

    # remove small objects
    detection_binary_image = morphology.remove_small_objects(detection_binary_image, min_size=4, connectivity=1)
    image_list.append(detection_binary_image)

    # search (threshold3)
    time_size = (2*4)*res_incr + 1
    range_size = 2*1 + 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image, structure=structure)
    image_list.append(detection_binary_image)

    detection_binary_image_3 = np.where((detection_binary_image) & (power >= threshold3), 1, 0)
    image_list.append(detection_binary_image_3 - detection_binary_image_2)

    # cushion
    time_size = (2*2)*res_incr + 1
    range_size = 2*1 + 1
    structure = np.ones((time_size, time_size))
    structure[: (time_size - range_size) // 2, :] = 0
    structure[-(time_size - range_size) // 2 :, :] = 0
    detection_binary_image = scipy.ndimage.binary_dilation(detection_binary_image_3, structure=structure)
    image_list.append(detection_binary_image)

    # remove small objects
    detection_binary_image = morphology.remove_small_objects(detection_binary_image, min_size=4, connectivity=1)
    image_list.append(detection_binary_image)

    # label the detected meteors
    meteors, meteor_cnt = scipy.ndimage.label(detection_binary_image)
    print(meteor_cnt, "meteors detected")

    return detection_binary_image, meteors, meteor_cnt, image_list

def classify_meteors(meteors, meteor_cnt, timestamp, res_incr=1):
    """
    Classify labelled meteor occurrences based on time and range characteristics.

    Parameters:
    -----------
    meteors : numpy.ndarray, dtype(integer), shape (n_time, n_rng). Nominally (600, 2000)
            An array of integers representing different meteors in the range-time space.
            Indices with value 0 are considered to have no meteor. Indices with a value
            greater than 0 are considered to have a meteor, shared with other indices with
            the same value.

    meteor_cnt : int
            The number of meteor labels used in meteors.

    timestamp : string type
            The timestamp to place as the first entry in the output tuples

    Returns:
    --------
    counts : tuple(int, int, int)
            The number of (specular, non-specular, ambiguous) meteors in the
            input array

    labels : list(tuple)
            Contains tuples of (timestamp, label_number, meteor_type, timespan, rangespan, start_tidx, start_ridx) for each label
            number for all meteors in the input array meteors
    """
    spec_cnt = 0
    nspec_cnt = 0
    ambig_cnt = 0
    labels = []

    # manually label the meteors
    #classifications = manual_classify(meteors, meteor_cnt)

    # bypass the manual classification
    classifications = [-1]*meteor_cnt

    # iterate through the labels
    for i in range(1, meteor_cnt+1):
        idxs = np.nonzero(meteors == i)
        tidxs = idxs[1]
        ridxs = idxs[0]

        timespan = np.abs(np.max(tidxs) - np.min(tidxs)) + 1
        rangespan = np.abs(np.max(ridxs) - np.min(ridxs)) + 1
        start_tidx = np.min(tidxs)
        start_ridx = np.min(ridxs)

        # check that it's not an airplane or direct feed (we'll say 300 km)
        if np.any(ridxs < (300 // 1.5)):
            labels.append((timestamp, i, 'invalid', classifications[i-1], timespan, rangespan, start_tidx, start_ridx))
            continue 
        # check the timespan of the meteor. If more than 1s, assign to non-spec. If less than 0.3s, assign to spec

        elif timespan > (10 + 2*2)*res_incr: # 10 x 0.1s samples = 1s, plus 2*2 to account for the binary dilation
            nspec_cnt += 1
            labels.append((timestamp, i, 'nspec', classifications[i-1], timespan, rangespan, start_tidx, start_ridx))
            continue
        elif timespan < (4 + 1*2)*res_incr:
            spec_cnt += 1
            labels.append((timestamp, i, 'spec', classifications[i-1], timespan, rangespan, start_tidx, start_ridx))
            continue

        # check the rangespan of the meteor. If more than 4 gates, assign to non-spec.
        elif rangespan > 4:
            nspec_cnt += 1
            labels.append((timestamp, i, 'nspec', classifications[i-1], timespan, rangespan, start_tidx, start_ridx))
            continue

        # anything left is ambiguous (but probably specular)
        ambig_cnt += 1
        labels.append((timestamp, i, 'ambig', classifications[i-1], timespan, rangespan, start_tidx, start_ridx))
        continue

    counts = (spec_cnt, nspec_cnt, ambig_cnt)
    return labels, counts

def manual_classify(meteors, meteor_cnt):
    print("Get ready to classify")
    print("Classify as:")
    print("0 = Airplane/Direct feed")
    print("1 = Non-specular meteor")
    print("2 = Specular underdense meteor")
    print("3 = Specular overdense meteor")
    print("4 = Ionospheric echo")

    classifications = []

    for i in range(1, meteor_cnt+1):
        idxs = np.nonzero(meteors == i)
        tidxs = idxs[1]
        ridxs = idxs[0]

        start_tidx = np.min(tidxs)
        start_ridx = np.min(ridxs)

        classifications.append(input(f"Enter classification for meteor {i} : (tidx, ridx) = ({start_tidx}, {start_ridx})\n"))
    return classifications

def svm_classification():
    return 0




































































































































































































































































































