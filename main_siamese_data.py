"""
@Created : 17/05/2021
@Edited  : 08/06/2022
@Author  : Alex Poppe
@File    : peak_correlations.py
@Software: Pycharm
@Description:
Uses Tracks generated through the scanalyser() function in peak_finder.py to synthesise Track pairs (called snippets),
which are used to train a Siamese-CNN to learn peak correlations. Data augmentation is applied to the snippets via an
image warping procedure in order to produce more training samples.

This image warping works as follows:
    1. Load in the Tracks and datasets to produce and preprocess the snippets
    2. Create a polynomial estimate of the 'centroid' data, then use the extrema points to augment a new polynomial
    3. From the polynomial data and border coordinates create two sets of coordinates through a Delaunay triangulation
    4. Perform the image warping using a piecewise affine transformation from the image input to the warped image using
        the tessellated coordinates of the original (source) and augmented (destination) coordinate sets

    Note: Step 1 is found in track_synthesis(), whereas the other steps are found in generate_pairs()
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress unnecessary TF loading messages

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import glob
import cv2
from sklearn.model_selection import train_test_split, KFold
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from main_cae_data import select_scan
from itertools import combinations
from pathlib import Path
from utils import append_value, hyperparams_setup


# <editor-fold desc="---=== [+] Configure GPU ===---">
def enable_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


# </editor-fold>
enable_gpu()


# <editor-fold desc="---=== [+] Delaunay Triangulation Functions ===---">
def get_triangulation_indices(points):
    """ Get triplet indices for every triangle

    Args:
        points: Array, (n, 2) array of coordinates ordered in (row, col) pairs
    """
    # Bounding rectangle
    rect = (*points.min(axis=0), *points.max(axis=0))

    # Make the bounding box larger (For Some Reason having points at 0 causes cv2.Subdiv2D to have problems)
    rect = list(rect)
    rect[2], rect[3] = rect[2] + 1, rect[3] + 1
    rect = tuple(rect)

    # Triangulate all points
    subdiv = cv2.Subdiv2D(rect)  # create a object representing the bounding rectangle
    subdiv.insert(list(points))  # adds the list of coordinates inside of the rectangle

    # Iterate over all triangles
    for x1, y1, x2, y2, x3, y3 in subdiv.getTriangleList():  # returns x,y coords of the three vertices of the triangle
        # Get index of all points
        yield [(points == point).all(axis=1).nonzero()[0][0] for point in [(x1, y1), (x2, y2), (x3, y3)]]

    return


def crop_to_triangle(img, triangle):
    """ Crop image to triangle

    Args:
        img: Array, The input track image
        triangle: Array, (3, 2) array of coordinates for the vertices of the current Delaunay triangle
    """
    # Get bounding rectangle
    rect = cv2.boundingRect(triangle)  # returns the (x, y, w, h) coords, i.e. lowest x and y values

    # Crop image to bounding box
    # (Note: dst_img is changed in-place as the crops are references to locations within the main image and are updated
    # accordingly in the following code)
    img_cropped = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

    # Move triangle to coordinates in cropped image
    triangle_cropped = [(point[0] - rect[0], point[1] - rect[1]) for point in triangle]
    return triangle_cropped, img_cropped


def transform(src_img, src_points, dst_img, dst_points):
    """ Transforms source image (input) to target image (output), overwriting the target image.

    Args:
        src_img: Array, The input track image
        src_points: Array, (n, 2) coordinate pairs in the source image with which to form Delaunay triangles
        dst_img: Array, The output track image
        dst_points: Array, (n, 2) coordinate pairs in the destination image with which to form Delaunay triangles
    """
    for indices in get_triangulation_indices(src_points):  # this is where the yield function is useful
        # Get triangles from indices
        src_triangle = src_points[indices]  # the fact that these triangular coordinate indices are used for both the...
        dst_triangle = dst_points[indices]  # ...src and dst coords implies that their order is crucial for this to work

        # Crop to triangle, to make calculations more efficient
        src_triangle_cropped, src_img_cropped = crop_to_triangle(src_img, src_triangle)
        dst_triangle_cropped, dst_img_cropped = crop_to_triangle(dst_img, dst_triangle)

        # Calculate transform to warp from old image to new
        tform = cv2.getAffineTransform(np.float32(src_triangle_cropped), np.float32(dst_triangle_cropped))

        # Warp image
        # (cv2.BORDER_REFLECT_101 == 'reflect', cv2.BORDER_REFLECT == 'symmetric')
        # (e.g. edcb|abcde|dcba, & edcba|abcde|edcba)
        # (None == dst = None)
        dst_img_warped = cv2.warpAffine(src_img_cropped, tform, (dst_img_cropped.shape[1], dst_img_cropped.shape[0]),
                                        None, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        # Create mask for the triangle we want to transform
        mask = np.zeros(dst_img_cropped.shape, dtype=np.uint8)
        # (1.0, 1.0, 1.0) is a 'polygon colour'
        # 16 is antialiasing
        # 0: "number of fractional bits in the vertex coordinates." as in 0 stops a fractal pattern from occurring (wtf)
        # tl;dr puts 1s where triangle is, and 0s where it is not
        cv2.fillConvexPoly(mask, np.int32(dst_triangle_cropped), (1.0, 1.0, 1.0), 16, 0)

        # Delete all existing pixels at given mask
        dst_img_cropped *= 1 - mask  # i.e. remove the triangle from the cropped image
        # Add new pixels to masked area
        dst_img_cropped += dst_img_warped * mask  # i.e. place the warped input image into the triangle

    return


# </editor-fold>


# <editor-fold desc="---=== [+] Pre-Training Dataset Creation & Loading ===---">
def create_snippets(params):
    """ Converts Tracks into snippets and saves four datasets to file: whole database, train, valid, and test

    Args:
        params: Dict, The hyperparameter dictionary
    """
    # Define constants
    min_duration = 100  # minimum duration of a Track to be used [Maybe try min_duration=50 & stride=10 for more data]
    width = 25  # width that Tracks are interpolated to (requirement for input into the Siamese-CNN)
    x_extend = 5  # horizontal padding on each 'Track image' (attempt to ensure that whole-peak is present in image)
    y_extend = 0  # vertical padding
    stride = 5  # stride between windows for Track snippets (e.g. duration 105 split into two: 1st @ 0-99, 2nd @ 5-104)

    # Instantiate lists of snippets, as well as their respective coords and label information
    snippets = []
    coords = []
    labels = []

    # Cycle through all Track filepaths, find Tracks with a duration >= min_duration, and extract bounding boxes
    filepaths = glob.glob(f'./data/tracks/{params["c_ver"]}/*tracker.pkl')
    if len(filepaths) == 0:
        print(f'ERROR: No trackers found in directory:\n./data/tracks/{params["c_ver"]}/*tracker.pkl')
        exit()

    for filepath in filepaths:
        with open(filepath, 'rb') as r:
            tracker = pickle.load(r)
            # Retrieve the scan associated with the current set of Tracks
            dataset = select_scan(particles=[f'Particle_{tracker.particle_num}'])

            for i, track in enumerate(tracker.tracks):
                # (NOTE: len() would contain bifurcations, therefore max and min must be used to calculate duration)
                if np.max(track.trace_time) - np.min(track.trace_time) >= min_duration:
                    for data, _ in dataset.as_numpy_iterator():
                        # Define the image edge bounds, using min/max wrappers to avoid negative indices
                        # (NOTE: left/right edges are padded with 5 px to attempt to include whole-peaks)
                        left = np.max((np.min(track.trace_bounds) - x_extend, 0))
                        right = np.min((np.max(track.trace_bounds) + x_extend, data.shape[1] - 1))
                        top = np.max((np.min(track.trace_time) - y_extend, 0))
                        bottom = np.min((np.max(track.trace_time) + y_extend, data.shape[0] - 1))
                        image = data[top:bottom, left:right].squeeze()

                        # Store the original image width for later use in resizing the sub-centroid arrays
                        old_width = image.shape[1]

                        # Resize the image width using linear interpolation
                        image = cv2.resize(image, dsize=(width, image.shape[0]), interpolation=cv2.INTER_LINEAR)

                        # Split the track image into snippets using a striding window, each with duration min_duration
                        num_sub_images = int(np.floor((image.shape[0] - min_duration) / stride) + 1)
                        for j in range(num_sub_images):
                            # ---=== Step 1. Create Sub-Tracks & Preprocess ===---
                            snippet = image[stride * j:stride * j + min_duration]

                            # Add the sub-image to its list
                            snippets.append(snippet)

                            # Add the label information to its list
                            label = ''.join(
                                [f"Particle{tracker.particle_num}", f"_groupID{track.group_id}", f"_sub{j}"])
                            labels.append(label)

                            # Retrieve the track centroids for the current sub-image
                            idxs = np.where(np.logical_and(np.array(track.trace_time) >= top + stride * j, np.array(
                                track.trace_time) < top + stride * j + min_duration))[0]
                            sub_centroids = np.array(track.trace)[idxs]

                            # Scale the centroid values between [0, right]. However, the centroid values themselves
                            # are strictly between [x_extend, right - x_extend] because of the x_extend values
                            sub_centroids = (sub_centroids - left) / (right - left) * snippet.shape[1]

                            # Rescale the centroid values to the resized track image, this is done by adding in min/max
                            # points based on the track image before resizing, then removing these points afterwards
                            sub_centroids = np.concatenate((sub_centroids, np.array([0, old_width])), axis=0)
                            sub_centroids = (sub_centroids - np.min(sub_centroids)) / (
                                    np.max(sub_centroids) - np.min(sub_centroids)) * width
                            sub_centroids = sub_centroids[:-2]  # remove the temp points

                            # Retrieve the time steps within this range (can contain duplicates due to bifurcation, or
                            # missing time steps due to the zipper function connecting tracks over a gap(s))
                            sub_times = np.array(track.trace_time)[idxs]

                            # Find the indices of the duplicates that are to be removed and replaced with mean values,
                            # but in reverse order as this allows us to remove/replace without breaking future indices
                            u, c = np.unique(sub_times, return_counts=True)
                            u = u[c != 1]  # retrieve the duplicate time steps
                            remove_mean = np.array([np.argwhere(sub_times == ui) for ui in u])[::-1]  # mean indices
                            new_mean = np.array([np.mean(sub_centroids[means]) for means in remove_mean])  # mean values
                            for d, m in enumerate(remove_mean):
                                # Replace the duplicate values with their mean in the same position as the first value
                                sub_centroids = np.insert(np.delete(sub_centroids, m), m[0], new_mean[d])
                                sub_times = np.delete(sub_times, m[1:])

                            # Use linear interpolation to fill in any missing time steps by defining the 'proper range'
                            new_times = np.arange(top + stride * j, top + stride * j + min_duration, 1)
                            sub_centroids = interp1d(sub_times, sub_centroids, kind='linear', fill_value='extrapolate')(
                                new_times)

                            # Define the new timescale centred on zero, this is to make the polynomial fitting more
                            # accurate, as if it started at zero, then there would be greater divergent effects for
                            # later points
                            sub_times = np.arange(-min_duration // 2, min_duration // 2)

                            # Form a centroid coordinates array and add them to list
                            coords.append(np.vstack((sub_centroids, sub_times)).T)

    # Convert the data lists to arrays
    snippets = np.array(snippets)
    coords = np.array(coords)
    labels = np.array(labels)

    # Temporarily concatenate the image and coordinate arrays, to allow for the three arrays to be split
    combined_arrays = np.concatenate((snippets, coords), axis=-1)

    # Split the database into an 90/5/5 split
    # - First split database into train/inference
    train_combined, infer_combined, train_labels, infer_labels = train_test_split(combined_arrays, labels,
                                                                                  test_size=0.1)
    # - Then split inference into valid/test
    valid_combined, test_combined, valid_labels, test_labels = train_test_split(infer_combined, infer_labels,
                                                                                test_size=0.5)

    # Split the combined arrays back into the image, coordinate pairs
    train_images = train_combined[:, :, :-2].astype('float32')  # default data type of tensorflow parameters is float32
    train_coords = train_combined[:, :, -2:].astype('float32')
    valid_images = valid_combined[:, :, :-2].astype('float32')
    valid_coords = valid_combined[:, :, -2:].astype('float32')
    test_images = test_combined[:, :, :-2].astype('float32')
    test_coords = test_combined[:, :, -2:].astype('float32')

    # Save the entire database and individual datasets to file
    path = f'./data/snippets/{params["c_ver"]}/raw'
    Path(path).mkdir(parents=True, exist_ok=True)

    # - Database
    np.save(f'{path}/database_images.npy', snippets)
    np.save(f'{path}/database_coords.npy', coords)
    np.save(f'{path}/database_labels.npy', labels)

    # - Train
    np.save(f'{path}/train_images.npy', train_images)
    np.save(f'{path}/train_coords.npy', train_coords)
    np.save(f'{path}/train_labels.npy', train_labels)

    # - Valid
    np.save(f'{path}/valid_images.npy', valid_images)
    np.save(f'{path}/valid_coords.npy', valid_coords)
    np.save(f'{path}/valid_labels.npy', valid_labels)

    # - Test
    np.save(f'{path}/test_images.npy', test_images)
    np.save(f'{path}/test_coords.npy', test_coords)
    np.save(f'{path}/test_labels.npy', test_labels)

    return


def synthesise_tracks(image, coord, batch_size):
    """ Augments snippets into labelled Track pairs

    Args:
        image: Array, The batch of Track snippets
        coord: Array, The batch of [wavenumber, time] positions (in pixels) for each centroid
        batch_size: Int, The batch size
    """
    # Define parameters
    deg = 5  # number of orders for fitting a polynomial to the centroids
    noise_per = 0.15  # amount of noise added to each variable polynomial position as a percentage of its magnitude

    # Instantiate a list to contain the warped image pairs for the current batch
    warped_images = []

    # Loop through one sample at a time to warp the snippet into Track pairs (not labelled yet)
    for img, crd in zip(image, coord):
        # Split the coord array into its centroid and time step counterparts for each sample in the batch
        centroids = crd[:, 0]
        times = crd[:, 1]

        # ---=== Step 2. Create Polynomials ===---
        # Fit polynomial coefficients to the centroids, then sample that polynomial across the time steps
        p = np.poly1d(np.polyfit(times, centroids, deg=deg))(times)

        # To augment, first find indices of extrema, ...
        uppers = argrelextrema(p, np.greater)[0]  # peaks
        lowers = argrelextrema(p, np.less)[0]  # troughs
        # ...then find indices of 'cross-over' points (i.e. changes in sign of acceleration), ...
        p_diff = np.diff(p)
        pos_cross = argrelextrema(p_diff, np.greater)[0] + 1  # +1 as p_diff.size = p.size - 1
        neg_cross = argrelextrema(p_diff, np.less)[0] + 1
        # ...then retrieve indices of end-points to 'pin' the augmentation to sensible bounds
        end_idxs = np.array([0, p.shape[0] - 1])

        # Assemble two arrays: the time steps for the above data points, and their indices
        saddle_idxs = np.concatenate((uppers, lowers, pos_cross, neg_cross, end_idxs), axis=-1)
        saddle_times = times[saddle_idxs]

        # Sort both arrays in time step order
        order = np.argsort(saddle_times)
        saddle_idxs = saddle_idxs[order]
        saddle_times = saddle_times[order]

        # Create the centroids array by retrieving their values at each previously selected point
        saddle_centroids = p[saddle_idxs]

        # Create two instances of it, as we require pairs of separately warped images
        paired_centroids = np.tile(saddle_centroids, 2).reshape((2, -1))

        # Add scaled, uniform noise to each image
        # (The end-points remain fixed, otherwise triangular folding could occur between border points and these)
        paired_centroids[:, 1:-1] *= (1 + np.random.randn(2, paired_centroids.shape[1] - 2) * noise_per)

        # Limit the data points to the edges of the image, in case the noise took them over it
        paired_centroids = np.min((paired_centroids, np.ones(paired_centroids.shape) * img.shape[1] - 1), axis=0)
        # (NOTE: There is a small possibility that data points could pass through a triangular edge of the tesselation,
        # which would produce a fold in the warping process. The solution would be to find the vertices that generate
        # that edge of the triangle, find its intercept at the specific row causing the fold, and then cap that point
        # to the value of the intercept - 1)

        # ---=== Step 3. Delaunay Triangulation ===---
        # Create an array of border point coordinates for both snippets. Start with the row coordinates, creating 8
        # evenly-spaced points for each vertical image edge
        border_rows = np.linspace(0, img.shape[0] - 1, 8).astype('int')
        # Create the x-axis coordinates to pair with the y-axis coordinates, one set for each edge
        left_right = np.ones(2 * border_rows.size) * (img.shape[1] - 1)  # right edge
        left_right[:border_rows.size] = 0  # left edge
        # Form the coordinate pairs
        border_rows = np.vstack((left_right, np.tile(border_rows, 2))).T

        # Now create the column coordinates, creating 4 evenly-spaced points for each horizontal image edge.
        border_cols = np.linspace(0, img.shape[1] - 1, 4).astype('int')[1:-1]  # (first and last points are redundant)
        # Create the y-axis coordinates to pair with the x-axis coordinates, one set for each edge
        top_bottom = np.ones(2 * border_cols.size) * (img.shape[0] - 1)  # bottom edge
        top_bottom[:border_cols.size] = 0  # top edge
        # Form the coordinate pairs
        border_cols = np.vstack((np.tile(border_cols, 2), top_bottom)).T

        # Assemble the polynomial coordinates from the original and augmented polynomials
        saddle_times += times.shape[0] // 2  # shift the time range back to 0-min_duration
        orig_poly = np.vstack((p[saddle_idxs], saddle_times)).T
        aug_poly_0 = np.stack((paired_centroids[0], saddle_times)).T
        aug_poly_1 = np.stack((paired_centroids[1], saddle_times)).T

        # Assemble the full coordinate arrays for both the source and two destination coordinates
        source = np.concatenate((border_rows, border_cols, orig_poly), axis=0).astype('int')
        destination_0 = np.concatenate((border_rows, border_cols, aug_poly_0), axis=0).astype('int')
        destination_1 = np.concatenate((border_rows, border_cols, aug_poly_1), axis=0).astype('int')

        # ---=== Step 4. Image Warping ===---
        # Apply transformation to create the warped images (these augment the warped_image_0/1 in-place)
        warped_image_0 = np.copy(img)
        warped_image_1 = np.copy(img)
        transform(img, source, warped_image_0, destination_0)
        transform(img, source, warped_image_1, destination_1)

        # Append Track pair to current batch
        warped_images.append(np.stack((warped_image_0, warped_image_1)))

    # Convert the batch list to an array
    warped_images = np.array(warped_images)

    # Linearly normalise the warped images individually
    sample_max = np.max(warped_images, axis=(2, 3), keepdims=True)
    sample_min = np.min(warped_images, axis=(2, 3), keepdims=True)
    warped_images = (warped_images - sample_min) / (sample_max - sample_min)

    # Randomly horizontally flip the second warped image in each pair on a coin flip
    flip_mask = np.random.randint(0, 2, batch_size).astype('bool')  # convert to boolean mask for indexing
    warped_images[flip_mask, 1] = np.flip(warped_images[flip_mask, 1], axis=-1)

    # Randomly horizontally flip both warped images in each pair on a coin flip
    # (NOTE: This process, combined with the previous flip, produces 1 of 4 perspectives with equal probability)
    # (NOTE: As this second process flips both Tracks in a pair, the correlation does not change)
    flip_mask_temp = np.random.randint(0, 2, batch_size).astype('bool')  # convert to boolean mask for indexing
    warped_images[flip_mask_temp, :] = np.flip(warped_images[flip_mask_temp, :], axis=-1)

    # Reverse values of flip mask to make 1 represents a positive correlation, and 0 a negative (convention)
    # Also, expand the dimensions of both the flip mask and the warped images for network input compatibility
    flip_mask = np.expand_dims((1 - flip_mask), axis=-1)
    warped_images = np.expand_dims(warped_images, axis=-1)

    return warped_images, flip_mask


def generate_pairs(params):
    """ Yield batches of the labelled Track pairs for the training dataset

    Args:
        params: Dict, The hyperparameter dictionary
    """
    # Load in the training dataset
    images = np.load(f'./data/snippets/{params["c_ver"]}/raw/train_images.npy')
    coords = np.load(f'./data/snippets/{params["c_ver"]}/raw/train_coords.npy')

    # Create a Tensorflow dataset of image-coordinate pairs, then shuffle and batch
    temp_dataset = tf.data.Dataset.from_tensor_slices((images, coords)).shuffle(images.shape[0]).batch(
        params['s_batch_size'], drop_remainder=True)

    # Cycle through each batch
    for image, coord in temp_dataset.as_numpy_iterator():
        # Augment the snippets to form labelled Track pairs
        data, labels = synthesise_tracks(image, coord, batch_size=params['s_batch_size'])

        # Yield data/labels to train the current step of the Siamese-CNN
        yield data, labels


def produce_inference(params):
    """ Generate and save fixed labelled Track pairs for the validation and testing datasets

    Args:
        params: Dict, The hyperparameter dictionary
    """
    # Load in the datasets
    valid_images = np.load(f'./data/snippets/{params["c_ver"]}/raw/valid_images.npy')
    valid_coords = np.load(f'./data/snippets/{params["c_ver"]}/raw/valid_coords.npy')
    test_images = np.load(f'./data/snippets/{params["c_ver"]}/raw/test_images.npy')
    test_coords = np.load(f'./data/snippets/{params["c_ver"]}/raw/test_coords.npy')

    # Augment the snippets to form labelled Track pairs
    valid_data, valid_labels = synthesise_tracks(valid_images, valid_coords, batch_size=valid_images.shape[0])
    test_data, test_labels = synthesise_tracks(test_images, test_coords, batch_size=test_images.shape[0])

    # Save the fixed Track pairs/labels to file
    path = f'./data/snippets/{params["c_ver"]}/synth'
    Path(path).mkdir(parents=True, exist_ok=True)

    np.save(f'{path}/valid_images_fixed.npy', valid_data)
    np.save(f'{path}/valid_labels_fixed.npy', valid_labels)
    np.save(f'{path}/test_images_fixed.npy', test_data)
    np.save(f'{path}/test_labels_fixed.npy', test_labels)

    return


def load_inference(params, dataset='valid'):
    """ Load in the chosen fixed inference dataset (i.e. valid or test) and yield as output one batch at a time

    Args:
        params: Dict, The hyperparameter dictionary
        dataset: Str, The chosen dataset. Options = 'valid' or 'test'

    Yields:
        image: Array, Data for the current batch
        label: Array, Labels for the current batch
    """
    # Exit if an incorrect dataset is specified by the user
    if dataset not in ['valid', 'test']:
        print('ERROR: Incorrect dataset chosen. Options = \'valid\', \'test\'\nExiting...')
        exit()

    # Load in the chosen dataset and its associated labels
    images = np.load(f'./data/snippets/{params["c_ver"]}/synth/{dataset}_images_fixed.npy')
    labels = np.load(f'./data/snippets/{params["c_ver"]}/synth/{dataset}_labels_fixed.npy')

    # Form a Tensorflow dataset to make batching easier
    infer_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(params['s_batch_size'],
                                                                               drop_remainder=True)

    return infer_dataset


# </editor-fold>


# <editor-fold desc="---=== [+] Fine-Tuning Dataset Creation & Loading ===---">
def create_snippets_ft(params):
    """ This function is very similar to create_snippets(), except that it saves snippets and unsupervised labels based
    on the Groups that those Tracks originated from. There is also a label to specify which part of the track each
    snippet came from (e.g. The first window of a Group containing three Tracks will have labels: ID=0, sub=0, and then
    each snippet is differentiated by its mean centroid [in pixels]).

    NOTE: This function is used to create the data for the fine-tuning process, however it does NOT assign correlation
    labels to each pair. Those labels must be MANUALLY assigned [if this wasn't the case then the Siamese-CNN would be
    redundant]; the results of this are found in fine_tuning_labels()

    Args:
        params: Dict, The hyperparameter dictionary
    """
    # Define constants
    min_duration = 100  # minimum duration of a Track to be used [Maybe try min_duration=50 & stride=10 for more data]
    width = 25  # width that Tracks are interpolated to (requirement for input into the Siamese-CNN)
    x_extend = 5  # horizontal padding on each 'Track image' (attempt to ensure that whole-peak is present in image)
    y_extend = 0  # vertical padding
    stride = 5  # stride between windows for Track snippets (e.g. duration 105 split into two: 1st @ 0-99, 2nd @ 5-104)

    # Instantiate label information list
    labels = []

    # Instantiate counters to enforce uniqueness in group IDs throughout different particles
    # (e.g. if Particle_0000 has IDs 0 & 1, then Particle_0001 will start at ID 2 instead of 0, and so on)
    # (Note: The unique IDs are only needed for the models. The original Group IDs and the particle numbers are already
    # sufficient for us to identify the origin of a particular track)
    total_max = -1  # max throughout all particles
    current_max = -1  # max for current particle

    # Cycle through all Track filepaths, find tracks with a duration >= min_duration
    filepaths = glob.glob(f'./data/tracks/{params["c_ver"]}/*tracker.pkl')
    if len(filepaths) == 0:
        print(f'ERROR: No trackers found in directory:\n./data/tracks/{params["c_ver"]}/*tracker.pkl')
        exit()

    for i, filepath in enumerate(filepaths):
        # Update the global max ID counter
        # (Note: +1 because if a particle only has one ID (ID 0), it still needs incrementing. Also, if this is the
        # first particle, then current_max + 1 will be zero, and it will not affect the process)
        total_max += current_max + 1

        # Reset the max counter for the current particle
        current_max = -1

        with open(filepath, 'rb') as r:
            tracker = pickle.load(r)  # obtain tracker object
            # Retrieve the scan associated with the current set of Tracks to find the scan dimensions
            dataset = select_scan(particles=[f'Particle_{tracker.particle_num}'])
            for data, _ in dataset.as_numpy_iterator():
                break

            for track in tracker.tracks:
                # Update the max ID counter for this particle if a larger ID is found (even if it is unused)
                if track.group_id > current_max:
                    current_max = track.group_id

                # (Note: len() contains bifurcations, therefore max and min must be used to calculate duration)
                if np.max(track.trace_time) - np.min(track.trace_time) >= min_duration:
                    # Calculate the global Group ID
                    if i == 0:  # treat the first particle differently (as total_max=-1, which we want to avoid)
                        global_id = track.group_id
                    else:
                        global_id = track.group_id + total_max

                    # Calculate the mean wavenumber location of the current track
                    mean_trace = np.mean(track.trace).astype('int')

                    # Define the image edge bounds, using a min/max wrapper to avoid negative indices
                    left = np.max((np.min(track.trace_bounds) - x_extend, 0))
                    right = np.min((np.max(track.trace_bounds) + x_extend, data.shape[1] - 1))
                    top = np.max((np.min(track.trace_time) - y_extend, 0))
                    bottom = np.min((np.max(track.trace_time) + y_extend, data.shape[0] - 1))

                    # Calculate the number of sub images that will be produced
                    num_sub_images = int(np.floor((bottom - top - min_duration) / stride) + 1)
                    for j in range(num_sub_images):
                        # Add the label information to its list
                        # (global ID, snippet number, mean wavenumber, start time step, particle number, left, right])
                        labels.append(
                            [global_id, j, mean_trace, track.trace_time[stride * j], tracker.particle_num, left, right])

    # Convert to array
    labels = np.array(labels)

    # <editor-fold desc="---=== [+] Form Lists of Multi-Sub-Track Groups ===---">
    # Instantiate a dictionary for the new test dataset
    test_idxs = {}
    for idx1, label1 in enumerate(labels[:-1]):
        for idx2m, label2 in enumerate(labels[idx1 + 1:]):
            idx2 = idx2m + idx1 + 1
            if label1[0] == label2[0] and label1[1] == label2[1]:  # i.e. if snippets belong to same Group at same time
                append_value(test_idxs, idx1, idx2)

    # Organise the dictionary into lists containing each set of to-be-merged indices
    # (e.g. if test_idxs={1: [2, 3], 3: [4], 5: [6]}, then merge_idxs=[[1, 2, 3, 4], [5, 6]]).
    pre_merge_idxs = []
    for merge in test_idxs.items():
        pre_merge_idxs.append([int(merge[0])] + merge[1])

    # Instantiate list of merge assignments
    merge_idxs = []

    # Repeat merging process until no more merges occur (in case graph-connectivity process skips a merge)
    num_matches = 1
    num_attempts = 0  # used to stop the assignments process from going on forever

    # Instantiate a boolean to treat the first assignment attempt differently
    first_attempt = True
    while num_matches != 0:
        if num_attempts >= 1000:  # stop trying if it is taking forever...
            break
        num_matches = 0

        # Set the new merge_list as the partially-assigned merge_list, then reinstantiate the merge_list
        if not first_attempt:
            # Convert the event_list into a list of lists, rather than a list of sets
            pre_merge_idxs = [list(merge) for merge in merge_idxs]
            merge_idxs = []

        # tl;dr cycle through sublists and search for any intersections with other sublists. If so, merge
        # them together and remove those sublists from 'searching' list. Any unmerged sublists are searched
        # until all sublists have either been merged or searched (i.e. pre_merge_list slowly empties)
        while len(pre_merge_idxs) > 0:
            first, *rest = pre_merge_idxs
            first = set(first)

            temp = -1
            if len(first) > temp:
                temp = len(first)  # (this is being used, ignore the statement)
                next_rest = []
                for r in rest:
                    if len(first.intersection(set(r))) > 0:
                        num_matches += 1
                        first |= set(r)  # e.g. x = set([1, 2]), y = set([2, 3]) --> x |= y --> x = {1, 2, 3}
                    else:
                        next_rest.append(r)  # append the sublist to the list of future comparisons
                rest = next_rest  # this essentially removes all merged lists this iteration
            merge_idxs.append(first)
            pre_merge_idxs = rest

        first_attempt = False
        num_attempts += 1

    # Convert merge_idxs from a list of sets to a list of lists
    merge_idxs = [list(merge) for merge in merge_idxs]
    # </editor-fold>

    # Instantiate a labels dictionary for the multi-Track Groups
    test_labels = {}

    # Cycle through the merge list, creating a new dictionary entry with a new unique ID as the key for each entry,
    # which contains all relevant labels (with the global IDs removed)
    for i, idxs in enumerate(merge_idxs):
        append_value(test_labels, i, labels[idxs, 1:], append_list=False)

    # Instantiate a sub-images dictionary, which is the counterpart to test_labels
    test_images = {}

    # Define the original wavenumber scale, used to convert the left/right pixel values into the corresponding
    # wavenumber range, which is required in later analysis steps
    wavenumber_scale = np.arange(268, 1611, 2.625)

    # Now, cycle through each unique ID (key), retrieving the associated sub-images based on the label information.
    # The starting time steps for each sub-image are determined by the earliest time step for each set of sub-images.
    # (e.g. [498, 1, 443, *359*, 0809] & [498, 1, 379, *354*, 0809] will use time step 354 for both,
    # otherwise the time steps for each sub-image will not align, which is VERY bad)
    for key, val in test_labels.items():
        # Retrieve the starting time step for the current sub-image set
        min_timestep = np.min(np.array(val[:, 2]).astype('int'))

        # Instantiate a list of sub-images for the current ID, which will be appended to the dictionary
        snippets = []

        # Retrieve the scan
        dataset = select_scan(particles=[f'Particle_{val[0, 3]}'])
        for data, _ in dataset.as_numpy_iterator():
            # Cycle through each label
            for i, v in enumerate(val):
                # Convert the left/right edges to ints
                left, right = int(v[4]), int(v[5])

                # Crop the current snippet from the scan
                snippet = data[min_timestep:min_timestep + min_duration, left:right].squeeze()

                # Replace the left/right pixel edges with the corresponding min/max wavenumber range
                test_labels[key][i, 4] = wavenumber_scale[left]  # remove the unnecessary left/right edges
                test_labels[key][i, 5] = wavenumber_scale[right - 1]  # remove the unnecessary left/right edges

                # Resize the snippet width using linear interpolation, and append it to the list
                snippets.append(cv2.resize(snippet, dsize=(width, snippet.shape[0]), interpolation=cv2.INTER_LINEAR))

            # Edit the value entry within the test_labels dictionary
            test_labels[key][:, 2] = min_timestep  # set all starting time steps as min_timestep

        # Convert the snippets list to an array, and append them to the test_images dictionary
        append_value(test_images, key, np.array(snippets), append_list=False)

    # Instantiate two temporary lists to contain the snippets and their matching labels
    temp_images = []
    temp_labels = []
    for (key, val1), (_, val2) in zip(test_images.items(), test_labels.items()):  # 1st key == 2nd key
        # Add the unique ID onto the start of the labels list
        val2 = np.concatenate((np.array([[key] * val2.shape[0]]).T, val2), axis=1)  # convert key into a column vector

        # Append the snippets and their associated labels to the appropriate lists
        temp_images.extend(val1)
        temp_labels.extend(val2)

    # Convert to arrays
    test_images = np.array(temp_images)
    test_labels = np.array(temp_labels)

    # Normalise the sub-images individually
    sample_max = np.max(test_images, axis=(1, 2), keepdims=True)
    sample_min = np.min(test_images, axis=(1, 2), keepdims=True)
    test_images = (test_images - sample_min) / (sample_max - sample_min)

    # Expand the dimensions of the images to be compatible with the Siamese-CNN. The labels are kept as they are, as
    # this test dataset is an unsupervised procedure
    test_images = np.expand_dims(test_images, axis=-1)

    # Save the entire database and individual datasets to file
    path = f'./data/snippets/{params["c_ver"]}/real_pairs'
    Path(path).mkdir(parents=True, exist_ok=True)

    np.save(f'{path}/test_images.npy', test_images)
    np.save(f'{path}/test_labels_unsupervised.npy', test_labels)

    return


def view_groups(params):
    """ Displays the fine-tuning dataset, one 'Group snippet' at a time (i.e. if a Group contains three Tracks, then it
    will display snippets of each Track between the same time steps).

    This purpose of this function is to enable viewing each Track pair that will be seen by the Siamese-CNN during
    fine-tuning, in order to manually assign labels. If you have a better method to do this, then ignore this function.

    Args:
        params: Dict, The hyperparameter dictionary
    """
    # Load in the fine-tuning dataset, and the associated 'unsupervised labels'
    data = np.load(f'./data/snippets/{params["c_ver"]}/real_pairs/test_images.npy').squeeze()
    unsupervised_labels = np.load(f'./data/snippets/{params["c_ver"]}/real_pairs/test_labels_unsupervised.npy')

    # Create arrays of unique global IDs and particle numbers
    unique_ids = np.unique(unsupervised_labels[:, 0])
    unique_particles = np.unique(unsupervised_labels[:, 4])

    # Loop through each unique particle (i.e. scan)
    for i, unique_particle in enumerate(unique_particles):
        # Instantiate a dictionary to hold all snippets from all Groups within the current particle
        current_group = {}

        # Instantiate two counters; one to track when a new group is found within the current particle, and the other
        # to give a unique ID to the current group
        old_snippet = -1
        count = 0

        # Sort the IDs numerically
        unique_ids = unique_ids[np.argsort(unique_ids.astype('int'))]

        # Cycle through each ID, creating a dictionary entry for each Group in the current particle
        for unique_id in unique_ids:
            # Find the intersection of the current ID and particle
            idxs = np.argwhere(unsupervised_labels[:, 0] == unique_id).squeeze()
            particle_ids = np.argwhere(unsupervised_labels[idxs, 4] == unique_particle).squeeze()

            if len(particle_ids) > 0:
                if int(unsupervised_labels[idxs[0], 1]) > old_snippet:  # if new snippet is found for current group
                    old_snippet = int(unsupervised_labels[idxs[0], 1])  # count stays same, old_snippet updated
                else:  # if the previous group has finished and a new one has begun (for current particle)
                    count += 1  # count is incremented
                    old_snippet = -1  # old_sub is reset back to -1

                # Append the current snippet information to relevant key in the dictionary
                append_value(current_group, count, unsupervised_labels[idxs])

        # Cycle through each group within the current particle
        for key, val in current_group.items():
            # Look through each track in the current snippet
            for v in val:
                # Find the indices of the current group
                idxs = np.argwhere(unsupervised_labels[:, 0] == v[0][0]).squeeze()

                # Retrieve the images for the current group
                images = data[idxs]

                # Sort the sub-image and unsupervised label orders from lowest to highest wavenumber
                key_sort = np.argsort(v[:, 2].astype('int'))
                images = images[key_sort]
                v_sort = v[key_sort]

                # Plot the images
                plt.figure(figsize=(2.5 * images.shape[0], 6))
                plt.suptitle(
                    f'Particle {v_sort[0][4]}; Time Step {v_sort[0][3]}; Global ID {v_sort[0][0]}; Sub {v_sort[0][1]}',
                    fontsize=14)
                for i in range(images.shape[0]):
                    plt.subplot(1, images.shape[0], i + 1)
                    plt.title(f'{v_sort[i][2]}', fontsize=10)
                    plt.imshow(images[i], cmap='gray')
                plt.show()

    return


def fine_tuning_labels(params):
    """ Assigns manually labelled correlations to the 'real_pairs' test dataset (used to fine-tune the Siamese-CNN).
    This also has the effect of removing unlabelled samples from the fine-tuning dataset.

    NOTE: This function works similarly to create_snippets() + produce_inference(), in that it creates pairs of Tracks
        and assigns correlation labels. The main differences are that the Track pairs are real (i.e. no augmentations),
        and the labels have been manually assigned, rather than determined through synthesis.

    NOTE: Similar to view_groups(), if you already have some manually assigned labels, then you can ignore this
    function. You will need your fine-tuning data/correlation labels/unsupervised labels in the formats:

        data: Array, shape = (number of pairs, pairs, height, width, 1)

        correlation labels: Array, shape = (number of pairs, 1)

        unsupervised labels: Array, shape = (number of pairs, pairs, ?), where ? is whatever identifying information
            that you need (e.g. ? = [groupID, snippetID, peak_centroid, start time step, particle number])

    Args:
        params: Dict, The hyperparameter dictionary
    """
    # Define the default filepath
    f = f'./data/snippets/{params["c_ver"]}/real_pairs'

    # Load in the group_test dataset, and the associated 'unsupervised labels'
    data = np.load(f'{f}/test_images.npy')
    unsupervised_labels = np.load(f'{f}/test_labels_unsupervised.npy')

    # Define the manual labels
    # Format:
    # - The values in the lists are: 0 = negative, 1 = positive, .5 = not sure, .25/75 = not sure (but closer to 0/1)
    # -- The labels that are neither 0/1 are ignored
    # - Each parent key is the particle (e.g. Particle_0039)
    # -- But, a particle might contain multiple Groups, hence there is a further delineation using the (approximate)
    #    first time step (e.g. '0506_525' means Particle_0506; the Group beginning around time step 525)
    # - Each child key is the range of snippetIDs (e.g. '0-3' means correlations assigned to these inclusive snippets)
    # - Each value is the flattened 'upper triangle' of correlations (without the main diagonal), for example if you
    #   manually label a Group with 3 Tracks:
    #   [.  1  0  0]
    #   [.  .  0  0] --> [1, 0, 0, 0, 0, 1] (i.e. each row/column is a Track)
    #   [.  .  .  1]
    #   [.  .  .  .]
    manual_labels = {
        '0039': {'0-3': [1, 0, 0, 0, 0, 1],
                 '4-8': [0, 0, 1],
                 '9-9': [1]},

        '0070': {'0-4': [.5, .5, 0],
                 '5-25': [0, 1, 0],
                 '26-27': [.5],
                 '28-43': [1]},

        '0176': {'0-3': [1, .5, 1, .5, 1, .5, .5, 1, .5, 1, .5, .5, .5, .5, .5, .5, 1, .5, .5, 1, .5],
                 '4-6': [1, .5, 1, 1, .5, .5, 1, 1, .5, .5, .5, .5, 1, .5, .5],
                 '7-12': [1, .5, 1, 1, .5, 1, 1, .5, .5, 1],
                 '13-17': [.5],
                 '18-19': [1],
                 '20-29': [.5]},

        '0186': {'0-0': [0]},

        '0197': {'0-21': [1]},

        '0265': {'0-19': [1]},

        '0284': {'0-2': [.5, .5, .5, 1, 1, 1]},

        '0362': {'0-34': [1]},

        '0506_525': {'0-17': [1, 1, 1],
                     '18-29': [1]},

        '0506_789': {'0-1': [1, 1, 1],
                     '2-3': [1]},

        '0506_402': {'0-1': [.75, .75, .75, 1, 1, 1],
                     '2-3': [1, 1, 1]},

        '0595': {'0-0': [0]},

        '0644': {'0-2': [.5, .5, 0],
                 '3-4': [0]},

        '0694_370': {'0-3': [1]},

        '0694_218': {'0-2': [.5],
                     '3-8': [1]},

        '0716': {'0-1': [.5, .5, 1]},

        '0865': {'0-4': [.5, .5, .5],
                 '5-5': [.5]},

        '0881': {'0-3': [.5, .5, .5, .5, 0, 0, 1, 1, 0, 0],
                 '4-6': [.5, .5, .5, 0, 1, 0],
                 '7-9': [.5, .5, .5],
                 '10-19': [.5, .5, 1],
                 '20-20': [1]},

        '1045': {'0-1': [1, 1, 1]},

        '1362': {'0-5': [1]}
    }

    # Instantiate three lists, one for snippet pairs, and the others for correlation and unsupervised labels
    group_test_dataset = []
    group_test_corr = []
    group_test_unsupervised = []

    # Cycle through the first layer of the dictionary (i.e. particles)
    for key1, val1 in manual_labels.items():
        # Retrieve the images and unsupervised labels for the current particle
        idxs1 = np.argwhere(unsupervised_labels[:, 4] == key1.split('_')[0]).squeeze()
        particle_data = data[idxs1]
        particle_unsupervised = unsupervised_labels[idxs1]

        # Retrieve the indices of each unique ID
        id_diff = np.diff(particle_unsupervised[:, 0].astype('int'))
        new_ids = np.argwhere(id_diff != 0).squeeze() + 1  # +1 adjusts the indexing

        # <editor-fold desc="---=== [+] Retrieve the Current Group ===---">
        # If there are multiple groups within the current particle (i.e. key1 format = ParticleNum_StartTimeStep),
        # then search for the current group (defined by StartTimeStep)
        # (Problem: 2+ groups can have the same key1 format - i.e. the two groups begin at the same time step. One
        # solution would be to replace StartTimeStep with GlobalID, as this is non-degenerate)
        if len(key1.split('_')) > 1:
            # Instantiate a list of indices where the different group splits occur
            splits = [0]  # index 0 is a default
            prev_sub = 0
            for i in new_ids:
                # If the sub-value has reset back to zero, then set the current sub-index as the start of a new group
                if int(particle_unsupervised[i, 1]) <= prev_sub:
                    splits.append(i)

                # Update the previous sub value counter
                prev_sub = int(particle_unsupervised[i, 1])

            # Loop through each group within the current particle to find the correct group (i.e. matches key1 & val1)
            for j in range(len(splits)):
                # Retrieve the next group...
                if j == len(splits) - 1:
                    group_data = particle_data[splits[j]:]
                    group_unsupervised = particle_unsupervised[splits[j]:]
                else:
                    group_data = particle_data[splits[j]:splits[j + 1]]
                    group_unsupervised = particle_unsupervised[splits[j]:splits[j + 1]]

                # ...use the current group if its starting time step matches that of the time step within key1
                if group_unsupervised[0, 3] == key1.split('_')[-1]:  # ignore warning
                    # e.g. for the key1's: '0506_402', '0506_525' '0506_789', this if statement makes sure that it is
                    # taking the correct Group from the set of all '0506' 'unsupervised labels' data
                    break

        # If there is only one group within the current particle (i.e. key1 format = ParticleNum), then use all data
        else:
            group_data = particle_data
            group_unsupervised = particle_unsupervised
        # </editor-fold>

        # Cycle through the second layer of the dictionary (i.e. snippets)
        for key2, val2 in val1.items():
            # Reformat the second key into a range
            first = int(key2.split('-')[0])
            last = int(key2.split('-')[-1])
            for k in range(first, last + 1):  # +1 to be inclusive of the second value
                # Retrieve the sub-images and sub-unsupervised labels for the current dictionary item
                idxs2 = np.argwhere(group_unsupervised[:, 1] == str(k)).squeeze()
                sub_data = group_data[idxs2]
                sub_unsupervised = group_unsupervised[idxs2]

                # Sort the labels and corresponding data in time step order
                sort_key = np.argsort(sub_unsupervised[:, 2].astype('int'))
                sub_data = sub_data[sort_key]
                sub_unsupervised = sub_unsupervised[sort_key]

                # Produce the indices for all possible combinations, and assemble the pairs data array
                idx = np.arange(sub_data.shape[0])
                combs = np.array(list(combinations(idx, 2)))
                combs_data = sub_data[combs]
                combs_unsupervised = sub_unsupervised[combs]

                # Extend the lists with their respective data
                group_test_dataset.extend(combs_data)
                group_test_corr.extend(val2)
                group_test_unsupervised.extend(combs_unsupervised)

    # Convert lists to arrays
    group_test_dataset = np.array(group_test_dataset)
    group_test_corr = np.expand_dims(np.array(group_test_corr), axis=-1)
    group_test_unsupervised = np.array(group_test_unsupervised)

    # Instantiate a counter to assign unique partition IDs to pairs of images (Note: images, not snippets)
    partition_id = 0

    # Create an array of all unique manually labelled particles
    particle_nums = np.unique(group_test_unsupervised[:, 0, 4])

    # Cycle through the list of manually labeled particle numbers
    for particle in particle_nums:
        # Find the pairs which are associated with the current particle
        particle_idx = np.argwhere(group_test_unsupervised[:, 0, 4] == particle).squeeze()

        # Find the unique pairs
        pairs = np.unique(group_test_unsupervised[particle_idx, :, 2], axis=0)

        if pairs.ndim == 2:  # i.e. if there is more than one unique pair...
            # Assign unique IDs to the unique pairs (all snippets of the pair will share the same ID)
            for pair in pairs:
                # Find the indices of the current pair
                pair_idx = np.argwhere(np.logical_and(group_test_unsupervised[particle_idx, 0, 2] == pair[0],
                                                      group_test_unsupervised[particle_idx, 1, 2] == pair[1]))

                # Update the previous ID with the partition ID
                # (Note: The partition ID is repeated, one for each in the pair ... Just thought you should know :) )
                group_test_unsupervised[particle_idx[pair_idx], :, 0] = partition_id

                partition_id += 1  # increment the partition ID

        else:  # ...else give all pairs the same partition ID as they're all snippets pairs of the same image pair
            group_test_unsupervised[particle_idx, :, 0] = partition_id

            partition_id += 1  # increment the partition ID

    # Save the group test dataset, with its associated binary and unsupervised labels, to file
    np.save(f'{f}/manual_test_images.npy', group_test_dataset)
    np.save(f'{f}/manual_test_labels.npy', group_test_corr)
    np.save(f'{f}/manual_test_unsupervised.npy', group_test_unsupervised)

    return


def load_group_test(params):
    """ Yield batches of manually labelled fine-tuning dataset with its associated correlation and unsupervised labels,
    similarly to generate_pairs(), for fine-tuning the Siamese-CNN

    Args:
        params: Dict, The hyperparameter dictionary

    Yields:
        image: Array, Data for the current batch
        label: Array, Correlation labels for the current batch
        unsupervised: Array, Additional information for each pair within the current batch
    """
    # Define the default filepath
    f = f'./data/snippets/{params["c_ver"]}/real_pairs'

    # Load in the chosen dataset and its associated labels
    images = np.load(f'{f}/manual_test_images.npy')
    labels = np.load(f'{f}/manual_test_labels.npy')
    unsupervised = np.load(f'{f}/manual_test_unsupervised.npy')

    # Form a Tensorflow dataset to make batching easier
    temp_dataset = tf.data.Dataset.from_tensor_slices((images, labels, unsupervised)).batch(params['s_batch_size'],
                                                                                            drop_remainder=True)

    # Cycle through the dataset, yielding one batch at a time
    for image, label, unsuper in temp_dataset.as_numpy_iterator():
        yield image, label, unsuper


def kfold_cv(params, partition):
    """ Retrieves the chosen partition of the KFold Cross Validation on the group_test dataset, used for fine-tuning a
    pre-trained Siamese-CNN. The partitions are defined by each Track image, rather than each snippet, as otherwise
    there would be, on average, very similar data shared between each train/test partition.

    Args:
        params: Dict, The hyperparameter dictionary
        partition: Int, The chosen partition

    Returns:
        train_dataset: array, The training dataset of the chosen partition
        test_dataset: array, The testing dataset of the chosen partition
        train_labels: array, The training labels of the chosen partition
        test_labels: array, The testing labels of the chosen partition
    """
    # Define the default filepath
    f = f'./data/snippets/{params["c_ver"]}/real_pairs'

    # Load in the manually labelled dataset, correlation labels, and 'unsupervised' information
    x = np.load(f'{f}/manual_test_images.npy')
    y = np.load(f'{f}/manual_test_labels.npy')
    z = np.load(f'{f}/manual_test_unsupervised.npy')

    # Remove the non-binary correlation labels
    ones = (y == 1).squeeze()
    zeroes = (y == 0).squeeze()
    x = np.concatenate((x[ones], x[zeroes]), axis=0)  # combine the two
    y = np.concatenate((y[ones], y[zeroes]), axis=0)  # combine the two
    z = np.concatenate((z[ones], z[zeroes]), axis=0)  # combine the two

    # Retrieve the indices that allow for partitioning based on each Track, rather than each snippet
    partition_ids = np.unique(z[:, 0, 0].astype('int'))

    # Perform k-fold cross-validation (i.e. split the dataset into train/test partitions)
    # (NOTE: A random state is used to produce the same partitions each time the function is called)
    cv = KFold(n_splits=params['s_kfold'], shuffle=True, random_state=27)

    # Cycle through the partitions until the chosen one is reached, then return it
    # (Note: The partitioning process operates on a list of numerically ordered indices, as this is a more efficient
    # way of retrieving the requested partition when three arrays are required, as well as having to cycle through the
    # partitions up until the requested one)
    for n, (train, test) in enumerate(cv.split(partition_ids, partition_ids)):
        if n == partition:
            # Convert the train/test indices to the partition indices
            train_partition = partition_ids[train]
            test_partition = partition_ids[test]

            # Cycle through each partition ID for the current partitions, finding the indices of samples within the
            # unsupervised labels array, z, which match the partition ID, to create respective train/test lists
            # (Note: A stacked list comprehension is used here, which acts like list.extend(), as the final for loop
            # simply appends one value at a time from the result of the initial for loop)
            train_idx = [i[0] for tr in train_partition for i in np.argwhere(z[:, 0, 0] == f'{tr}')]
            test_idx = [i[0] for te in test_partition for i in np.argwhere(z[:, 0, 0] == f'{te}')]

            return x[train_idx], x[test_idx], y[train_idx], y[test_idx], z[train_idx], z[test_idx]


# </editor-fold>


if __name__ == "__main__":
    # Load chosen hyperparameters
    hyperparams = hyperparams_setup()

    # if hyperparams['c_ver'] == 'cae_v1':
    #     print('You cannot create new Track pairs dataset(s) on the original cae_v1 model\n(These should already exist for all scans, so you should not have to run this main_siamese_data.py module as main)')
    #     exit()

    # Run this if you want to save train/valid/test datasets of Track snippets to file (used to synthesise pairs)
    # create_snippets(hyperparams)

    # Run this if you want to save fixed valid/test synthesised Track pairs
    # produce_inference(hyperparams)

    # Run this if you want to save a dataset of Track snippets to file (from which the fine-tuning dataset is created)
    # create_snippets_ft(hyperparams)

    # Run this if you want to create the fine-tuning dataset of genuine pairs (from the same Group) of Track snippets
    # fine_tuning_labels(hyperparams)
