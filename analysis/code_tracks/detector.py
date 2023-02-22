"""
@Created : 05/02/2021
@Edited  : 05/02/2021
@Author  : Alex Poppe
@File    : detector.py
@Software: Pycharm
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import opening, rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from pathlib import Path


# Takes input coordinates and converts them to a len(num_time_steps) list containing peak centroids and bounds
def bounds_generator(coords):
    """ Take in input coords, converting them into lists containing peak centroid, bound, and time step

    Args:
        coords: (n, 2) Array, each index contains [time, position] pairs of each detected outlier,
            e.g. [[3, 372], [3, 373], [3, 374], ..., [3, 380], [3, 381], [3, 382], ..., [6, 373], [6, 374]]

    Returns:
        track_centroids: List of lists, each sublist is for a different track section, and contains each peak centroid.
            e.g. [[373], ..., [381], ..., [373.5]]
        track_bounds: List of lists, each sublist is for a different track section, and contains each peak bound.
            e.g. [[371, 373], ..., [380, 382], ..., [373, 374]]
        track_times: List of lists, each sublist is for a different track section, and contains each peak time step.
            e.g. [[3], ..., [3], ..., [6]]
    """
    # Cycle through each time step, creating lists for centroids, bounds, and time steps for those coordinates
    track_centroids = []
    track_bounds = []
    track_times = []
    if len(coords) != 0:  # skip this process if there are no detections in the whole scan
        for timestep in np.unique(coords[:, 0]):
            # Sort all outliers for current time step based on the column, then discard rows (already known (timestep))
            temp_coords = coords[coords[:, 0] == timestep][:, 1]
            outliers = temp_coords[np.argsort(temp_coords, axis=0)]

            # Instantiate lists that will contain all outlier centroids and bounds for the current spectrum
            peak_centroids = []
            peak_bounds = []
            peak_times = []

            if len(outliers) > 1:
                # Calculate outlier index spacings to determine whether they are the same or different peaks
                # (outliers are connected if their spacing is between 1-2 pixels)
                gaps = np.diff(outliers)

                # Start a counter to track the current index of the gaps array
                k = 0

                # Iterate through a maximum number of steps equal to the length of the gaps array
                # (This would only iterate through the maximum number of times if every peak had a width of 1 px)
                for _ in range(len(gaps) + 1):
                    if k == len(gaps):
                        # Check for the final outlier first to stop a possible IndexError from occurring
                        peak_centroids.append(outliers[k])
                        peak_bounds.append([outliers[k], outliers[k]])
                        peak_times.append(timestep)

                        break
                    if gaps[k] > 2:  # (!= 1)
                        # Input only a single outlier index if it is a 'lone peak'
                        peak_centroids.append(outliers[k])
                        peak_bounds.append([outliers[k], outliers[k]])
                        peak_times.append(timestep)

                        k += 1
                    else:
                        # Find the next greater than 2 gap to determine the width of the current peak
                        skip = np.where(gaps[k:] > 2)  # used for indexing (!= 1)
                        if len(skip[0]) != 0:
                            # If there exists a greater than 2 gap later in the gaps array, find the current peak
                            peak_centroids.append((outliers[k] + outliers[skip[0][0] + k]) / 2)
                            peak_bounds.append([outliers[k], outliers[skip[0][0] + k]])
                            peak_times.append(timestep)

                            # k skips ahead by the width of the current peak + 1
                            k += skip[0][0] + 1
                        else:
                            # This is the final outlier if there are no more gaps greater than 2
                            peak_centroids.append((outliers[k] + outliers[-1]) / 2)
                            peak_bounds.append([outliers[k], outliers[-1]])
                            peak_times.append(timestep)

                            break

            else:  # this covers the situation where a track only has a single-width peak in a time step
                peak_centroids.append(outliers[0])
                peak_bounds.append([outliers[0], outliers[0]])
                peak_times.append(timestep)

            # Append the centroids and bounds from the current time step/spectrum to their respective global lists,
            # even if they are empty
            track_centroids.extend(peak_centroids)
            track_bounds.extend(peak_bounds)
            track_times.extend(peak_times)

    return track_centroids, track_bounds, track_times


# Takes input scan and finds the coordinates of the top percentile_outliers
def percentile_finder(scan, percentile_outliers):
    """ Receives an input scan, finding the coordinates of the top percentile_outliers percentile data points. Note that
    if a List is given to the percentile_outliers parameter, outliers are only kept if they exist in both the column and
    row-wise percentiles (i.e. an intersection is made of the two detected sets)

    Args:
        scan: Array, The input scan to detect outliers on
        percentile_outliers: Int or List of Arrays, If Int: The percentile of data points to locate, If List of Arrays:
            The column and row-wise percentiles for locating data points

    Returns:
        coords: array, An (n, 2) array containing the [time step, wavenumber] positions (in pixels) of all outliers
    """
    # Single percentile case
    if isinstance(percentile_outliers, int):
        # Flatten and sort the difference scan
        diff_sort = np.argsort(scan, axis=None)

        # Calculate the number of points that form the top-percentile_outliers data points
        num_points = int(diff_sort.shape[0] * (1 - (percentile_outliers / 100)))

        # Calculate number of non-zero values in the difference spectra, to prevent zeroes from becoming outliers
        # (i.e. limit num_points to non_zeroes if it extends into the zero-values)
        non_zeroes = scan.size - np.where(np.sort(scan, axis=None) == 0)[0].size
        if num_points > non_zeroes:
            num_points = non_zeroes

        # Find the (row, col) values of the top-percentile_outliers data points
        rows = diff_sort[-num_points:] // scan.shape[1]
        cols = diff_sort[-num_points:] % scan.shape[1]

        # Stack the rows and columns together into a coordinates array
        coords = np.stack((rows, cols), axis=-1)  # i.e. coords = [[row_0, col_0], [row_1, col_1], ...]

        return coords

    # Multiple percentiles case (i.e. probability density functions (PDFs))
    elif isinstance(percentile_outliers, list):
        # Form a list of the transposed and regular scan, the former is used in column-wise (wavenumber) percentile
        # detections, and the latter is used in row-wise (time step) percentile detections
        orientations = [scan.T, scan]

        # Instantiate a list to contain the detections from, firstly, the column-wise percentile detections and,
        # secondly, the row-wise percentile detections
        double_detections = []

        # Cycle through each percentile and scan orientation pair
        for percentile_outlier, orientation in zip(percentile_outliers, orientations):
            # Instantiate a list of detections for the current orientation
            detections = []

            # Split the percentile and scan orientation into individual columns/rows
            for j, (percentile, vector) in enumerate(zip(percentile_outlier, orientation)):

                # Flatten and sort the difference vector
                diff_sort = np.argsort(vector, axis=None)

                # Calculate the number of points that form the top-percentile_outliers data points
                num_points = int(diff_sort.shape[0] * (1 - (percentile / 100)))

                # Calculate number of non-zero values in the difference spectra, to prevent zeroes from becoming
                # outliers (i.e. limit num_points to non_zeroes if it extends into the zero-values)
                non_zeroes = vector.size - np.where(np.sort(vector, axis=None) == 0)[0].size
                if num_points > non_zeroes:
                    num_points = non_zeroes

                if num_points != 0:  # skip 100th percentile detections
                    # Find the (row, col) values of the top-percentile_outliers data points
                    if vector.shape[0] == 1000:  # (first orientation)
                        rows = diff_sort[-num_points:]  # e.g. [56, 57, ..., 302, 303, ...]
                        cols = (j * np.ones(diff_sort[-num_points:].shape)).astype(int)  # e.g. [0, 0, ..., 0, 0, ...]
                    else:  # (second orientation)
                        rows = (j * np.ones(diff_sort[-num_points:].shape)).astype(int)
                        cols = diff_sort[-num_points:]

                    # Stack the rows and columns together into a coordinates array
                    # (Note: This must be converted to list; see future explanation*)
                    detections.extend(np.stack((rows, cols), axis=-1).tolist())  # [[row_0, col_0], ...]

            double_detections.append(detections)

        # Keep only the coords that intersect between both PDFs
        # (*Note: The lists are mapped to tuples, then converted to sets and compared. Finally they are converted to
        # a list (to remove the set brackets), then to an array (to remove the tuples). If this intersection is
        # performed directly on lists or arrays then the process can take over 200x longer!)
        coords = np.array(list(set(map(tuple, double_detections[0])) & set(map(tuple, double_detections[1]))))

        return coords

    else:
        print('Incorrect percentile_outliers given to function!\nExiting...')
        exit()

    return


# Calculate difference scan (this would be in utils.py, but it would cause an ImportError due to circular dependencies)
def difference_scans(data, recon, offset=0.05):
    """ Calculate the difference scans from the input/recon data

    Args:
        data: Array, The input scan
        recon: Array, The reconstructed scan
        offset: Float, The offset performance (i.e., decimal percentage of the standard deviation for the entire scan)
    """
    # Calculate the global Offset
    global_offset = np.std(data) * offset

    # Apply a Savitzky-Golay filter to the input and difference spectra, then calculate the 'difference' spectra
    data = savgol_filter(data.squeeze(), window_length=7, polyorder=2, axis=0)
    recon = savgol_filter(recon.squeeze(), window_length=7, polyorder=2, axis=0)
    difference = data - recon

    # Set values below global_offset to global_offset, then reduce all data by global_offset (i.e. min value = 0)
    return np.squeeze(np.max((difference, np.zeros(difference.shape) + global_offset), 0) - global_offset)


# Detects outliers (peaks) in the difference spectra based on specified offset and sigma parameters
def detector(params, data, recon, label_str=None, save_figs=False):
    """ Uses the following pipeline:
        - Calculate the global Offset value based on the std of the entire scan
        - De-noise the input and reconstructed spectra using a Savitzky-Golay filter
        - Calculate the Difference spectra (Input - Reconstruction), offset by the Offset value, for all spectra
        - Find the coordinates of percentile outliers in the difference scan
        - Lower the percentile to detect more outliers amongst wavenumbers and time steps that already have detections
        - Build a scaling percentile for each dimension to make further detections (PDFs)
        - (Morphologically open to remove noise at the previous three detection stages)
        - Output the binary detections scan

    Args:
        params: Dict, The hyperparameter dictionary
        data: array, The input scan
        recon: array, The reconstructed scan
        label_str: Str, Short-hand label for the current scan
        save_figs: Bool, if True: Saves figures
    """
    # Define percentiles for each detections stage
    thr_1 = 98  # 98
    thr_2 = 96  # 96
    thr_3 = [96, 90]  # [96, 90]

    # Remove redundant dimensions
    data = np.squeeze(data)
    recon = np.squeeze(recon)

    # Calculate difference scans
    difference = difference_scans(data, recon)

    # Find the indices of the top-percentile_outliers data points
    if np.std(difference) == 0.0:  # 0 std means no outliers, therefore create an empty list
        coords = []
    else:
        coords = percentile_finder(scan=difference, percentile_outliers=thr_1)

    # ---=== 1st detection step (top-%) ===---
    # Generate an empty array, and fill it with the original detected outliers, from coords
    binary_outliers_1 = np.zeros((data.shape[0], data.shape[1]))
    for coord in coords:
        binary_outliers_1[coord[0], coord[1]] = 1

    # Morphologically open the outliers scan
    opened_outliers_1 = opening(binary_outliers_1, rectangle(3, 3))

    # Generate a new coords array, featuring only the coords which 'survived' morphological opening
    coords = np.argwhere(opened_outliers_1 == 1)

    if len(coords) > 0:
        # ---=== 2nd detection step (informed look) ===---
        # Look again for outliers in spectra already containing others, this time with a lower threshold
        # (First, 'connect' timesteps that are close-by to account for some deviance in detections due to physical or
        # algorithmic errors)
        extended_ts = np.unique(coords[:, 0])

        # Find the indices of the 'small gaps' to fill-in
        # (NOTE: This doesn't artificially add detections, it only allows detections near existing ones)
        small_gaps = np.where(np.logical_and(np.diff(extended_ts) <= 5, np.diff(extended_ts) > 1))[0]
        for gap in reversed(small_gaps):  # reversed to preserve gap indices
            # Create the timesteps between the gap
            missing_ts = np.arange(extended_ts[gap] + 1, extended_ts[gap + 1], 1)

            # Insert these timesteps into the existing time steps array
            extended_ts = np.insert(extended_ts, gap + 1, missing_ts)

        # Find the 'pseudo-coordinates' in the trimmed scan
        informed_coords = percentile_finder(scan=difference, percentile_outliers=thr_2)
        informed_coords = informed_coords[np.argsort(informed_coords[:, 0])]  # sort row-wise

        # Find the new detected rows (time steps) that are not a part of coords, and remove them...
        remove_ts = np.setdiff1d(np.unique(informed_coords[:, 0]), extended_ts)
        for remove in reversed(remove_ts):
            informed_coords = np.delete(informed_coords, np.where(informed_coords[:, 0] == remove)[0], axis=0)
        # ...do the same for columns (wavenumbers)
        remove_wn = np.setdiff1d(np.unique(informed_coords[:, 1]), np.unique(coords[:, 1]))
        for remove in reversed(remove_wn):
            informed_coords = np.delete(informed_coords, np.where(informed_coords[:, 1] == remove)[0], axis=0)

        # Create a scan featuring the detected outliers
        # (NOTE: no intersection is necessary between detection stages 1 and 2 because stage 1 is,
        # by nature of coming from a larger percentile, a subset of stage 2)
        binary_outliers_2 = np.zeros((data.shape[0], data.shape[1]))
        for coord in informed_coords:
            binary_outliers_2[coord[0], coord[1]] = 1

        # Morphologically open the outliers scan
        opened_outliers_2 = opening(binary_outliers_2, rectangle(3, 3))

        # Generate a new coords array, featuring only the coords which 'survived' morphological opening
        informed_coords = np.argwhere(opened_outliers_2 == 1)

        if len(informed_coords) > 0:
            # ---=== 3rd detection step (probability density function) ===---
            # Look again for outliers in the whole scan by creating a probability density function based on detected
            # outliers from the informed look stage, and lowering the top-% pixel-wise based on these detections

            # Find all time steps and wavenumbers already detected
            existing_ts, counts_ts = np.unique(informed_coords[:, 0], return_counts=True)
            existing_wn, counts_wn = np.unique(informed_coords[:, 1], return_counts=True)

            # Create the probability density estimates for the wavenumbers and time steps
            ts_occurrence = np.zeros(data.shape[0])  # time steps
            ts_occurrence[existing_ts] = counts_ts
            wn_occurrence = np.zeros(data.shape[1])  # wavenumbers
            wn_occurrence[existing_wn] = counts_wn

            # Smooth only the wavenumber probability density estimates, then normalise them both between 0 - 1
            # (NOTE: The time step probability density estimate is not smoothed, this reduces the likelihood that
            # noise from a small gap between two events is detected as a noise outlier, thus filling the gap)
            # (NOTE: It is smoothed for wavenumbers, however, as we attempt to account for situations where the
            # currently detected parts of a Track are all shifted away from the centroid)
            ts_occurrence_n = (ts_occurrence - np.min(ts_occurrence)) / (np.max(ts_occurrence) - np.min(ts_occurrence))
            wn_occurrence_s = gaussian_filter1d(wn_occurrence, sigma=5, axis=-1)
            wn_occurrence_n = (wn_occurrence_s - np.min(wn_occurrence_s)) / (
                    np.max(wn_occurrence_s) - np.min(wn_occurrence_s))

            # Create the column-wise (wavenumber) and row-wise (time step) top-% based on their occurrence values
            wn_percentile = thr_3[0] + wn_occurrence_n * (thr_3[1] - thr_3[0])
            ts_percentile = thr_3[0] + ts_occurrence_n * (thr_3[1] - thr_3[0])

            # Make 100th percentile the lowest value, as we do not want to find values where none have been found
            # previously along those rows/columns
            wn_percentile[np.argwhere(wn_percentile == thr_3[0])] = 100.0
            ts_percentile[np.argwhere(ts_percentile == thr_3[0])] = 100.0

            # Find the new coordinates using the column and row-wise scaled percentiles
            prob_coords = percentile_finder(scan=difference, percentile_outliers=[wn_percentile, ts_percentile])

            # Assemble the complete coords array by taking an OR of the set of coords and prob_coords, then converting
            # back to an array
            prob_coords = np.array(
                list(set(map(tuple, informed_coords.tolist())) | set(map(tuple, prob_coords.tolist()))))

            # Create a scan featuring the pre-operated outliers
            binary_outliers_3 = np.zeros((data.shape[0], data.shape[1]))
            for coord in prob_coords:
                binary_outliers_3[coord[0], coord[1]] = 1

            # Morphologically open the outliers scan (this is the final detected output in 2D array format)
            opened_outliers_3 = opening(binary_outliers_3, rectangle(3, 3))

    if save_figs:
        # ---=== Plot detections overlaid onto the scan, colour-coded for each detection stage ===---
        plt.figure(figsize=(6, 8))
        plt.title(f'Morphological Opening Stages [3, 3]', fontsize=14)

        # Create an empty mask that will be filled in with each detection stage
        # (Note: The later coords arrays are plotted first, otherwise they will hide the earlier ones)
        empty_mask = np.zeros(np.squeeze(difference).shape)
        mask = np.ma.masked_array(empty_mask, empty_mask != 1)

        # PDF in red
        for row, col in np.argwhere(opened_outliers_3 == 1):
            mask[row, col] = 3

        # Informed in green
        for row, col in np.argwhere(opened_outliers_2 == 1):
            mask[row, col] = 2

        # Top-% in white
        for row, col in np.argwhere(opened_outliers_1 == 1):
            mask[row, col] = 1

        # Create a listed colourmap to plot the detection stages with their respective colours
        cmap = ListedColormap(['w', 'g', 'r'])
        cmap.set_bad(alpha=1)
        sections = [0, 1.1, 2.1, 3]  # The middle bounds are inclusive of the second/third colour, so make it >1/2
        norm = BoundaryNorm(sections, cmap.N)
        plt.imshow(mask, cmap=cmap, norm=norm, alpha=1)
        plt.xlabel('Pixel', fontsize=12)
        plt.ylabel('Time Step', fontsize=12)
        plt.tight_layout()

        if params['molecule'] == 'BPT':
            path = f'./analysis/{params["c_ver"]}/detections'
        else:
            path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/detections'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{path}/particle_{label_str}_1_detections.png')
        plt.close()

    # Return the appropriate opened detections scan (or an equivalent zeroes, if nothing was detected)
    if len(coords) == 0:
        return np.zeros(data.shape)
    else:
        return opened_outliers_3
