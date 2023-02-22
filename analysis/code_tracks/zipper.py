"""
@Created : 05/02/2021
@Edited  : 31/05/2022
@Author  : Alex Poppe
@File    : zipper.py
@Software: Pycharm
@Description:
Compares all tracks, ignoring symmetrical comparisons, to see if they are close enough in wavenumber-space and time to
be merged. After all comparisons have been made, the matching tracks are merged together and a new track list is made.
"""
import numpy as np
from analysis.code_tracks.tracker import Track
from utils import append_value


def zipper(tracks, zip_count=0):
    """ Function to combine separate tracks deemed to be a part of the same whole

    The process for combining tracks is as follows:
        1. Cycle through each track, and on each one find a nearby track to compare it to (i.e. do not bother if the
            distance between two tracks' closest centroids is over max_switch)
        2. Place the sliding window with its top at the start of the earliest shared time step
        3. Ignore eligible windows (i.e. there must be a fair ratio of each track)
        4. Calculate and store the differences between the average centroid positions in each window
        5. Calculate the average centroid separation between the two tracks
        6. Mark the two tracks to be merged if their average centroid separation is less than the merge tolerance
        7. Repeat this 'marking' process for all track combinations (ignoring symmetric comparisons, i.e. ignore 1 -> 0
            if 0 -> 1 is already checked), then merge all tracks in the same 'marked groups' together
        8. Trim down any rows with bounds that are too wide in comparison to the mean bound width of that track. This
            is centred around the centroids of each trimmed row

    Args:
        tracks: List of all track classes (featuring the trace, trace_bounds, trace_time, etc.)
        zip_count: Int, The count of the number of times this scan has had zips attempted

    Returns:
        tracks: List of all track classes, combined as necessary
    """
    # Define zipper window parameters
    w = 10  # width
    h = 3  # height
    s = 1  # stride
    d = 5  # merge tolerance

    # Sort the tracks list from largest to smallest
    sorted_tracks = sorted(tracks, key=lambda x: -len(x.trace))

    #####
    # This piece of code is not needed if arrays are input (which they are currently not, so keep it)
    for i in range(len(sorted_tracks)):
        sorted_tracks[i].trace = np.array(sorted_tracks[i].trace)
        sorted_tracks[i].trace_time = np.array(sorted_tracks[i].trace_time)
        sorted_tracks[i].trace_bounds = np.array(sorted_tracks[i].trace_bounds)
    #####

    # Instantiate list and dictionary
    merge_assignments = {}  # indices in this list will be merged after all comparisons are made

    # <editor-fold desc="---=== [+] Assign Track Mergers ===---">
    # Loop through each possible combination of track comparisons and append the merge assignments to the dictionary
    # (ignoring symmetric comparisons, i.e. if 0 & 1 are compared, no need to compared 1 & 0, same comparisons are also
    # ignored, i.e. 0 & 0 are not compared, and the final track is ignored. There are (n-1)^2/2 comparisons to be made).
    for idx1, track1 in enumerate(sorted_tracks[:-1]):
        # Find the left & right-most centroids of the first track
        tr1_min = np.min(track1.trace)  # left
        tr1_max = np.max(track1.trace)  # right

        for idx2m, track2 in enumerate(sorted_tracks[idx1 + 1:]):  # upper triangle
            idx2 = idx2m + idx1 + 1  # correct the second indexing to select the correct position in the upper triangle

            # Find the left & right-most centroids of the second track
            tr2_min = np.min(track2.trace)  # left
            tr2_max = np.max(track2.trace)  # right

            # Check to see if the two tracks are close enough to attempt to zip them together
            # (i.e. if the closest edges are 'within range')
            if abs(tr1_min - tr2_max) < w or abs(tr1_max - tr2_min) < w:  # abs() as we don't know which is left/right
                # Find the earliest and latest timesteps between both Tracks
                latest_start = max(np.min(track1.trace_time),
                                   np.min(track2.trace_time))

                earliest_stop = min(np.max(track1.trace_time),
                                    np.max(track2.trace_time))

                # Calculate window start/top positions
                first_window = latest_start - h + 1  # i.e. first window position that may contain both Tracks
                last_window = earliest_stop + 1  # i.e. last window position that may contain both Tracks

                # Instantiate counters
                sum_mean_sep = 0  # the sum of all mean separations
                count_mean_sep = 0  # the number of separations calculated
                for t in range(first_window, last_window, s):
                    # Calculate the mean separation within the current window
                    mean_sep = window(tr1=track1, tr2=track2, tw1=t, h=h, w=w)
                    if mean_sep >= 0:
                        sum_mean_sep += mean_sep  # append current mean separation to global list
                        count_mean_sep += 1
                        pass

                if count_mean_sep > 0:
                    global_mean_sep = sum_mean_sep / count_mean_sep  # mean separation of eligible windows
                    if 0 <= global_mean_sep <= d:
                        # Append the candidate track (idx2) to the dictionary entry of the focus track (idx1)
                        # to be later merged when the zipping process completes
                        append_value(merge_assignments, idx1, idx2)
    # </editor-fold>

    if len(merge_assignments.keys()) != 0:
        # <editor-fold desc="---=== [+] Assemble Merge Lists ===---">
        # Organise the combined merge dictionary into lists containing each set of to-be-merged tracks
        # (e.g. if merge_assignments={1: [2, 3], 3: [4], 5: [6]}, then merge_lists=[[1, 2, 3, 4], [5, 6]]).
        pre_merge_list = []
        for merge in merge_assignments.items():
            pre_merge_list.append([int(merge[0])] + merge[1])

        # Instantiate list of merge assignments
        merge_lists = []

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
                pre_merge_list = [list(merge) for merge in merge_lists]
                merge_lists = []

            # tl;dr cycle through sublists and search for any intersections with other sublists. If so, merge
            # them together and remove those sublists from 'searching' list. Any unmerged sublists are searched
            # until all sublists have either been merged or searched (i.e. pre_merge_list slowly empties)
            while len(pre_merge_list) > 0:
                first, *rest = pre_merge_list
                first = set(first)

                temp = -1
                if len(first) > temp:
                    temp = len(first)
                    next_rest = []
                    for r in rest:
                        if len(first.intersection(set(r))) > 0:
                            num_matches += 1
                            first |= set(
                                r)  # e.g. x = set([1, 2]), y = set([2, 3]) --> x |= y --> x = {1, 2, 3}
                        else:
                            next_rest.append(r)  # append the sublist to the list of future comparisons
                    rest = next_rest  # this essentially removes all merged lists this iteration
                merge_lists.append(first)
                pre_merge_list = rest

            first_attempt = False
            num_attempts += 1

        # Convert merge_lists from a list of sets to a list of lists
        merge_lists = [list(merge) for merge in merge_lists]
        # </editor-fold>

        # <editor-fold desc="---=== [+] Create New Track List (First Containing Unmerged Tracks) ===---">
        # Start of the new tracks list with the tracks not chosen for any merging
        merged_idxs = np.array([idx for idxs in merge_lists for idx in idxs]).flatten()
        all_idxs = np.arange(len(sorted_tracks))
        unmerged_idxs = [i for i in all_idxs if i not in merged_idxs]  # finds all indices not chosen for merging
        # Create a new track list to store the unmerged and soon-to-be merged tracks
        new_tracks = []
        # Loop through the list of unassigned tracks, appending each to the new list
        for i in unmerged_idxs:
            new_tracks.append(sorted_tracks[i])
        # </editor-fold>

        # <editor-fold desc="---=== [+] Perform Track Merges & Finalise Track List ===---">
        # Cycle through each merge assignment
        for merge in merge_lists:
            # Create a new track to represent the to-be-merged tracks
            new_tracks.append(Track())
            new_tracks[-1].trace_time = np.concatenate([sorted_tracks[m].trace_time for m in merge])
            order = np.argsort(new_tracks[-1].trace_time)
            new_tracks[-1].trace = np.concatenate([sorted_tracks[m].trace for m in merge])[order]
            new_tracks[-1].trace_bounds = np.concatenate([sorted_tracks[m].trace_bounds for m in merge])[order]
            new_tracks[-1].trace_time = new_tracks[-1].trace_time[order]

            # Update the zip counter of the new track
            new_tracks[-1].zip_count = zip_count
        # </editor-fold>

    else:
        new_tracks = sorted_tracks

    return new_tracks


def window(tr1, tr2, tw1, h, w):
    """ Function to place the zipper window over the two inputs tracks starting at the input timestep
    Args:
        tr1: class, the first track
        tr2: class, the second track
        tw1: int, the first timestep
        h: int, the height of the window
        w: int, the width of the window
    Returns:
        Not sure yet
    """
    # Calculate the end timestep for the current window
    tw2 = tw1 + h  # time window 2

    # Find if there are any traces between the two timesteps [tw1, tw2) for both tracks
    tr1_time_mask = np.logical_and(tr1.trace_time >= tw1, tr1.trace_time < tw2)
    tr2_time_mask = np.logical_and(tr2.trace_time >= tw1, tr2.trace_time < tw2)

    # Instantiate a PH value, which means that no mean separation could be found
    mean_sep = -1
    # Calculate the average centroid position between all traces between h, to place the centre of w
    tr1_centroids = tr1.trace[tr1_time_mask]  # all trace centroids in track 1 between tw1 & tw2
    tr2_centroids = tr2.trace[tr2_time_mask]  # all trace centroids in track 2 between tw1 & tw2
    # mean position of centroids
    mean_centroid = float(np.mean(np.concatenate((tr1_centroids, tr2_centroids))))

    # See which centroids in each track's trace fit within w (+ w % 2 accounts for odd widths, i.e. [) -> [])
    tr1_centroids_idx = \
        np.where(
            np.logical_and(tr1_centroids >= mean_centroid - w // 2, tr1_centroids < mean_centroid + w // 2 + w % 2))[0]
    tr2_centroids_idx = \
        np.where(
            np.logical_and(tr2_centroids >= mean_centroid - w // 2, tr2_centroids < mean_centroid + w // 2 + w % 2))[0]

    if len(tr1_centroids_idx) > 0 and len(tr2_centroids_idx) > 0:
        # Process the refined window if there is a fair balance of traces between each track (max. tol. = 4x)
        ratio = max(len(tr1_centroids_idx), len(tr2_centroids_idx)) / min(len(tr1_centroids_idx),
                                                                          len(tr2_centroids_idx))
        if ratio <= 4:
            tr1_centroids = tr1_centroids[tr1_centroids_idx]  # tr1 trace centroids in the window
            tr2_centroids = tr2_centroids[tr2_centroids_idx]  # tr2 trace centroids in the window

            # Calculate the average separation between the two track's centroids contained within the window
            mean_sep = abs(np.mean(tr1_centroids) - np.mean(tr2_centroids))

    return mean_sep
