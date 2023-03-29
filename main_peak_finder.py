"""
@Created : 05/02/2021
@Edited  : 23/06/2022
@Author  : Alex Poppe
@File    : peak_finder.py
@Software: Pycharm
@Description:
Uses a CAE to reconstruct the resting states of the input BPT scans. Detections are made for outlier data points in the
'difference' scans (input - reconstruction), which are morphologically opened to remove noise. The remaining detections
are processed by a tracking algorithm which places the peak detections into sequential sets called Tracks. These Tracks
are combined with other co-existing Tracks at different wavenumbers into Groups. Groups are then clustered into Events.
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress unnecessary TF loading messages

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import glob
import csv
from analysis.code_tracks.detector import detector, difference_scans
from analysis.code_tracks.tracker import Tracker, Events_Tracker, Event, Groups
from analysis.code_tracks.zipper import zipper
from scipy.sparse.linalg import spsolve
from matplotlib.lines import Line2D
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from skimage.measure import label as connectedness
from nn.models.cae import Autoencoder
from scipy.sparse import csc_matrix, spdiags
from scipy.stats import wasserstein_distance
from main_cae_data import select_scan
from itertools import zip_longest
from pathlib import Path
from utils import overlays, group_converter, append_value, wavenumber_range, hyperparams_setup
from utils import particle_name, real_particles

with tf.device('/device:GPU:0'):
    # [Not Used] Asymmetric Least Squares (ALS) smoothing - baseline estimation
    def als(y, lam=10 ** 3, p=0.01, n_iter=10):
        """ Estimate the baseline of the input spectrum using asymmetric least squares
        *paper: Eilers P., Boelens H., "Baseline Correction with Asymmetric Least Squares Smoothings", 2015

        Args:
            y: Array, The input spectrum/a
            lam: Float, The smoothness of the fit (values generally 10^2 <= lam <= 10^9 - quoted from paper*)
            p: Float, The degree of asymmetric for the fit (values generally 0.001 <= p <= 0.1 - quoted from paper*)
            n_iter: Int, The number of iterations

        Returns:
            z: Array, The output baseline(s)
        """
        L = len(y)
        D = csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for _ in range(n_iter):
            W = spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z


    # [Not Used] Apply ALS to SERS spectra
    def baseline_als(dataset):
        """ Estimate the baseline of each SERS spectra using asymmetric least squares smoothing

        Args:
            dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
        """
        # Plot an example of the ALS baseline subtraction
        for data, label in dataset.as_numpy_iterator():
            if label[1].decode('utf-8')[-4:] != '1045':
                continue

            # Remove redundant dimensions
            data = data.squeeze()

            # Refine nano/pico spectra
            data_nano = data[275]
            data_pico = data[400]

            # Obtain reconstructions for two spectra within the scan: both with and without picocavity peaks
            recon_nano = als(y=data_nano, lam=10 ** 4, p=0.01)
            recon_pico = als(y=data_pico, lam=10 ** 4, p=0.01)

            plt.figure(figsize=(12, 8))
            plt.suptitle('ALS baseline removal from the same scan', fontsize=16)

            plt.subplot(211)
            plt.plot(data_nano, 'k', label='Raw')
            plt.plot(recon_nano, 'b', label='Baseline')
            plt.plot(data_nano - recon_nano, 'r', label='Baseline-subtracted')
            plt.ylabel('Intensity (a.u.)', fontsize=12)
            plt.ylim([-0.1, 0.85])
            leg = plt.legend(title='Resting State Spectrum', loc='upper left')
            plt.setp(leg.get_title(), fontsize=12)

            plt.subplot(212)
            plt.plot(data_pico, 'k', label='Raw')
            plt.plot(recon_pico, 'b', label='Baseline')
            plt.plot(data_pico - recon_pico, 'r', label='Baseline-subtracted')
            plt.xlabel('Wavenumber cm$^{-1}$', fontsize=12)
            plt.ylabel('Intensity (a.u.)', fontsize=12)
            plt.ylim([-0.1, 0.85])
            leg = plt.legend(title='Picocavity Spectrum', loc='upper left')
            plt.setp(leg.get_title(), fontsize=12)

            plt.subplots_adjust(top=0.925)

            plt.show()
            # plt.savefig(f'./analysis/als_vs_cae/baseline_als.png')
            # plt.close()
            exit()

            # np.save('./analysis/als_vs_cae/resting_spectrum.npy', data_nano)
            # np.save('./analysis/als_vs_cae/pico_spectrum.npy', data_pico)
            #
            # np.save('./analysis/als_vs_cae/als_resting_baseline.npy', recon_nano)
            # np.save('./analysis/als_vs_cae/als_resting_subtract.npy', data_nano - recon_nano)
            # np.save('./analysis/als_vs_cae/als_pico_baseline.npy', recon_pico)
            # np.save('./analysis/als_vs_cae/als_pico_subtract.npy', data_pico - recon_pico)
            # exit()

        return


    # Obtain the difference scans
    def obtain_difference_scans(params, dataset):
        """ Extracts Tracks from transient peak detections made on scans, then forms Groups from those scans.

        Args:
            params: Dict, The hyperparameter dictionary
            dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
        """
        # Define optimiser function and instantiate models
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
        model = Autoencoder(params)
        encoder = model.layers[0]
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        decoder = model.layers[1]
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

        # Load the latest checkpoints for the encoder and decoder
        # (NOTE: This assumes the original BPT molecule dataset is the basis for the pre-trained CAE, and that any
        # other molecule datasets constitute fine-tuned versions of that initial model)
        if params['molecule'] == 'BPT':
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder'
        else:
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/decoder'
        encoder_ckpt.restore(tf.train.latest_checkpoint(f'{enc_dir}')).expect_partial()
        decoder_ckpt.restore(tf.train.latest_checkpoint(f'{dec_dir}')).expect_partial()
        if tf.train.latest_checkpoint(f'{enc_dir}') and tf.train.latest_checkpoint(f'{dec_dir}'):
            print(f"Restored encoder from {tf.train.latest_checkpoint(f'{enc_dir}')}")
            print(f"Restored decoder from {tf.train.latest_checkpoint(f'{dec_dir}')}")
        else:
            print("No encoder and/or decoder model(s) found. Exiting...")
            exit()

        for data, label in dataset.as_numpy_iterator():
            # Make the label into a whole string with better formatting for display purposes
            label_str = label.astype('str')[1][-4:]
            print(f'>> Calculating difference spectra for Particle {label_str}')

            # Generate the embeddings and reconstructions
            nbatches = data.shape[0] // params['c_batch_size']
            embed = np.empty((data.shape[0], params['c_embedding_dim'])).astype('float32')  # float32 used in TF
            recon = np.empty((data.shape[0], data.shape[1], 1)).astype('float32')
            for n in range(nbatches):
                # Generate the embedding and reconstruction tensors
                embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = encoder(
                    data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)])
                recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = decoder(
                    embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)])

            # Remove redundant dimensions and calculate the difference scan
            data = np.squeeze(data)
            recon = np.squeeze(recon)
            difference = difference_scans(data, recon)

            # Save the difference scan to file
            if params['molecule'] == 'BPT':
                path = f'./data/difference/{params["c_ver"]}'
            else:
                path = f'./data/BPT_new/difference/{params["c_ver_ft"]}'
            Path(path).mkdir(parents=True, exist_ok=True)
            np.savefig(f'{path}/particle_{label_str}.npy', difference)

        return


    # Produce Groups from each scan, then save the Group feature vectors
    def form_tracks_and_groups(params, dataset, save_figs=False):
        """ Extracts Tracks from transient peak detections made on scans, then forms Groups from those scans.

        Args:
            params: Dict, The hyperparameter dictionary
            dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
            save_figs: Bool, If True: Saves figures to file in various places within the ./analysis/ directory

        Returns:
            Tracks: Class files (.pkl) found inside ./data/*/tracks/c_ver/ (one per scan)
            Groups: Class files (.pkl) found inside ./data/*/groups/c_ver/ (one per scan)
        """
        # Define optimiser function and instantiate models
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
        model = Autoencoder(params)
        encoder = model.layers[0]
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        decoder = model.layers[1]
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

        # Load the latest checkpoints for the encoder and decoder
        # (NOTE: This assumes the original BPT molecule dataset is the basis for the pre-trained CAE, and that any
        # other molecule datasets constitute fine-tuned versions of that initial model)
        if params['molecule'] == 'BPT':
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder'
        else:
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/decoder'
        encoder_ckpt.restore(tf.train.latest_checkpoint(f'{enc_dir}')).expect_partial()
        decoder_ckpt.restore(tf.train.latest_checkpoint(f'{dec_dir}')).expect_partial()
        if tf.train.latest_checkpoint(f'{enc_dir}') and tf.train.latest_checkpoint(f'{dec_dir}'):
            print(f"Restored encoder from {tf.train.latest_checkpoint(f'{enc_dir}')}")
            print(f"Restored decoder from {tf.train.latest_checkpoint(f'{dec_dir}')}")
        else:
            print("No encoder and/or decoder model(s) found. Exiting...")
            exit()

        for data, label in dataset.as_numpy_iterator():
            # #####
            # if label[1].decode('utf-8')[-4:] != '1045':
            #     continue
            # #####

            # Make the label into a whole string with better formatting for display purposes
            label_str = label.astype('str')[1][-4:]
            print('---------------------------')
            print(f'---=== Particle {label_str} ===---')
            print('>> Finding Tracks')

            # Generate the embeddings and reconstructions
            nbatches = data.shape[0] // params['c_batch_size']
            embed = np.empty((data.shape[0], params['c_embedding_dim'])).astype('float32')  # float32 used in TF
            recon = np.empty((data.shape[0], data.shape[1], 1)).astype('float32')
            for n in range(nbatches):
                # Generate the embedding and reconstruction tensors
                embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = encoder(
                    data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)])
                recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = decoder(
                    embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)])

            # #####
            # data = np.squeeze(data)
            # recon = np.squeeze(recon)
            # difference = difference_scans(data, recon)
            #
            # # Refine nano/pico spectra/recons/difference
            # data_nano = data[275]
            # data_pico = data[400]
            # recon_nano = recon[275]
            # recon_pico = recon[400]
            # difference_nano = difference[275]
            # difference_pico = difference[400]
            #
            # plt.figure(figsize=(12, 8))
            # plt.suptitle('CAE resting state removal from the same scan', fontsize=16)
            #
            # plt.subplot(211)
            # plt.plot(data_nano, 'k', label='Raw')
            # plt.plot(recon_nano, 'b', label='Resting state')
            # plt.plot(difference_nano, 'r', label='Transient peaks')
            # plt.ylabel('Intensity (a.u.)', fontsize=12)
            # plt.ylim([-0.1, 0.85])
            # leg = plt.legend(title='Resting State Spectrum', loc='upper left')
            # plt.setp(leg.get_title(), fontsize=12)
            #
            # plt.subplot(212)
            # plt.plot(data_pico, 'k', label='Raw')
            # plt.plot(recon_pico, 'b', label='Resting state')
            # plt.plot(difference_pico, 'r', label='Transient peaks')
            # plt.xlabel('Wavenumber cm$^{-1}$', fontsize=12)
            # plt.ylabel('Intensity (a.u.)', fontsize=12)
            # plt.ylim([-0.1, 0.85])
            # leg = plt.legend(title='Picocavity Spectrum', loc='upper left')
            # plt.setp(leg.get_title(), fontsize=12)
            #
            # plt.subplots_adjust(top=0.925)
            #
            # plt.savefig(f'./analysis/als_vs_cae/baseline_cae.png')
            # plt.close()
            #
            # np.save('./analysis/als_vs_cae/cae_resting_baseline.npy', recon_nano)
            # np.save('./analysis/als_vs_cae/cae_resting_subtract.npy', difference_nano)
            # np.save('./analysis/als_vs_cae/cae_pico_baseline.npy', recon_pico)
            # np.save('./analysis/als_vs_cae/cae_pico_subtract.npy', difference_pico)
            # exit()
            # #####

            # ---=== Peak Finder Algorithm ===---
            # <editor-fold desc="---=== [+] Steps 1. Detect outliers ===---">
            # Instantiate the peak tracker class
            tracker = Tracker(particle_num=label_str)

            # Detect outliers
            detections_scan = detector(params=params, data=data, recon=recon, label_str=label_str,
                                       save_figs=False)  # save_figs=save_figs
            # </editor-fold>

            # <editor-fold desc="---=== [+] Step 2. Form Rudimentary Tracks Through 8-Connectedness ===---">
            # Detect rudimentary tracks through 8-connectivity
            tracks = connectedness(detections_scan, connectivity=2)  # (1-connectivity=cardinals, 2=intercards also)

            # Convert the tracks contained within the scan into individual tracks within the tracker object
            tracker.update(tracks)
            # </editor-fold>

            # <editor-fold desc="---=== [+] Step 3. Recursively Zip Tracks ===---">
            zip_count = 0  # the number of zips performed (i.e. iteration counter)
            num_zips = 1000  # the difference in the number of tracks
            while num_zips > 0:
                # Iterate the counter
                zip_count += 1

                # Store the old track count
                prev_tracks = len(tracker.tracks)

                # Cycle through each track, applying the zipper function where appropriate
                tracker.tracks = zipper(tracks=tracker.tracks, zip_count=zip_count)

                # Calculate the change in number of tracks; repeat the loop if there was a change
                num_zips = prev_tracks - len(tracker.tracks)

            # if save_figs:
            #     # Plot the colour-coded overlaid onto the Scan based on the rudimentary tracks
            #     overlays(params=params, data=data, tracker=tracker, label_str=label_str, overlay_type='zip')
            # </editor-fold>

            # ---=== Group Finder Algorithm ===---
            # Instantiate list and dictionary
            checked_idx = []  # indices placed in this list are ignored for future grouping comparisons
            group_assignments = {}  # indices in this list will be grouped after all comparisons are made

            # <editor-fold desc="---=== [+] Assign Track Groupings ===---">
            # Loop through each possible combination of track comparisons and append the Group assignments to a
            # dictionary (ignoring symmetric comparisons, i.e. if 0 & 1 are compared, no need to compare 1 & 0,
            # same comparisons are also ignored, i.e. 0 & 0 are not compared, and the final track is ignored.
            # There are (n-1)^2/2 comparisons to be made).
            print('>> Finding Groups')

            for idx1, track1 in enumerate(tracker.tracks[:-1]):
                checked_idx.append(idx1)

                # Obtain the first and last time steps of the first track
                tr1_first = min(track1.trace_time)
                tr1_final = max(track1.trace_time)

                for idx2, track2 in enumerate(tracker.tracks):
                    if idx2 in checked_idx:  # skip tracks that have already been checked this iteration
                        continue

                    # Obtain the first and last time steps of the second track
                    tr2_first = min(track2.trace_time)
                    tr2_final = max(track2.trace_time)

                    # Designate the longest overall track as the `focus'
                    # (NOTE: If _first == _final then length is 0, so +1 to all calculations to be correct)
                    length = max(tr1_final - tr1_first + 1, tr2_final - tr2_first + 1)

                    # Count the number of matching time steps shared between the two tracks
                    # (Note: Does not care about bifurcations)
                    # (Note: We use intersections rather than absolute time steps, as otherwise interstitial/
                    # 'out-of-sync' tracks would group together)
                    matches = len(np.intersect1d(track1.trace_time, track2.trace_time))

                    # Assign the two groups to be merged if the threshold is reached or exceeded
                    if matches / length >= 0.7:  # Group threshold = 0.7
                        append_value(group_assignments, idx1, idx2)
            # </editor-fold>

            # <editor-fold desc="---=== [+] Assemble Group Lists ===---">
            # Organise the Group assignments dictionary into lists containing each set of track groupings
            # (e.g. if group_assignments={1: [2, 3], 3: [4], 5: [6]}, then group_lists=[[1, 2, 3, 4], [5, 6]]).
            group_lists = []  # Instantiate the complete list of tracks to-be-grouped
            first_entry = True  # boolean for handling the first list entry differently
            for group in group_assignments.items():
                # Create the values list, containing the key and its associated values
                # (e.g. {0: [1, 2]} -> [0, 1, 2])
                values = [int(group[0])] + group[1]
                if first_entry:
                    # create the first list entry
                    group_lists.append(values)
                    first_entry = False  # this part of the loop is only visited once
                else:
                    already_exists = False  # boolean to skip remaining values if an assignment has occurred
                    for group_list_idx in group_lists:
                        for value in values:
                            if already_exists:
                                break  # i.e. skip all remaining values
                            if value in group_list_idx:
                                # if any value from the current values list is contained within the current
                                # merge list: append
                                for v in values:
                                    group_list_idx.append(v)
                                already_exists = True
                    if not already_exists:
                        # if none of the individual values were contained within any existing merge list:
                        # create a new entry
                        group_lists.append(values)
            # Tidy up the merge list, removing duplicate entries (e.g. [1, 2, 3, 3, 4] --> [1, 2, 3, 4])
            for idx, group_list_idx in enumerate(group_lists):
                group_lists[idx] = list(np.unique(group_list_idx))
            # </editor-fold>

            # <editor-fold desc="---=== [+] Group the Tracks ===---">
            # Assign IDs to each track based on the group they have been assigned to
            idx = 0  # (removes a later warning)
            for idx, group in enumerate(group_lists):
                for g in group:
                    tracker.tracks[g].group_id = idx

            # Find all ungrouped tracks, giving them sequential IDs starting after the final assigned group ID
            group_idxs = np.array([idx for idxs in group_lists for idx in idxs]).flatten()
            all_idxs = np.arange(len(tracker.tracks))
            ungrouped_idxs = [j for j in all_idxs if j not in group_idxs]

            # Instantiate a counter to assigning unique group IDs to the remaining ungrouped tracks
            counter = 1
            for g in ungrouped_idxs:
                if len(group_lists) == 0:
                    idx = 0  # could change this to -1 so that the first group ID will be 0 instead of 1 ...
                    # ... however, it might be useful to leave, as a scan without groups could be identified
                    # easily by seeing that it has no group ID 0
                tracker.tracks[g].group_id = idx + counter  # continue counting from the last assigned group ID
                counter += 1
            # </editor-fold>

            # Save the tracker class object to file
            if params['molecule'] == 'BPT':
                main_path = f'./data/tracks/{params["c_ver"]}'
            else:
                main_path = f'./data/BPT_new/tracks/{params["c_ver_ft"]}'
            Path(main_path).mkdir(parents=True, exist_ok=True)
            with open(f'{main_path}/particle_{label_str}_tracker.pkl', 'wb') as output:
                pickle.dump(tracker, output, pickle.HIGHEST_PROTOCOL)

            if save_figs:
                # Plot the colour-coded Groups overlaid onto the Scan
                overlays(params=params, data=data, recon=recon, tracker=tracker, label_str=label_str,
                         overlay_type='group')

            # Create and save group feature vectors to file
            group_converter(params=params, data=data.squeeze(), tracker=tracker,
                            timesteps=data.shape[0], width=data.shape[1], label_str=label_str)

        return


    # Record the formation times for the first picocavity event of each particle, categorised per sample/power
    def event_formation_stats(params, ph=False, plot_fig=False):
        """ Records the starting time step for the first picocavity event in each Particle (e.g. If the first event
        appears in the second scan in a particle, then the value would be '500 + starting time step', etc.). This
        process is carried out on each sample type and power combination (e.g. 'Aa_50', 'Aa_100', ..., 'APa_200'), and
        written to the same file. This process is then also repeated on the original BPT dataset, in the same manner,
        but saved to a separate file

        Args:
            params: Dict, The hyperparameter dictionary
            ph: Bool, Value used to indicate that no picocavities were found for a particular particle. If True: Uses
                the last possible time step for each particle, if False: Skips that particular particle(s)
            plot_fig: Bool, If True: Plots the figure, if False: Doesn't
        """
        # <editor-fold desc="---=== [+] Form Dictionary of Nested Lists of Particles per Sample/Power ===---">
        # Instantiate dictionary to store the categorised scans
        categories = {}

        # Retrieve the particle numbers and sample_power information to categorise each scan
        particle_idxs, keys = real_particles(params, keep_strings=True)

        if params['molecule'] == 'BPT':
            # Need to sort both outputs from real_particles() such that the keys (i.e. BPT_447, BPT_564, and BPT_709)
            # are neighbouring each other. This is a criterion of the below loop to form the categories correctly
            keys = np.array(keys)
            idx = np.argsort(keys)
            keys = keys[idx]
            particle_idxs = particle_idxs[idx]

        # Define placeholder variables
        prev_idx = -1  # PH
        prev_key = -1  # PH

        # Group scans together based on particle numbers, in global particle number order*
        # *e.g. [28, 28, 3, 3, ..., 3, 12, 12, ..., 12, ...], where numbers are IDs (based on string-ordered numbers)
        for i, (idx, key) in enumerate(zip(particle_idxs, keys)):
            # Reform particle_num (e.g. 0 -> 'Particle_0000' & 256 -> 'Particle_0256')
            i = str(i)
            particle_num = '0' * (4 - len(i)) + i

            # If the key is new...
            if key != prev_key:
                # ...form a new nested list as the first value of the new key
                categories[key] = [[f'particle_{particle_num}']]
            # Else if the key is not new then check if it is the same particle as last time (assumes they neighbour)...
            else:
                # ...if it is a new particle...
                if idx != prev_idx:
                    # ...form the next nested list in the current key
                    categories[key].extend([[f'particle_{particle_num}']])
                # ...if it is not a new particle...
                else:
                    # ...place the particle_num inside the current nested list
                    categories[key][-1].append(f'particle_{particle_num}')

            # Lastly, update the variables
            prev_key = key
            prev_idx = idx
        # </editor-fold>

        # <editor-fold desc="---=== [+] Form Dictionary of The Earliest Formation Time Steps per Sample/Power ===---">
        # Instantiate new dictionary that will store the recorded values (the keys will match the previous one)
        formations = {}

        # Define the filepath for the Groups, as well as the size of each scan in time steps
        if params['molecule'] == 'BPT':
            vector_paths = f'./data/groups/{params["c_ver"]}'
            timesteps = 1000
        else:
            vector_paths = f'./data/BPT_new/groups/{params["c_ver_ft"]}'
            timesteps = 500
        # Loop through each 'sample_power'
        for key, particles in categories.items():
            # Loop through each particle
            for particle in particles:
                # Instantiate the formation time as the maximum possible time step for the current particle
                if ph:
                    form = timesteps * len(particle)
                else:
                    form = -1

                # Instantiate a boolean to break out of the current particle once a start time has been found (as later
                # scans are inherently later starting than earlier ones)
                found = False

                # Loop through each scan within the current particle
                for i, scan in enumerate(particle):
                    # Attempt to load in the Groups for the current scan...
                    try:
                        with open(f'{vector_paths}/{scan}.pkl', 'rb') as r:
                            groups = pickle.load(r)

                            # Loop through each Group, ignoring those with only one Track
                            for vector in groups.vectors:
                                if len(vector['mean_centroids']) == 1:  # remove all single-peak Groups
                                    continue

                                # Check if the start time for each Group is earlier than the current 'form' value
                                start = timesteps * i + vector['start']  # add time steps from all previous scans
                                if start < form or form == -1:
                                    form = start

                        # Skip the next scan(s) if a time step has been found in the current scan
                        if found:
                            break

                    # ...if the scan does not exist (i.e. it is a False scan, hence no Groups are saved, then it is
                    # skipped. This will automatically add on 500 or 1000 [based on the 'molecule'] due to the 'i' var)
                    except FileNotFoundError:
                        pass

                # Record value in final dictionary (we do not record which particle it came from)
                if form != -1:  # (value is not recorded if no picocavities were found for the current particle)
                    append_value(formations, key, form)
        # </editor-fold>

        # <editor-fold desc="---=== [+] Write Results to File ===---">
        # Define main filepath
        if params['molecule'] == 'BPT':
            fpath = f'./analysis/{params["c_ver"]}'
            suffix1 = '_orig'
        else:
            fpath = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}'
            suffix1 = '_new'

        # Define filename suffix
        if ph:
            suffix2 = '_ph'
        else:
            suffix2 = ''

        # Write the results to file
        with open(f'{fpath}/picocavity_formations{suffix1}{suffix2}.csv', 'w', newline='') as w:
            writer = csv.writer(w, delimiter=' ')  # delimiter = ' ' for windows, delimiter = '\t' for linux

            # Define custom sort-order for the sample_power keys (i.e. Aa -> APa -> Ap ... 50 -> 200)
            # (NOTE: This is done because regular np.sort() does 100/150/200/50 for the powers as they are strings)
            # (NOTE: A similar thing is done for the original BPT value [see below*])
            keys = np.array(list(formations.keys()))
            if params['molecule'] == 'BPT':
                # *Technically this would sort the way I want it to, but I chose to remain consistent with the other one
                arg_keys = np.array([np.argwhere(keys == 'BPT_447').squeeze(),
                                     np.argwhere(keys == 'BPT_564').squeeze(),
                                     np.argwhere(keys == 'BPT_709').squeeze()])
            else:
                arg_keys = np.array([np.argwhere(keys == 'Aa_50').squeeze(),
                                     np.argwhere(keys == 'Aa_100').squeeze(),
                                     np.argwhere(keys == 'Aa_150').squeeze(),
                                     np.argwhere(keys == 'Aa_200').squeeze(),
                                     np.argwhere(keys == 'APa_50').squeeze(),
                                     np.argwhere(keys == 'APa_100').squeeze(),
                                     np.argwhere(keys == 'APa_150').squeeze(),
                                     np.argwhere(keys == 'APa_200').squeeze(),
                                     np.argwhere(keys == 'Ap_50').squeeze(),
                                     np.argwhere(keys == 'Ap_100').squeeze(),
                                     np.argwhere(keys == 'Ap_200').squeeze()])

            # Write the keys as the header
            writer.writerow(np.array(list(formations.keys()))[arg_keys])

            # Loop through each row up to the key with the most values recorded (keys with fewer values have rows
            # beyond their last replaced with spaces)
            for row in zip_longest(*formations.values(), fillvalue=' '):
                # Sort the current row to correctly align with the sorted keys
                row = np.array(row)[arg_keys]

                # Write the row to file
                writer.writerow(row)
        # </editor-fold>

        if plot_fig:
            # <editor-fold desc="---=== [+] Plot Results ===---">
            plt.figure(figsize=(12, 7))
            for key, val in formations.items():
                # Colour the points based on the sample setup (i.e. power independent)
                # Also define the power (slightly shifted in the x-axis for visual aid)
                if key.split('_')[0] == 'Aa':
                    c = 'r'
                    power = int(key.split('_')[-1]) - 2
                elif key.split('_')[0] == 'Ap':
                    c = 'g'
                    power = int(key.split('_')[-1])
                elif key.split('_')[0] == 'APa':
                    c = 'b'
                    power = int(key.split('_')[-1]) + 2
                else:  # i.e. key.split('_')[0] == 'BPT'
                    c = 'k'
                    power = int(key.split('_')[-1])

                # plt.scatter([power] * len(val), val, color=c)
                plt.errorbar(power, np.mean(val), yerr=np.std(val), fmt='o', color=c)

            plt.ylabel('Formation Time (time steps)', fontsize=12)
            plt.xlabel('Power (mW)', fontsize=12)
            if params['molecule'] == 'BPT':
                plt.xticks(ticks=[447, 564, 709])
            else:
                plt.xticks(ticks=[50, 100, 150, 200])
                custom_lines = [Line2D([0], [0], color='w', marker='o', markersize=10, markerfacecolor='r'),
                                Line2D([0], [0], color='w', marker='o', markersize=10, markerfacecolor='g'),
                                Line2D([0], [0], color='w', marker='o', markersize=10, markerfacecolor='b')]
                plt.legend(custom_lines, ['Au NP on Au mirror',
                                          'Au NP on 1ML Pd mirror',
                                          'Au_Pd NP on Au mirror'])
            plt.show()
            # </editor-fold>

        return


    # Take in input feature vectors, and cluster them using a 'difference of integrals' thresholding method
    def cluster_groups_into_events(params, dataset, remove_singles=True, prefix=None, optimise_k=False,
                                   avg_group=False):
        """ Attempts to cluster Groups into Events by comparing the integral of the squared differences of their
        feature vectors to the summation of their integrals. If the former value is under some fraction of the latter,
        the Groups are assigned the same event ID.

        Args:
            params: Dict, The hyperparameter dictionary
            dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
            remove_singles: Bool, If True: Keeps all Groups, if False: Removes any Groups only containing a single
                transient peak (default to avoid Events forming from potentially erroneous data)
            prefix: Str or None: If Str: saves the Events object with the Str provided as a prefix, If None: No Prefix
            optimise_k: Bool, If True: Calculates optimal number of clusters using Silhouette score (between 6 and 12
                clusters) and uses that amount, If False: Uses user-specified c_nclusters value from chosen cfg
            avg_group: Bool, If True: Uses the Groups formed from averaging Groups found via initial clustering, If
                False: Uses the Groups obtained from the initial Group formation stage

        Returns:
            Events: Class file (.pkl) named c_nclusters (or optimised k-value) - found inside ./data/events/c_ver/name/
        """
        # Define parameter used to reference the correct scans and saving to correct file destinations
        if prefix is None:
            prfx = ''
        else:
            prfx = f'{prefix}_'

            if prefix == 'Aa':
                ref = 'Au NP on Au mirror'
            elif prefix == 'Ap':
                ref = 'Au NP on 1ML Pd mirror'
            else:
                ref = 'Au_Pd NP on Au mirror'

        # Load in the selected Group file paths
        if params['particles'] is not None:  # specific particles
            vector_paths = []
            for particle in params['particles']:
                if params['molecule'] == 'BPT':
                    if avg_group:  # load in the averaged Groups
                        try:
                            vector_paths.append(
                                glob.glob(f'./data/groups/{params["c_ver"]}/{prfx}averaged/{particle.lower()}.pkl')[0])
                        except IndexError:  # Skip particle if not saved (due to only having single-peak Groups)
                            continue
                    else:  # load in the initial Groups
                        vector_paths.append(glob.glob(f'./data/groups/{params["c_ver"]}/{particle.lower()}.pkl')[0])
                else:
                    if avg_group:  # load in the averaged Groups
                        try:
                            vector_paths.append(glob.glob(
                                f'./data/BPT_new/groups/{params["c_ver_ft"]}/{prfx}averaged/{particle.lower()}.pkl')[0])
                        except IndexError:  # Skip particle if not saved (due to only having single-peak Groups)
                            continue
                    else:  # load in the initial Groups
                        vector_paths.append(
                            glob.glob(f'./data/BPT_new/groups/{params["c_ver_ft"]}/{particle.lower()}.pkl')[0])
        else:  # all particles
            if params['molecule'] == 'BPT':
                if avg_group:  # load in the averaged Groups
                    vector_paths = glob.glob(f'./data/groups/{params["c_ver"]}/{prfx}averaged/*.pkl')
                else:  # load in the initial Groups
                    vector_paths = glob.glob(f'./data/groups/{params["c_ver"]}/*.pkl')
            else:
                if avg_group:  # load in the averaged Groups
                    vector_paths = glob.glob(f'./data/BPT_new/groups/{params["c_ver_ft"]}/{prfx}averaged/*.pkl')
                else:  # load in the initial Groups
                    vector_paths = glob.glob(f'./data/BPT_new/groups/{params["c_ver_ft"]}/*.pkl')

        if params['molecule'] != 'BPT':  # only select Groups from those that belong to the current experiment
            vector_paths_temp = []
            for vector_path in vector_paths:
                if particle_name(params, vector_path.split('/')[-1].split('.')[0].capitalize())[0] == ref:
                    vector_paths_temp.append(vector_path)
            vector_paths = vector_paths_temp

        # Create a dictionary with keys as group IDs and values as the group dictionaries
        group_vectors = {}
        # Loop through each Group filepath, which direct to Groups from a single scan
        for vector_path in vector_paths:
            # Load in the Groups object
            with open(vector_path, 'rb') as r:
                groups = pickle.load(r)
                # Loop through each Group
                for vector in groups.vectors:
                    if remove_singles and len(vector['mean_centroids']) == 1:  # remove all single-peak Groups
                        continue
                    # Append the current Groups onto the list of all selected Groups
                    append_value(group_vectors, f'{groups.particle_num}_{vector["id"]}', vector, append_list=False)

        # Instantiate a list of all spectra from the group feature vectors
        spectra = []
        for n, group in enumerate(group_vectors.values()):
            spectra.append(group['spectrum'])
        spectra = np.array(spectra)  # convert to array

        print('>> Calculating the Normalised Mean Difference Spectra (Clustering Requirement)')
        # <editor-fold desc="---=== [+] Add normalised mean difference spectra to group feature vectors ===---">
        # Instantiate lists to contain all scans and labels
        datas = []
        labels = []
        for data, label in dataset.as_numpy_iterator():
            datas.append(data.squeeze())
            labels.append(label[1].decode('utf-8')[-4:])
        # Convert to arrays
        datas = np.array(datas)
        labels = np.array(labels)

        # Define optimiser function and instantiate models
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
        model = Autoencoder(params)
        encoder = model.layers[0]
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        decoder = model.layers[1]
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

        # Load the latest checkpoints for the encoder and decoder
        if params['molecule'] == 'BPT':
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder'
        else:
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/decoder'
        encoder_ckpt.restore(tf.train.latest_checkpoint(f'{enc_dir}')).expect_partial()
        decoder_ckpt.restore(tf.train.latest_checkpoint(f'{dec_dir}')).expect_partial()
        if tf.train.latest_checkpoint(f'{enc_dir}') and tf.train.latest_checkpoint(f'{dec_dir}'):
            print(f"Restored encoder from {tf.train.latest_checkpoint(f'{enc_dir}')}")
            print(f"Restored decoder from {tf.train.latest_checkpoint(f'{dec_dir}')}")
        else:
            print("No encoder and/or decoder model(s) found. Exiting...")
            exit()

        # Instantiate a list to contain all mean Group difference spectra
        group_diff = []

        # Cycle through each group vector
        for vector in group_vectors.values():
            # Find the appropriate scan
            data = datas[np.argwhere(labels == vector['particle_num']).squeeze()]
            data = np.expand_dims(data, axis=-1)  # expand dimensions to fit the CAE input shape

            # Generate the embeddings and reconstructions
            nbatches = data.shape[0] // params['c_batch_size']
            embed = np.empty((data.shape[0], params['c_embedding_dim'])).astype('float32')
            recon = np.empty((data.shape[0], data.shape[1], 1)).astype('float32')
            for n in range(nbatches):
                # Generate the embedding and reconstruction tensors
                embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = encoder(
                    data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)])
                recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = decoder(
                    embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)])
            data = data.squeeze()
            recon = recon.squeeze()

            # Calculate the difference scan
            difference = difference_scans(data, recon)

            # Assign mean difference vector to current dictionary key (i.e. Group)
            if isinstance(vector['start'], list):
                diff = []
                for start, end in zip(vector['start'], vector['end']):
                    diff.append(difference[start:end + 1])
                vector['mean_difference'] = np.mean(np.concatenate(diff, axis=0), axis=0)
            else:
                vector['mean_difference'] = np.mean(difference[vector['start']:vector['end'] + 1], axis=0)

            # Assemble all mean group difference spectra into an array
            group_diff.append(vector['mean_difference'])

        # Normalise the mean group difference spectra (required for Wasserstein distance)
        group_diff = np.array(group_diff)  # convert to array beforehand
        group_diff /= np.sum(group_diff, axis=1, keepdims=True)

        # Now, loop through all group feature vectors, and reassign mean group difference spectra post-normalisation
        # (Note: The Dict and Array are in the same order, therefore no need to check)
        for vector, norm in zip(group_vectors.values(), group_diff):
            vector['mean_difference'] = norm
        # </editor-fold>

        events_tracker = Events_Tracker()

        # ---=== Cluster Groups into Events ===---
        print('>> Calculating Wasserstein Distance Between Groups')
        # <editor-fold desc="---=== [+] Assign Event Groupings ===---">
        # Instantiate the distance matrix used to cluster groups into events
        ratio_matrix = np.zeros((spectra.shape[0], spectra.shape[0]))

        # Instantiate list and dictionary
        for idx1, group1 in enumerate(list(group_vectors.keys())[:-1]):
            # Retrieve the first mean difference spectrum from the group dictionary
            spectrum1 = group_vectors[group1]['mean_difference']

            # Set the current main diagonal of the ratio matrix to 1
            ratio_matrix[idx1, idx1] = 0

            for idx2m, group2 in enumerate(list(group_vectors.keys())[idx1 + 1:]):  # upper triangle only
                idx2 = idx2m + idx1 + 1  # adjust the index to account for the lower triangle & main diagonal skip

                # Retrieve the second mean difference spectrum from the group dictionary
                spectrum2 = group_vectors[group2]['mean_difference']

                # Calculate Earth Mover's distance and insert it into the appropriate indices of the affinity matrix
                dist = wasserstein_distance(spectrum1, spectrum2)
                ratio_matrix[idx1, idx2] = dist  # upper triangle
                ratio_matrix[idx2, idx1] = dist  # lower triangle

        # Store the distance matrix within the main events class object
        events_tracker.distance_matrix = ratio_matrix

        # Scale the distance matrix values linearly between the range [0, 1] - this makes the RBF kernel fit affinity
        # values between [0, 1] (based on the chosen delta value below)
        ratio_matrix = (ratio_matrix - np.min(ratio_matrix)) / (np.max(ratio_matrix) - np.min(ratio_matrix))

        # Now convert distance matrix into an affinity matrix using a Gaussian kernel
        # (NOTE: This delta value was chosen to give a mean ~= 0.5 and a std ~= 0.35. It would be useful to
        # automatically scale the affinity values between a sensible range)
        delta = 0.275
        ratio_matrix = np.exp(-ratio_matrix ** 2 / (2 * delta ** 2))

        best_k = params['c_nclusters']  # define parameter to specify #clusters (avoids warning at end of function)
        if optimise_k:
            print('>> Optimising the number of Events/clusters')

            # Set the main diagonal of the affinity matrix to zero (requirement of Silhouette score)
            distance_zero = np.copy(events_tracker.distance_matrix)
            np.fill_diagonal(distance_zero, 0)

            # Cycle through 2 to 30 clusters*, calculate the Silhouette score for each, and append the results to dict
            # (*NOTE: I have chosen 6 to 21 here, as I can see at least 6 unique picocavity events through a quick
            # inspection, and the silhouette score consistently decreases beyond ~15 clusters, so stretching the test
            # beyond 21 provides no benefit with increased computation time)
            best_score = 0
            best_std = 0
            best_k = 0
            for k in range(6, 21):  # [2, 31)
                print(f'> Trying {k} clusters')

                # Instantiate list to contain the Silhouette scores for the current #clusters
                s_scores = []

                # Iterate each clustering attempt 10 times to get an average of the performance
                for i in range(10):
                    cluster_labels = SpectralClustering(n_clusters=k,
                                                        assign_labels='kmeans',
                                                        affinity='precomputed',
                                                        n_jobs=2 * k).fit_predict(ratio_matrix)

                    # Calculate Silhouette score for all events, and append to dictionary
                    s_scores.append(silhouette_score(X=distance_zero, labels=cluster_labels, metric='precomputed'))

                mean_score = np.mean(s_scores)
                std_score = np.std(s_scores)
                if mean_score >= best_score:
                    best_score = mean_score
                    best_std = std_score
                    best_k = k
                print(f'\tSilhouette score = {mean_score:.4f} +- {std_score:.4f}')
            print(f'---=== Optimal number of clusters = {best_k} (score = {best_score:.4f} +- {best_std:.4f}) ===---')

            # Repeat spectral clustering until at least the mean Silhouette score of the best clustering results is met
            # (Note: It would be a good idea to optimise this process by holding onto the clustering results from the
            # [currently] best silhouette score until all scores have been checked, which would avoid having to
            # reproduce the same clusters repeatedly until your expected clustering result is met [and potentially stop
            # checking early if there is a 'downwards' trend in those scores])
            print('>> Recreating optimal clustering result')
            score = -1
            num_attempts = 0
            while score < best_score:
                num_attempts += 1
                cluster_labels = SpectralClustering(n_clusters=best_k,
                                                    assign_labels='kmeans',
                                                    affinity='precomputed',
                                                    n_jobs=2 * best_k).fit_predict(ratio_matrix)
                score = silhouette_score(X=distance_zero, labels=cluster_labels, metric='precomputed')
            print(f'\t({num_attempts} attempts to recreate)')

        else:
            # Now do spectral clustering
            print('>> Clustering Groups to Create Events')
            cluster_labels = SpectralClustering(n_clusters=params['c_nclusters'],
                                                assign_labels='kmeans',
                                                affinity='precomputed',
                                                n_jobs=2 * params['c_nclusters']).fit_predict(ratio_matrix)

        # Store the affinity matrix and associated cluster labels within the main events class object
        events_tracker.affinity_matrix = ratio_matrix
        events_tracker.labels = cluster_labels
        # </editor-fold>

        print('>> Forming Events Class')
        # <editor-fold desc="---=== [+] Assemble Event Lists ===---">
        # Create the list of groups to be combined into events
        uniques = np.unique(cluster_labels)
        event_lists = []
        all_keys = np.array(list(group_vectors.keys()))  # names of all groups (e.g. 0506_0, 0506_1, 1045_0, ...)
        for unique in uniques:
            # Find indices of all groups clustered into current Event by taking Group names indexed from cluster_labels
            groups = all_keys[np.argwhere(cluster_labels == unique).squeeze()].tolist()
            if isinstance(groups, str):
                groups = [groups]
            event_lists.append(groups)
        # </editor-fold>

        # Update the num_events variable within the Events tracker class
        events_tracker.num_events = len(event_lists)

        for i, e in enumerate(event_lists):
            # Create a new class for the current Event and update it with the required parameters/variables
            event = Event(event_id=i)

            # Loop through each group feature vector within this event, collecting necessary data
            particle_nums = []  # instantiate list of all particle numbers
            min_duration = 1000  # instantiate variable for shortest-lived Group
            max_duration = 0  # instantiate variable for longest-lived Group
            mean_vectors = []  # instantiate list of mean difference spectra
            for ei in e:  # ei = '0506_0' (for example)
                # Append each group feature vector onto list of event vectors
                event.vectors.append(group_vectors[ei])

                # Append particle numbers onto the *unique* list
                if group_vectors[ei]['particle_num'] not in particle_nums:
                    particle_nums.append(group_vectors[ei]['particle_num'])

                # Update the longest duration within this event...
                if group_vectors[ei]['duration'] > max_duration:
                    max_duration = group_vectors[ei]['duration']

                # ...do the same for the shortest
                if group_vectors[ei]['duration'] < min_duration:
                    min_duration = group_vectors[ei]['duration']

                # Append mean difference spectrum to list
                mean_vectors.append(group_vectors[ei]['mean_difference'])

            # Store the (sorted) list of particles for the event
            event.particles.extend(particle_nums)
            event.particles = list(np.sort(event.particles))

            # Update the min/max durations for groups in this current event
            event.duration_stats = [min_duration, max_duration]

            # <editor-fold desc="---=== [+] Intra-Event Strength ===---">
            # Instantiate a list to contain intra-Event strengths
            strengths = []  # instantiate list of intra-event strengths
            for idx1, group1 in enumerate(mean_vectors[:-1]):
                for group2 in mean_vectors[idx1 + 1:]:  # upper triangle only
                    # Calculate Earth Mover's distance
                    dist = wasserstein_distance(group1, group2)

                    # Now convert distance into an affinity using the previous RBF kernel, then append to list
                    strengths.append(np.exp(-dist ** 2 / 2 * delta ** 2))

            event.intra_strength = [np.mean(strengths), np.std(strengths)]  # append to current event class object
            # </editor-fold>

            # Calculate the mean group feature vector for this event
            event.mean_difference = np.mean(mean_vectors, axis=0)

            # Append the finished Event to the list of Events in the Events_Tracker class
            events_tracker.events.append(event)

        # Save the tracker class object to file
        if params['molecule'] == 'BPT':
            main_path = f'./data/events/{params["c_ver"]}/{params["name"]}'
        else:
            main_path = f'./data/BPT_new/events/{params["c_ver_ft"]}/{params["name"]}'
        Path(main_path).mkdir(parents=True, exist_ok=True)

        if avg_group:
            with open(f'{main_path}/{prfx}{best_k}clusters_avg.pkl', 'wb') as output:
                pickle.dump(events_tracker, output, pickle.HIGHEST_PROTOCOL)

        else:
            with open(f'{main_path}/{prfx}{best_k}clusters.pkl', 'wb') as output:
                pickle.dump(events_tracker, output, pickle.HIGHEST_PROTOCOL)

        return


    # Remove bias from 'flickering' Groups (i.e. one that are constantly formed/destroyed) by averaging into one Group
    def average_groups(params, dataset, prefix=None, optimise_k=False):
        """ Reviews the Groups clustered into the same Events from the previous cluster_groups_into_events() function.
        Groups within the same particle/scan are averaged if they are within 100 time steps of each other. The averaged
        Groups are then saved in a new folder, and the previous function is called again to create new clusters.

        The purpose of this function is to attempt to remove any bias incurred from picocavity events that 'flicker'
        on and off multiple times in the clustering process, which would likely appear as multiple Groups, thus forming
        a highly populated Event for what was in reality a single occurrence of that picocavity.

        Args:
            params: Dict, The hyperparameter dictionary
            dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
            prefix: Str or None: If Str: saves the Events object with the Str provided as a prefix, If None: No Prefix
            optimise_k: Bool, If True: Calculates optimal number of clusters using Silhouette score (between 6 and 12
                clusters) and uses that amount, If False: Uses user-specified c_nclusters value from chosen cfg

        Returns:
            Events: Class file (.pkl) named c_nclusters (or optimised k-value) - found inside ./data/events/c_ver/name/
        """
        # Define the threshold for the maximum separation in units of time steps that two Groups can have to be
        # considered a part of the same physical picocavity Event
        thresh = 100

        # <editor-fold desc="---=== [+] Load/Produce Input/Recon Scans and Associated Labels ===---">
        # Define optimiser function and instantiate models
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
        model = Autoencoder(params)
        encoder = model.layers[0]
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        decoder = model.layers[1]
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

        # Load the latest checkpoints for the encoder and decoder
        # (NOTE: This assumes the original BPT molecule dataset is the basis for the pre-trained CAE, and that any
        # other molecule datasets constitute fine-tuned versions of that initial model)
        if params['molecule'] == 'BPT':
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder'
        else:
            enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/encoder'
            dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/decoder'
        encoder_ckpt.restore(tf.train.latest_checkpoint(f'{enc_dir}')).expect_partial()
        decoder_ckpt.restore(tf.train.latest_checkpoint(f'{dec_dir}')).expect_partial()
        if tf.train.latest_checkpoint(f'{enc_dir}') and tf.train.latest_checkpoint(f'{dec_dir}'):
            # print(f"Restored encoder from {tf.train.latest_checkpoint(f'{enc_dir}')}")
            # print(f"Restored decoder from {tf.train.latest_checkpoint(f'{dec_dir}')}")
            pass
        else:
            print("No encoder and/or decoder model(s) found. Exiting...")
            exit()

        # Instantiate lists to contain all input and recon scans, as well as labels
        datas = []
        labels = []
        recons = []
        for data, label in dataset.as_numpy_iterator():
            datas.append(data.squeeze())
            labels.append(label[1].decode('utf-8')[-4:])

            # Generate the embeddings and reconstructions
            nbatches = data.shape[0] // params['c_batch_size']
            recon = np.empty((data.shape[0], data.shape[1], 1)).astype('float32')
            for n in range(nbatches):
                # Generate the embedding and reconstruction tensors
                recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = decoder(
                    encoder(data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)]))
            recons.append(recon)
        # Convert to arrays
        datas = np.array(datas)
        recons = np.array(recons).squeeze()
        datas_labels = np.array(labels)

        # Calculate difference scans
        differences = []
        for data, recon in zip(datas, recons):
            differences.append(difference_scans(data, recon))
        differences = np.array(differences)
        # </editor-fold>

        if prefix is None:
            prfx = ''
        else:
            prfx = f'{prefix}_'

        if params['molecule'] == 'BPT':
            event_path = f'./data/events/{params["c_ver"]}/{params["name"]}/'
            k = int(os.listdir(event_path)[0].split('c')[0])  # assumes there is only a single Events object file
            event_path += f'{prfx}{k}clusters.pkl'
        else:
            event_path = f'./data/BPT_new/events/{params["c_ver_ft"]}/{params["name"]}/'

            files = os.listdir(event_path)  # assumes there is only a single Events object file for each experiment
            k = -1
            for file in files:
                if prfx[:-1] in file:
                    k = int(file.split('_')[1].split('c')[0])
                    break
            event_path += f'{prfx}{k}clusters.pkl'

        # Instantiate a Group dictionary to store Groups that are needed to be merged, and a counter to store
        # an arbitrary ID
        groups = {}
        counter = 0

        # Calculate the number of pixels representing 20 wavenumbers (rounding up) - used in future peak comparisons
        wn_20 = np.ceil(20 / np.mean(np.diff(wavenumber_range(default=True))))  # ~8 px

        # View Events
        with open(event_path, 'rb') as loader:
            event = pickle.load(loader)

            # Cycle through each Group vector
            for e in event.events:
                # Instantiate dictionary with one key per particle within the current Event, and where each value will
                # be the Groups belonging to that particle
                parts = {}
                for vector in e.vectors:
                    append_value(parts, vector['particle_num'], vector)

                for key, vals in parts.items():
                    times = []
                    for val in vals:
                        times.append([val['start'], val['end']])
                    times = np.array(times)

                    # Sort times (and original list of Groups) based on starting time (in ascending order)
                    srt = np.argsort(times[:, 0], axis=0)
                    times = times[srt]
                    vals = list(np.array(vals)[srt])

                    # Flatten the times array, and remove the first start and last end times
                    idx = times.flatten()[1:-1].reshape((-1, 2))

                    # Calculate the separation between each Group (end of earlier start vs start of later start)
                    idx = np.diff(idx, axis=1).squeeze()

                    # Find the partitions where each Group should be combined
                    # (e.g. if there are 3 groups, and partition array has [2], then Groups 0 & 1 should be averaged
                    # separately to Group 2, because Group 2 starts more than 100 time steps away than end of Group 1)
                    partitions = np.argwhere(idx > thresh).squeeze() + 1  # (+1 to make future indexing simpler)
                    if isinstance(partitions, np.int64):  # convert to array if only one partition is found
                        partitions = np.array(partitions)
                    partitions = np.append(np.insert(partitions, 0, 0), len(times))  # add [0, ..] & [.. , len(times)]

                    # Place each to-be-merged Group within the same key in the groups dictionary
                    for i in range(len(partitions) - 1):
                        append_value(groups, counter, vals[partitions[i]:partitions[i + 1]], append_list=False)
                        counter += 1

            for key, val in groups.items():
                # Define new group to represent the average of all current groups
                new_val = {'id': val[0]['id'],
                           'particle_num': val[0]['particle_num']}

                # Cycle through each Group, updating relevant features in the new Group dictionary
                centroids = []
                starts = []
                ends = []
                for v in val:
                    starts.append(v['start'])
                    ends.append(v['end'])
                    centroids.append(v['mean_centroids'])

                # Assign start/end steps to the new Group dictionary
                new_val['start'] = starts
                new_val['end'] = ends

                # <editor-fold desc="---=== [+] Pair Matching Centroids ===---">
                # Separate first Group to use as the main comparison list
                c0 = centroids[0]

                # Cycle through each Group (except the control Group, c0)
                for i, ci in enumerate(centroids[1:]):
                    # Indices of ci that will be included in c0 (numerical order)
                    adds = []  # e.g. [0]
                    # Pairs of indices indicating value from ci that will be averaged with the corresponding c0 value
                    means = []  # e.g. [[1, 5], ...], where sublists = [idx, value]

                    # Cycle through each centroid in the current Group
                    for cii in ci:
                        # Calculate the closest difference between current centroid and all centroids in control Group
                        mn = np.min(np.abs(c0 - cii))

                        # Find idx of the closest difference
                        argmn = np.argmin(np.abs(c0 - cii))

                        # If the closest difference is outside the tolerance...
                        if mn > wn_20:  # ...store the current centroid in the 'adds' list
                            adds.append(cii)

                        # If the closest difference is within (or equal to) the tolerance
                        else:  # ...store the current centroid in the 'means' list
                            means.append([argmn, cii])

                    # Average the designed peaks between the control and current Groups
                    for mean in means:
                        c0[mean[0]] = np.round((c0[mean[0]] + mean[1]) / 2).astype(int)
                    # Append the additional peaks from the 'adds' list
                    c0 = np.sort(np.append(c0, adds)).astype(int)

                # Assign the mean_centroids key to the new Group
                new_val['mean_centroids'] = c0
                # </editor-fold>

                # Calculate the new Group duration (latest end - earliest start + 1)
                new_val['duration'] = max(new_val['end']) - min(new_val['start']) + 1

                # Calculate the new mean input spectrum and difference spectrum
                idx = np.argwhere(datas_labels == new_val['particle_num']).squeeze()  # location of particle in array

                spectrum = []
                difference = []
                for start, end in zip(new_val['start'], new_val['end']):
                    spectrum.append(datas[idx, start:end + 1])
                    difference.append(differences[idx, start:end + 1])
                new_val['spectrum'] = np.mean(np.concatenate(spectrum, axis=0), axis=0)
                new_val['difference'] = np.mean(np.concatenate(difference, axis=0), axis=0)

                # Replace the list of Groups inside the current key to the new averaged Group
                groups[key] = new_val

        # Instantiate a dictionary for new Groups per scan (i.e. keys are particles/scans, which contain all the Groups
        # that were created/remain from the above process, regardless of the initial Event(s) that they belonged to)
        new_groups = {}
        for key, val in groups.items():
            append_value(new_groups, val['particle_num'], val)

        # Create a folder where the new averaged Group feature vectors will be stored as class objects, per particle,
        # separated by experiment (e.g. Aa/APa/Ap)
        if params['molecule'] == 'BPT':
            path = f'./data/groups/{params["c_ver"]}/{prfx}averaged/'
        else:
            path = f'./data/BPT_new/groups/{params["c_ver_ft"]}/{prfx}averaged/'
        Path(path).mkdir(parents=True, exist_ok=True)

        # Next, loop through the new dictionary, creating a new Group object class for each particle, then save each
        # to file under their respective experiment dictionary
        for key, val in new_groups.items():
            groups = Groups(particle_num=key)

            # Append the current Group feature vector to the list of vectors within the current Group object
            for vector in val:
                groups.vectors.append(vector)

            with open(f'{path}particle_{key}.pkl', 'wb') as w:
                pickle.dump(groups, w, pickle.HIGHEST_PROTOCOL)

        # Finally, re-run the cluster_groups_into_events() again using the new Groups
        cluster_groups_into_events(params=hyperparams, dataset=scans, remove_singles=True, prefix=name,
                                   optimise_k=optimise_k, avg_group=True)

        return


    # View random examples of events
    def events_summary(params, dataset, prefix=None, save_figs=False, avg_group=False):
        """ Saves the Events to an object file. This contains the mean raw spectrum of all spectra in each Event

        Args:
            params: Int, The hyperparameter dictionary
            dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
            prefix: Str or None: If Str: saves the Events object with the Str provided as a prefix, If None: No Prefix
            save_figs: Bool, If True: Saves figures
            avg_group: Bool, If True: Uses the Groups formed from averaging Groups found via initial clustering, If
                False: Uses the Groups obtained from the initial Group formation stage
        """
        # Instantiate lists to contain all scans and labels (normalised)
        datas = []
        labels = []
        for data, label in dataset.as_numpy_iterator():
            datas.append(data.squeeze())
            labels.append(label[1].decode('utf-8')[-4:])
        # Convert to arrays
        datas = np.array(datas)
        datas_labels = np.array(labels)

        # Instantiate a new events class, used to store the top-k events from the global events .pkl file
        # (NOTE: A new Events object is created and saved separately (in the /analysis directory) to the original so
        # that future analyses do not overwrite the base information (in the /data directory)
        k_events = Events_Tracker()

        if prefix is None:
            prfx = ''
        else:
            prfx = f'{prefix}_'

        if params['molecule'] == 'BPT':
            event_path = f'./data/events/{params["c_ver"]}/{params["name"]}/'
            k = int(os.listdir(event_path)[0].split('c')[0])  # assumes there is only a single Events object file
            event_path += f'{prfx}{k}clusters.pkl'
        else:
            event_path = f'./data/BPT_new/events/{params["c_ver_ft"]}/{params["name"]}/'

            files = os.listdir(event_path)  # assumes there is only a single Events object file for each experiment
            k = -1
            for file in files:
                if prfx[:-1] in file:
                    if not avg_group and '_avg' not in file:  # number of clusters in 'initial' Events
                        k = int(file.split('_')[1].split('c')[0])
                        break
                    elif avg_group and '_avg' in file:  # number of clusters in 'averaged' Events
                        k = int(file.split('_')[1].split('c')[0])
                        break

            if avg_group:  # load clusters formed from 'averaged' Groups
                event_path += f'{prfx}{k}clusters_avg.pkl'
            else:  # load clusters formed from 'initial' Groups
                event_path += f'{prfx}{k}clusters.pkl'

        # Define the filepath used to save all events files/figures
        if params['molecule'] == 'BPT':
            if avg_group:
                path = f'./analysis/{params["c_ver"]}/events/{params["name"]}/{prfx}{k}clusters_avg'
            else:
                path = f'./analysis/{params["c_ver"]}/events/{params["name"]}/{prfx}{k}clusters'
        else:
            if avg_group:
                path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/events/{params["name"]}/{prfx}{k}clusters_avg'
            else:
                path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/events/{params["name"]}/{prfx}{k}clusters'
        Path(path).mkdir(parents=True, exist_ok=True)

        with open(event_path, 'rb') as loader:
            event = pickle.load(loader)

            # Copy over additional information from the previous Events object
            k_events.num_events = event.num_events
            k_events.affinity_matrix = event.affinity_matrix
            k_events.distance_matrix = event.distance_matrix
            k_events.labels = event.labels

            # Instantiate list to contain [Event ID, Number of multi-peak Groups, 0] on each row, where '0' is a temp
            # value eventually becomes the number of multi-peak Groups after -ve Silhouettes sample scores are removed
            event_sizes = []
            for ev in event.events:
                # Instantiate a counter for group feature vectors with more than one track to sort top_events
                event_sizes.append([ev.id, len(ev.vectors), 0])  # append row
            event_sizes = np.array(event_sizes)  # convert to array

            # Create a new .top_events parameter, showing Events in descending order of occurrence
            k_events.top_events = event_sizes[np.argsort(event_sizes[:, 1])[::-1]]

            # Set the main diagonal of the affinity matrix to zero (requirement of Silhouette score)
            distance_zero = np.copy(k_events.distance_matrix)
            np.fill_diagonal(distance_zero, 0)

            # Calculate the Silhouette score for all events, and store it in the Events object
            s_score = silhouette_score(X=distance_zero, labels=k_events.labels, metric='precomputed')
            k_events.silhouette = s_score

            # Calculate the Silhouette sample scores used to calculate the average cluster score and identify outliers
            s_samples = silhouette_samples(X=distance_zero, labels=k_events.labels, metric='precomputed')

            # Instantiate lists of Silhouette cluster scores
            s_events = []
            s_outliers = []  # idx within event
            s_outliers_global = []  # overall idx
            for label in np.unique(k_events.labels):
                # Retrieve the appropriate samples (i.e. for the current Event/cluster)
                s_values = s_samples[np.argwhere(k_events.labels == label).squeeze()]

                # Append the mean and std of the Silhouette sample scores for the current Event
                s_events.append([np.mean(s_values), np.std(s_values)])

                # Append the indices of the outlier samples (-ve S-value suggests data is in wrong cluster)
                s_outliers.append(np.where(s_values <= 0)[0])
                s_outliers_global.extend(np.argwhere(k_events.labels == label).squeeze()[np.where(s_values <= 0)[0]])
            s_events = np.array(s_events)  # convert to array

            if avg_group:
                # Remove outlier samples from the array of Silhouette sample scores
                s_samples = np.delete(s_samples, s_outliers_global)

            # Define mode and the first line for events summary csv file
            if not avg_group:
                mode = 'w'
                line1 = ['ORIGINAL', f'{prfx[:-1]}']
            else:
                mode = 'a'
                line1 = ['AVERAGED']

            # Instantiate a list of lines to write to file, starting with a set of headers
            lines = [line1,
                     [f'{k} Clusters', f'{np.sum(k_events.top_events[:, 1])} Groups'],
                     ['Global Silhouette', f'{s_score:.4f}', 'Silhouette Scores'],
                     ['Event ID', '#Groups', 'Mean', 'Std', '#Particles']]

            # (NOTE: This csv formatting works for Windows and NOT for Linux)
            # (For Linux formatting, remove "delimiter='  '" from "wr = csv.writer(writer, delimiter=' ')")
            if params['molecule'] == 'BPT':
                summary_path = f'./analysis/{params["c_ver"]}/events/{params["name"]}'
            else:
                summary_path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/events/{params["name"]}'
            with open(f'{summary_path}/{prfx}events_summary.csv', mode, newline='') as writer:
                wr = csv.writer(writer, delimiter=' ')  # delimiter = ' ' for windows, delimiter = '\t' for linux

                wr.writerows(lines)

                for j, e in enumerate(k_events.top_events[:, 0]):
                    top = k_events.top_events[j]
                    ev = event.events[e]

                    wr.writerow([top[0], top[1], f'{s_events[e][0]:.4f}', f'{s_events[e][1]:.4f}', len(ev.particles)])

                if not avg_group:  # linebreak ahead of further additions to file
                    wr.writerow([])

            if avg_group:  # only remove outliers on averaged Groups (otherwise the averaging process will be biased)
                # <editor-fold desc="---=== [+] Remove Outliers and Refine Events ===---">
                # Remove the outlier entries from the Events dictionary
                # (Uses the event-specific index lists)
                for ev, out in zip(event.events, s_outliers):
                    # Remove outlier Groups from the Event
                    ev.vectors = list(np.delete(ev.vectors, out))

                    # Update the particles list with those that remain within the Event after outliers were removed
                    ev.particles = list(np.unique([ve['particle_num'] for ve in ev.vectors]))

                    # Update the duration statistics list (min, max) with the remaining Groups
                    ev.duration_stats = list(np.sort([ve['duration'] for ve in ev.vectors])[[0, -1]])

                    # Update the top_events list with the new #Groups (i.e. Refined)
                    k_events.top_events[np.argwhere(k_events.top_events[:, 0] == ev.id).squeeze(), 2] = len(ev.vectors)

                # Remove the outlier entries from the affinity matrix, distance matrix, and events labels array
                # (Uses the global index list)
                k_events.affinity_matrix = np.delete(np.delete(k_events.affinity_matrix, s_outliers_global, axis=0),
                                                     s_outliers_global, axis=1)  # deletes rows, then columns
                k_events.distance_matrix = np.delete(np.delete(k_events.distance_matrix, s_outliers_global, axis=0),
                                                     s_outliers_global, axis=1)  # deletes rows, then columns
                k_events.labels = np.delete(k_events.labels, s_outliers_global)

                # Redefine the zeroed affinity matrix (needed for recalculating Silhouette scores)
                affinity_zero = np.copy(k_events.distance_matrix)
                np.fill_diagonal(affinity_zero, 0)

                # Update the Global Silhouette score
                s_score = np.mean(s_samples)
                k_events.silhouette = s_score

                # Update the Events Silhouette scores
                s_events = []
                for label in np.unique(k_events.labels):
                    # Retrieve the appropriate samples (i.e. for the current Event/cluster)
                    s_values = s_samples[np.argwhere(k_events.labels == label).squeeze()]

                    # Append the mean and std of the Silhouette sample scores for the current Event
                    s_events.append([np.mean(s_values), np.std(s_values)])
                s_events = np.array(s_events)  # convert to array

                # Append the refined events summary to the existing .csv file
                lines = [['REFINED'],
                         [f'{k} Clusters', f'{np.sum(k_events.top_events[:, 2])} Groups'],
                         ['Global Silhouette', f'{s_score:.4f}', 'Silhouette Scores'],
                         ['Event ID', '#Groups', 'Mean', 'Std', '#Particles', '#Outliers Removed']]

                with open(f'{summary_path}/{prfx}events_summary.csv', 'a', newline='') as writer:
                    wr = csv.writer(writer, delimiter=' ')  # delimiter = ' ' for windows, delimiter = '\t' for linux

                    wr.writerow([])
                    wr.writerows(lines)

                    for j, e in enumerate(k_events.top_events[:, 0]):
                        top = k_events.top_events[j]
                        ev = event.events[e]

                        # Event ID, number of Groups in Event, Mean Silhouette score, std of Silhouette score, number of
                        # unique particles in Event, number of outliers removed from Event
                        wr.writerow(
                            [top[0], top[2], f'{s_events[e][0]:.4f}', f'{s_events[e][1]:.4f}', len(ev.particles),
                             top[1] - top[2]])
                # </editor-fold>

            # <editor-fold desc="---=== [+] Save Peak Wavenumbers to File ===---">
            # Save all peak locations (in wavenumbers), for all Events*, as a .txt file
            # *This is used for future analysis of peak correlations
            wavenumber_scale = wavenumber_range()  # obtain the wavenumber range

            # Instantiate a list of all peak locations
            peak_positions = []

            # Loop through all events, and then each group feature vector
            for ev in event.events:
                for vector in ev.vectors:
                    peak_positions.extend(wavenumber_scale[vector['mean_centroids']])

            # Save the list as a .txt file
            with open(f'{path}/peak_wavenumbers.txt', 'w') as w:
                for position in peak_positions:
                    w.write(f'{position}\n')
            # </editor-fold>

            # <editor-fold desc="---=== [+] Load CAE ===---">
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
            model = Autoencoder(params)
            encoder = model.layers[0]
            encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
            decoder = model.layers[1]
            decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
            if params['molecule'] == 'BPT':
                enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder'
                dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder'
            else:
                enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/encoder'
                dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/decoder'
            encoder_ckpt.restore(tf.train.latest_checkpoint(f'{enc_dir}')).expect_partial()
            decoder_ckpt.restore(tf.train.latest_checkpoint(f'{dec_dir}')).expect_partial()
            if tf.train.latest_checkpoint(f'{enc_dir}') and tf.train.latest_checkpoint(f'{dec_dir}'):
                pass
            else:
                print("No encoder and/or decoder model(s) found. Exiting...")
                exit()
            # </editor-fold>

            # Cycle through the top-k events
            for e in k_events.top_events[:, 0]:
                # Instantiate list of mean raw spectra for each group within the current event
                spectra = []
                differences = []  # same thing but for difference spectra

                # Instantiate list of particles, used to identify the 'real particle' that each scan originated from
                particles = []

                # Store the individual difference spectra in a separate list
                indiv_diff = []

                # Cycle through each group vector
                for vector in event.events[e].vectors:
                    # Find the appropriate scan
                    data = datas[np.argwhere(datas_labels == vector['particle_num']).squeeze()]

                    # Produce difference scan (splitting each scan into batches, if necessary)
                    nbatches = data.shape[0] // params['c_batch_size']
                    recon = np.empty((data.shape[0], data.shape[1], 1)).astype('float32')
                    for n in range(nbatches):
                        recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)] = decoder(encoder(
                            np.expand_dims(data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)], axis=-1)))
                    recon = recon.squeeze()
                    difference = difference_scans(data, recon)

                    # Extend list with the spectra from the appropriate time steps
                    if isinstance(vector['start'], list):
                        spectrum = []
                        diff = []
                        for start, end in zip(vector['start'], vector['end']):
                            spectrum.append(data[start:end + 1])
                            diff.append(difference[start:end + 1])
                        spectra.append(np.mean(np.concatenate(spectrum, axis=0), axis=0))
                        differences.append(np.mean(np.concatenate(diff, axis=0), axis=0))
                        indiv_diff.append(list(np.concatenate(diff, axis=0)))
                    else:
                        spectra.append(np.mean(data[vector['start']:vector['end'] + 1], axis=0))
                        differences.append(np.mean(difference[vector['start']:vector['end'] + 1], axis=0))
                        indiv_diff.append(difference[vector['start']:vector['end'] + 1])

                    # Append the particle number to list
                    particles.append(vector['particle_num'])

                # Instantiate list to contain all mean Group spectra
                spec = []
                diff = []
                for s, d in zip(spectra, differences):
                    spec.append(s)
                    diff.append(d)
                spec = np.array(spec)  # convert to array
                diff = np.array(diff)  # convert to array

                # Weight contributions of individual spectra to 'Event spectra' based on scaled silhouette sample scores
                x = s_samples[np.argwhere(k_events.labels == event.events[e].id).squeeze()]
                s_weights = np.max((np.zeros(x.shape), x), axis=0)  # linear weights (-ve = 0)

                # Define new filepath to store individual difference spectra within their respective configuration dirs
                if avg_group:
                    new_path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/events/{params["name"]}/{prfx}{k}clusters_avg/config{event.events[e].id}'
                else:
                    new_path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/events/{params["name"]}/{prfx}{k}clusters/config{event.events[e].id}'
                Path(new_path).mkdir(parents=True, exist_ok=True)
                for ii, d in enumerate(indiv_diff):
                    d_scaled = np.array(d) * s_weights[ii]
                    # Save the individual difference spectrum that make up each 'Event' (new terminology)
                    np.save(f'{new_path}/difference_event{ii}.npy', d_scaled)

                # Scale the spectra from the Event by their respective weighted silhouette sample scores
                spec_scaled = np.multiply(spec, s_weights[:, np.newaxis])
                diff_scaled = np.multiply(diff, s_weights[:, np.newaxis])

                # Save all mean Group spectra to file
                # np.save(f'{path}/mean_group_spectra_{event.events[e].id}.npy', spec)  # original spectra
                np.save(f'{path}/mean_group_spectra_{event.events[e].id}.npy', spec_scaled)  # weighted spectra
                np.save(f'{path}/mean_group_difference_{event.events[e].id}.npy', diff_scaled)  # weighted diff spec

                # Take the 'global' mean of all mean Group spectra
                mean_spectrum = np.mean(spec, axis=0)  # mean 'Event spectrum' (i.e. representative spectrum of Event)
                std_spectrum = np.std(spec, axis=0)  # std spectrum needed to show upper/lower bounds
                mean_spectrum_s = np.mean(spec_scaled, axis=0)  # mean weighted spectra
                std_spectrum_s = np.std(spec_scaled, axis=0)  # std weighted spectra
                # Scale the weighted spectra to the scale of the original spectra (this maintains the original scale,
                # only changing the 'profile' of each spectrum)
                mean_spectrum = (np.mean(mean_spectrum) / np.mean(mean_spectrum_s)) * mean_spectrum_s
                std_spectrum = (np.mean(std_spectrum) / np.mean(std_spectrum_s)) * std_spectrum_s

                # Repeat for difference spectra
                mean_difference = np.mean(diff, axis=0)
                std_difference = np.std(diff, axis=0)
                mean_difference_s = np.mean(diff_scaled, axis=0)
                std_difference_s = np.std(diff_scaled, axis=0)
                mean_difference = (np.mean(mean_difference) / np.mean(mean_difference_s)) * mean_difference_s
                std_difference = (np.mean(std_difference) / np.mean(std_difference_s)) * std_difference_s

                # Assign the mean Event spectrum to the current Event
                event.events[e].mean_spectrum = mean_spectrum
                event.events[e].mean_difference = mean_difference

                # Save the mean raw spectrum as an array
                np.save(f'{path}/mean_spectrum_event_{event.events[e].id}.npy', mean_spectrum)
                np.save(f'{path}/mean_difference_event_{event.events[e].id}.npy', mean_difference)

                if save_figs:
                    # Form lists of all data relevant to the regular/difference spectra
                    plot_data = [[mean_spectrum, std_spectrum, 'spectrum'],
                                 [mean_difference, std_difference, 'difference']]

                    # Cycle through plots (first = normal spectrum, second = difference spectrum)
                    for i in range(2):
                        mean, std, fname = plot_data[i]

                        # <editor-fold desc="---=== [+] Plot the Mean Raw Spectra ===---">
                        fig, ax1 = plt.subplots(figsize=(14, 6))
                        plt.suptitle(
                            f'Event {e} ({len(event.events[e].vectors)} Groups), Mean Raw Spectrum\nDuration = {event.events[e].duration_stats[0]} - {event.events[e].duration_stats[1]} time steps, Silhouette Score = {s_events[e][0]:.4f} $\pm$ {s_events[e][1]:.4f}',
                            fontsize=12)

                        ax1.plot(wavenumber_range(), mean, color='k', label='Mean')
                        ax1.plot(wavenumber_range(), mean + (std / 2), 'r--', label=u'\u00BD std', alpha=0.5)
                        ax1.plot(wavenumber_range(), mean - (std / 2), 'b--', label=u'-\u00BD std', alpha=0.5)

                        ax1.set_ylabel('Intensity (a.u.)', fontsize=10)
                        ax1.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=10)
                        ax1.legend(loc='lower right')

                        # <editor-fold desc="---=== [+] Inset Silhouette Sample Scores (Particle Colour-Coded) ===---">
                        # Create an axis to become the inset silhouette sample scores figure for the Event
                        ax2 = fig.add_axes([0.14, 0.465, 0.3, 0.4])

                        # Show the same range of coefficients to make cross-referencing between Event spectrum plots easier
                        if avg_group:
                            ax2.set_xlim([-0.1, 1])
                        else:
                            ax2.set_xlim([-0.4, 1])
                        # Insert blank space around silhouette plot for visual clarity
                        ax2.set_ylim([0, x.shape[0] + 2 * 2])

                        # Define upper and lower y-positions for silhouette sample scores to range between on plot
                        # (these values are arbitrary and therefore not displayed on plot)
                        y_lower = 3
                        y_upper = y_lower + x.shape[0]

                        # Retrieve labels for each sample associated with each physical particle they belong to
                        particles_r = real_particles(params, particles)

                        # Define a colour scheme based on the number of physical particles present in the Event
                        colours = np.arange(0, np.max(particles_r) + 1)
                        num_colours = len(colours)
                        if len(colours) == 1:  # avoid true_divide error
                            colours_n = [1]
                        else:
                            colours_n = (colours - np.min(colours)) / (np.max(colours) - np.min(colours))  # normalise
                        cmap = plt.get_cmap('rainbow', num_colours)(colours_n)

                        # Plot the sorted range of silhouette sample scores
                        y = np.arange(y_lower, y_upper)
                        x_idx = np.argsort(x)
                        x_srt = x[x_idx]
                        particles_r = particles_r[x_idx]

                        if i == 0:
                            # Combine (sorted) silhouette sample scores with arbitrary labels, then save to file
                            # (only need to do this once)
                            np.save(f'{path}/silhouette_scores_event_{event.events[e].id}.npy',
                                    np.vstack((x_srt, particles_r)).T)

                        for u, c in zip(colours, cmap):
                            idx = np.argwhere(particles_r == u).squeeze()
                            ax2.barh(y[idx], x_srt[idx],
                                     height=1, align='center', color=c, edgecolor='k', lw=0.5)

                        ax2.set_xlabel(f'Silhouette Sample Scores ({num_colours} Particles)', fontsize=10)
                        ax2.set_yticks([])  # Clear the yaxis labels / ticks
                        # Specify coefficients to display on x-axis
                        if avg_group:
                            ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                        else:
                            ax2.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
                        # </editor-fold>

                        plt.savefig(f'{path}/mean_{fname}_event_{event.events[e].id}.png')
                        plt.close()
                        # </editor-fold>

                # Append this event to list of events in the top-k events class
                k_events.events.append(event.events[e])

        # Save the top-k Events object to file
        with open(f'{path}/events.pkl', 'wb') as output:
            pickle.dump(k_events, output, pickle.HIGHEST_PROTOCOL)

        return

if __name__ == '__main__':
    # Load chosen hyperparameters
    hyperparams = hyperparams_setup()

    # # Collect event formation data
    # if hyperparams['molecule'] == 'BPT':
    #     # (NOTE: Don't want to add placeholder time steps as there are 999 'False' files to the 416 'True' ones, hence
    #     # the vast majority of the entries would be placeholders)
    #     event_formation_stats(params=hyperparams, ph=False)
    # else:
    #     # event_formation_stats(params=hyperparams, ph=True)
    #     event_formation_stats(params=hyperparams, ph=False)
    # exit()

    if hyperparams['molecule'] == 'BPT':
        # Set optional naming parameter to None (only used for extra ("fine-tuning") datasets)
        names = [None]

    else:
        # Each experiment is clustered separately, hence define names in order to select the correct scans & filepaths
        names = ['Aa', 'Ap', 'APa']

    for name in names:
        # Load chosen scan(s)
        scans = select_scan(params=hyperparams, particles=hyperparams['particles'], picos=['True'], exps=[name])
        # scans = select_scan(params=hyperparams, picos=['True'], exps=[name])  # all (pico) scans)

        # # Estimate the baseline for each picocavity spectrum for a computation time comparison with the CAE method
        # baseline_als(dataset=select_scan(hyperparams, picos=['True'], exps=[name]))
        # exit()

        # Run this if you want to extract only the difference scans from the CAE and reconstruction subtraction process
        obtain_difference_scans(params=hyperparams, dataset=scans)

        # Run this if you want to extract Tracks and Groups from the chosen scans
        # (NOTE: Setting save_figs=True saves 1 figure per scan to ./analysis/c_ver/*/detections/)
        form_tracks_and_groups(params=hyperparams, dataset=scans, save_figs=True)  # subset of scans

        # Run this if you want to cluster Groups into Events
        cluster_groups_into_events(params=hyperparams, dataset=scans, remove_singles=True, prefix=name, optimise_k=True)

        # Run this if you want to inspect the clustering results, saving figures and objects for each Event
        # (NOTE: Setting save_figs=True saves 1 figure per event to ./analysis/c_ver/*/events/name/c_ncluster/)
        events_summary(params=hyperparams, dataset=scans, prefix=name, save_figs=True)  # using original clusters

        # Run this if you want to average Groups within each Event, then repeat the clustering process
        # (NOTE: Setting save_figs=True saves 1 figure per event to ./analysis/c_ver/*/events/name/c_ncluster_avg/)
        # (WARNING: Running only these functions (and not the above events_summary()) will append new clustering
        # results onto the '*_events_summary.csv' file - without replacing any data already present in the file)
        average_groups(params=hyperparams, dataset=scans, prefix=name, optimise_k=True)
        events_summary(params=hyperparams, dataset=scans, prefix=name, save_figs=True, avg_group=True)
