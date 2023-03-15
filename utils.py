"""
@Created : 05/02/2021
@Edited  : 15/01/2023
@Author  : Alex Poppe
@File    : utils.py
@Software: Pycharm
@Description:
This is a collection of utility/helper functions used in various areas of the ML pipeline
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import glob
import os
from matplotlib.colors import ListedColormap
from pathlib import Path
from analysis.code_tracks.detector import difference_scans
from analysis.code_tracks.tracker import Groups


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


# <editor-fold desc="---=== [+] Functions to Check Valid User Inputs in Configurations ===---">
def displayargs(args):
    string = f'{args[0]}'
    for arg in args[1:]:
        string += f', {arg}'
    return string


def checkint(key, args):
    """ Checks that the given input is an int: outputs inside an int if so; exits if not

    Args:
        key: Str, The name of the key
        args: List, The name of the value

    Returns:
        value: Int, The user-specified value for the key
    """
    try:
        return int(args[0])
    except:
        print(f'Could not cast value(s) {displayargs(args)} to correct data type for key {key} (expected Int)')
        exit()


def checkfloat(key, args):
    """ Checks that the given input is a float: outputs inside a float if so; exits if not

    Args:
        key: Str, The name of the key
        args: List, The name of the value

    Returns:
        value: Float, The user-specified value for the key
    """
    try:
        return float(args[0])
    except:
        print(f'Could not cast value(s) {displayargs(args)} to correct data type for key {key} (expected Float)')
        exit()


def checkstr(key, args):
    """ Checks that the given input is a string: outputs inside a string if so; exits if not

    Args:
        key: Str, The name of the key
        args: List, The name of the value

    Returns:
        value: String, The user-specified value for the key
    """
    try:
        return str(args[0])
    except:
        print(f'Could not cast value(s) {displayargs(args)} to correct data type for key {key} (expected Str)')
        exit()


def checkbool(key, args):
    """ Checks that the given input is a boolean: outputs inside a boolean if so; exits if not

    Args:
        key: Str, The name of the key
        args: List, The name of the value

    Returns:
        value: Bool, The user-specified value for the key
    """
    if args[0] == 'True' or args[0] == 'true':
        return True

    elif args[0] == 'False' or args[0] == 'false':
        return False

    else:
        print(f'Could not cast value(s) {displayargs(args)} to correct data type for key {key} (expected Bool)')
        exit()


def checkliststr(key, args):
    """ Checks that the given input(s) are strings: outputs inside a list if so; exits if not

    Args:
        key: Str, The name of the key
        args: List, The name(s) of the value(s)

    Returns:
        value: None or List of Str, Contains the user-specified value for the key
    """
    if args[0] == 'all':
        return None

    else:
        value = []

        for arg in args:
            try:
                value.append(str(arg))
            except:
                print(
                    f'Could not cast value(s) {displayargs(args)} to correct data type for key {key} (expected List of Str(s))')
                exit()
        return value


def checktupleint(key, args):
    """ Checks that the given input(s) are ints: outputs inside a tuple if so; exits if not

    Args:
        key: Str, The name of the key
        args: List, The name(s) of the value(s)

    Returns:
        value: Tuple, Contains the user-specified value for the key
    """
    value = []
    for arg in args:
        try:
            value.append(int(arg))
        except:
            print(
                f'Could not cast value(s) {displayargs(args)} to correct data type for key {key} (expected Tuple of Int(s))')
            exit()
    return tuple(value)


# Define default hyperparameter names and types
key_checks = {'particles': checkliststr,
              'name': checkstr,
              'molecule': checkstr,

              'c_ver': checkstr,
              'c_ver_ft': checkstr,
              'c_record': checkbool,
              'c_record_ft': checkbool,
              'c_epochs': checkint,
              'c_epochs_ft': checkint,
              'c_learning_rate': checkfloat,
              'c_learning_rate_ft': checkfloat,
              'c_input_shape': checktupleint,
              'c_batch_size': checkint,
              'c_embedding_dim': checkint,
              'c_nclusters': checkint,

              's_ver': checkstr,
              's_ver_ft': checkstr,
              's_record': checkbool,
              's_record_ft': checkbool,
              's_epochs': checkint,
              's_epochs_ft': checkint,
              's_learning_rate': checkfloat,
              's_learning_rate_ft': checkfloat,
              's_input_shape': checktupleint,
              's_batch_size': checkint,
              's_embedding_dim': checkint,
              's_kfold': checkint}


# </editor-fold>


# <editor-fold desc="---=== [+] Functions to Instantiate User-Specified Hyperparameter Dictionary ===---">
def hyperparams_setup():
    """ Loads in the chosen config file to instantiate user-selected hyperparameters

    Returns:
        params: Dict, The complete hyperparameter dictionary
    """
    return load_config(choose_config_name())


def choose_config_name():
    """ Prompts user to specify config version they wish to use in order to load in user-selected hyperparameters

    Returns:
        cfg_name: Str, The name of the chosen config file (e.g. 'version_0' uses './configs/version_0.txt')
    """
    # Load the full file paths of all .txt files
    full_paths = np.sort(glob.glob('./configs/*.txt'))

    # Create a list of file names (without the .txt file type) to display to the user
    cfg_versions = [os.path.basename(x)[:-4] for x in full_paths]

    # Request the user to specify the config version
    print('Input the number corresponding to the hyperparameter config would you like to use. Options:')
    for i, cfg_version in enumerate(cfg_versions):
        print(f'{i} - {cfg_version}')

    # Load in their choice, if valid/available
    try:
        return full_paths[int(input('\nChoice: '))]

    except:
        print('\nChoice of config is either invalid or unavailable! Please try again')
        exit()


def load_config(cfg_name):
    """ Loads in the chosen config file to instantiate user-selected hyperparameters. All missing values are set to
    their defaults (see: default_hyperparams())

    Args:
        cfg_name: Str, The name of the chosen config file (e.g. 'version_0' uses './configs/version_0.txt')

    Returns:
        params: Dict, The complete hyperparameter dictionary
    """
    # Instantiate the hyperparameter dictionary
    params = {}

    with open(cfg_name, 'r') as reader:
        lines = reader.readlines()

    # Cycle through each line in the config file
    for line in lines:
        # Remove all lines that are comments or empty
        line = line.split('#')[0]

        # If the line is now not empty...
        if line.strip() != '':
            # Split the line into a list of [key, value1, value2, ...]
            args = line.split()

            # If there is a key and at least one value...
            if len(args) > 1:
                key = args[0]  # find the key

                if key in params:
                    print(f'Duplicate instance of key {key} found in config file')
                    exit()

                # Check if the value(s) are of the valid data type for the key
                if key in key_checks:
                    params[key] = key_checks[key](key, args[1:])

            else:
                print(f'{args} is not a valid key and/or value(s)!')
                exit()

    return default_hyperparams(params)


def default_hyperparams(params=None):
    """ Create new key/value pairs inside the CAE/Siamese-CNN input hyperparameter dictionary wherever there are any
    missing. Keys with 'c_' or 's_' prefixes are for CAE and Siamese-CNN, respectively. Keys with no prefix are generic.

    Args:
        params: Dict, The hyperparameter dictionary

    Returns:
        params: Dict, The modified hyperparameter dictionary
    """
    # <editor-fold desc="---=== [+] Generic ===---">
    if 'particles' not in params:  # the chosen Particles (i.e. scans) from which to form Events
        # (if you want to use all particles, set value of key to None)
        params['particles'] = ['Particle_0284', 'Particle_0506', 'Particle_0865', 'Particle_1045', 'Particle_1362']
    elif params['particles'] == 'all':  # user specifies 'all' (as it is more intuitive), but code requires 'NoneType'
        params['particles'] = None

    if 'name' not in params:  # prefix for filepaths/filenames
        # (e.g. 'allscans' when using all particles, or 'subsetscans' when using a subset [see above])
        params['name'] = 'subsetscans'

    if 'molecule_pt' not in params:  # which molecule dataset to use for pre-training
        params['molecule_pt'] = 'BPT'

    if 'molecule' not in params:  # which molecule dataset to use for fine-tuning and analysis
        params['molecule'] = params['molecule_pt']  # options: BPT, BPT_new
    # </editor-fold>

    # <editor-fold desc="---=== [+] CAE ===---">
    c_default = False  # boolean used to prevent retraining of the original CAE
    if 'c_ver' not in params:  # CAE version number (CAE is trained on all scans regardless of name/particles options)
        c_default = True
    elif params['c_ver'] == 'cae_v1':
        print(
            'cae_v1 is the name of the original CAE.\nPlease choose a different name for a new CAE, or remove the c_ver option from the config file if you wish to use the default model.')
        exit()
    if c_default:
        params['c_ver'] = 'cae_v1'

    c_default_ft = False  # boolean used to prevent retraining of the original fine-tuned CAE
    if 'c_ver_ft' not in params:  # CAE Fine-tuned version number
        c_default_ft = True
    elif params['c_ver_ft'] == 'cae_v1_ftPH' and c_default:  # #### REMOVE 'PH' FROM DEFAULT ONCE MODEL IS TRAINED ####
        print(
            'cae_v1_ft is the name of the original fine-tuned CAE (version cae_v1).\nPlease choose a different name for a new CAE model, or remove the c_ver_ft option from the config file if you wish to use the default model.')
        exit()
    if c_default_ft:
        params['c_ver_ft'] = 'cae_v1_ft'

    if 'c_record' not in params:  # whether to store the learned weights/loss curves during training or not
        params['c_record'] = False
    elif params['c_record'] and c_default:
        print(
            'You cannot train on the original CAE.\nIf you wish to train it for more epochs, copy and rename the cae_v1 folders inside ./nn/checkpoints/cae/ and ./nn/logs/cae/ then change c_ver option appropriately')
        exit()

    if 'c_record_ft' not in params:  # whether to store the learned weights/loss curves during fine-tuning
        params['c_record_ft'] = False
    elif params['c_record_ft'] and c_default and c_default_ft:
        print(
            'You cannot train on the original fine-tuned CAE.\nIf you wish to train it for more epochs, copy and rename the cae_v1_ft folders inside ./nn/checkpoints/cae/cae_v1_ft and ./nn/logs/cae/cae_v1_ft then change c_ver_ft option appropriately')
        exit()

    if 'c_epochs' not in params:  # number of epochs
        params['c_epochs'] = 2500

    if 'c_epochs_ft' not in params:  # number of epochs during fine-tuning
        params['c_epochs_ft'] = 1000

    if 'c_learning_rate' not in params:  # the learning rate
        params['c_learning_rate'] = 0.001

    if 'c_learning_rate_ft' not in params:  # the learning rate during fine-tuning
        params['c_learning_rate_ft'] = 0.0001

    if 'c_input_shape' not in params:  # the input shape (ignoring the batch size)
        params['c_input_shape'] = (512, 1)

    if 'c_batch_size' not in params:  # the batch size
        params['c_batch_size'] = 500

    if 'c_embedding_dim' not in params:  # size of the CAE embedding
        params['c_embedding_dim'] = 32

    if 'c_nclusters' not in params:  # number of clusters (i.e. number of Events) during spectral clustering
        params['c_nclusters'] = 8
    # </editor-fold>

    # <editor-fold desc="---=== [+] Siamese-CNN ===---">
    s_default = False  # boolean used to prevent retraining of the original pre-trained Siamese-CNN
    if 's_ver' not in params:  # Siamese-CNN Pre-trained version number
        s_default = True
    elif params['s_ver'] == 'siamese_v1':
        print(
            'siamese_v1 is the name of the original Siamese-CNN.\nPlease choose a different name for a new Siamese-CNN, or remove the s_ver option from the config file if you wish to use the default model.')
        exit()
    if s_default:
        params['s_ver'] = 'siamese_v1'

    s_default_ft = False  # boolean used to prevent retraining of the original fine-tuned Siamese-CNN
    if 's_ver_ft' not in params:  # Siamese-CNN Fine-tuned version number
        s_default_ft = True
    elif params['s_ver_ft'] == 'finetune_v1' and s_default:
        print(
            'finetune_v1 is the name of the original fine-tuned Siamese-CNN (version siamese_v1).\nPlease choose a different name for a new Siamese-CNN model, or remove the s_ver_ft option from the config file if you wish to use the default model.')
        exit()
    if s_default_ft:
        params['s_ver_ft'] = 'finetune_v1'

    if 's_record' not in params:  # whether to store the learned weights/loss curves during pre-training
        params['s_record'] = False
    elif params['s_record'] and s_default:
        print(
            'You cannot train on the original pre-trained Siamese-CNN.\nIf you wish to train it for more epochs, copy and rename the siamese_v1 folders inside ./nn/checkpoints/siamese_cnn/ and ./nn/logs/siamese_cnn/ then change s_ver option appropriately')
        exit()

    if 's_record_ft' not in params:  # whether to store the learned weights/loss curves during fine-tuning
        params['s_record_ft'] = False
    elif params['s_record_ft'] and s_default and s_default_ft:
        print(
            'You cannot train on the original fine-tuned Siamese-CNN.\nIf you wish to train it for more epochs, copy and rename the finetune_v1 folders inside ./nn/checkpoints/siamese_cnn/siamese_v1 and ./nn/logs/siamese_cnn/siamese_v1 then change s_ver_ft option appropriately')
        exit()

    if 's_epochs' not in params:  # number of epochs during pre-training
        params['s_epochs'] = 1000

    if 's_epochs_ft' not in params:  # number of epochs during fine-tuning
        params['s_epochs_ft'] = 13

    if 's_learning_rate' not in params:  # the learning rate during pre-training
        params['s_learning_rate'] = 0.01

    if 's_learning_rate_ft' not in params:  # the learning rate during fine-tuning
        params['s_learning_rate_ft'] = 0.001

    if 's_input_shape' not in params:  # the input shape (ignoring the batch size)
        params['s_input_shape'] = (100, 25, 1)

    if 's_batch_size' not in params:  # the batch size
        params['s_batch_size'] = 64

    if 's_embedding_dim' not in params:  # size of the last FC layer in the siamese-CNN
        params['s_embedding_dim'] = 128

    if 's_kfold' not in params:  # value of k in k-fold cross-validation (i.e. number of data partitions)
        params['s_kfold'] = 10
    # </editor-fold>

    return params


# </editor-fold>


def append_value(dict_obj, key, value, append_list=True):
    """ Function to append to a list of a key if that key already exists, else create the key and append the first value

    Args:
        dict_obj: The dictionary
        key: The key
        value: The value to be appended to the list of the key
        append_list: Bool, if True: Creates/appends values onto new/existing keys as lists, if False: Creates/appends
            these values as their own type (this assumes that each key will only be created once, and never appended)
    """
    if append_list:
        # Check if the key exists in the dictionary or not
        if key in dict_obj:
            # Append the value to the list
            dict_obj[key].append(value)
        else:
            # Create a new entry to the dictionary with name 'key' and append the first value
            dict_obj[key] = [value]
    else:
        # Create a new entry to the dictionary with name 'key' and append the first and only value
        dict_obj[key] = value
    return


def shuffle_in_unison(arrs, axis=0, seed=False):
    """ Function to shuffle n (n > 1) input arrays - of the same size along the chosen axis - the exact same way

    Args:
        arrs: List, List of arrays
        axis: Int, The axis to shuffle along
        seed: Bool, If False: Uses 'true random' shuffle key, if True: Uses seeded shuffle key

    Returns:
        arrs: List, List of shuffled arrays
    """
    # Assert that all arrays are of the same size
    assert all(len(arr) == len(arrs[axis]) for arr in arrs)

    if seed is None:
        # Create a shuffle key
        key = np.random.permutation(arrs[0].shape[axis])
    else:
        # Use user-specified shuffle key
        key = np.random.RandomState(42).permutation(arrs[0].shape[axis])

    # Permute every array by the same key, and return the reformed the list
    if axis == 0:
        return [arr[key] for arr in arrs]

    else:
        # Swap the first and chosen axis, shuffle based on the chosen key, then swap back the axes
        return [np.swapaxes(np.swapaxes(arr, 0, axis)[key], 0, axis) for arr in arrs]


def distance_metric(features):
    """ Calculate the L1-difference between the two feature vectors

    Args:
        features: List, The 0-index contains the feature vector output from one siamese network, the 1-index contains
            the feature vector output from the other siamese network
    """

    embed_a, embed_b = features  # unpack features
    return tf.math.abs(embed_a - embed_b)


def wavenumber_range(default=False):
    """ Returns the size (512,) wavenumber range used in various plotting and data analysis purposes

    Args
        params: Dict, The hyperparameters dictionary
    """
    if default:
        return np.arange(268, 1611, 2.625)

    else:
        try:  # return a corrected wavenumber range if available...
            return np.genfromtxt(f'./data/BPT_wavenumber_range.txt')

        except OSError:  # ...else return a default (the same as used in bpt_dataset() in sers.scanalyser.data.dataset)
            return np.arange(268, 1611, 2.625)


def overlays(params, data, tracker, label_str, overlay_type, recon=None):
    """ Plots the input scan with the tracks overlaid, colour-coded based on the chosen style (track or ID-based)

    Args:
        params: Dict, The hyperparameter dictionary
        data: Array, The input scan
        tracker: class, the tracker for the current scan
        label_str: str, label list formatted to a single str used for display purposes
        overlay_type: Str, Changes various properties of the plotted data and figure information, based on what is
            being overlaid. Options: 'rudimentary', 'zip', or 'group'
        recon: Array, The reconstructed scan
    """
    if overlay_type not in ['zip', 'group']:
        print('ERROR: Incorrect overlay_type specified!\n(Options: zip or group)\nExiting...')
        exit()

    data = np.squeeze(data)

    # Create the full colourmap containing colours for all tracks or group IDs
    if overlay_type == 'group':
        # Count the total number of unique track IDs
        num_groups = len(np.unique([x.group_id for idx, x in enumerate(tracker.tracks)]))
        colours = np.random.permutation(plt.cm.rainbow_r(np.linspace(0, 1, num_groups)))
    else:
        colours = np.random.permutation(plt.cm.rainbow_r(np.linspace(0, 1, len(tracker.tracks))))
    full_cmap = ListedColormap(colours)
    full_cmap.set_bad(alpha=0)

    # Create a full mask, which will contain all tracks
    empty_full_mask = np.zeros(data.shape)
    full_mask = np.ma.masked_array(empty_full_mask, empty_full_mask != 1)

    # Loop through tracks, assigning the colour (group_id) to the mask x-y coords (trace(_time/_bound))
    for idx in range(len(tracker.tracks)):
        # Obtain the x-y coords to be plotted
        trace_time = np.round(np.array(tracker.tracks[idx].trace_time)).astype('int')
        trace_bounds = np.round(np.array(tracker.tracks[idx].trace_bounds)).astype('int')

        # Apply group_id/idx to the (full_)mask, for applying the appropriate colour
        for ti, bound in zip(trace_time, trace_bounds):
            if overlay_type == 'zip':
                full_mask[ti][bound[0]:bound[1] + 1] = idx
            else:  # overlay_type == 'group'
                full_mask[ti][bound[0]:bound[1] + 1] = tracker.tracks[idx].group_id

    # plt.figure(figsize=(6, 8))
    # if overlay_type == 'zip':
    #     plt.title(f'Particle {label_str} ({len(tracker.tracks)} Tracks)', fontsize=14)
    # else:  # overlay_type == 'group'
    #     plt.title(f'Particle {label_str} ({num_groups} Groups)', fontsize=14)
    # plt.imshow(data, cmap='gray')
    # plt.xlabel('Pixel', fontsize=12)
    # plt.ylabel('Time Step', fontsize=12)
    # plt.imshow(full_mask, cmap=full_cmap, alpha=1)
    # plt.tight_layout()
    #
    # if params['molecule'] == 'BPT':
    #     path = f'./analysis/{params["c_ver"]}/detections'
    # else:
    #     path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/detections'
    # Path(path).mkdir(parents=True, exist_ok=True)
    # if overlay_type == 'zip':
    #     plt.savefig(f'{path}/particle_{label_str}_2_zip_overlay.png')
    # else:  # overlay_type == 'group'
    #     plt.savefig(f'{path}/particle_{label_str}_3_group_overlay.png')
    # plt.close()

    # Produce a 1x4 plot of [input, recon, diff, groups]
    if overlay_type == 'group' and recon is not None:
        recon = recon.squeeze()

        # Calculate the difference scan
        difference = difference_scans(data=data, recon=recon)

        # Create ticks/labels for the wavenumber axis
        wn_range = np.arange(268, 1611, 2.625)
        m = 2
        wavenumber_samples = np.arange(np.ceil(wn_range[0] / (100 * m)) * 100 * m,
                                       np.ceil(wn_range[-1] / (100 * m)) * 100 * m,
                                       100 * m).astype('int')
        ticks = []
        for sample in wavenumber_samples:
            lr = np.abs(wn_range - sample)
            a, b = np.partition(lr, 1)[:2]  # finds the two smallest values
            left = np.argwhere(lr == a).squeeze()
            right = np.argwhere(lr == b).squeeze()
            ticks.append(a / (wn_range[right] - wn_range[left]) * (right - left) + left)

        # Retrieve the original particle information from the global particle number
        particle_info = particle_name(particle=f'Particle_{label_str}')

        _, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(16, 5))
        plt.suptitle(f'Particle {label_str} ({particle_info[0]})', fontsize=14)

        ax0.set_title('Original', fontsize=12)
        ax0.imshow(data, cmap='inferno')
        ax0.set_xlabel('Wavenumber cm$^{-1}$', fontsize=12)
        ax0.set_ylabel('Time Step (a.u.)', fontsize=12)
        ax0.set_xticks(ticks)
        ax0.set_xticklabels(wavenumber_samples)

        ax1.set_title('Reconstruction', fontsize=12)
        ax1.imshow(recon, cmap='inferno')
        ax1.set_xlabel('Wavenumber cm$^{-1}$', fontsize=12)
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(wavenumber_samples)

        ax2.set_title('Difference', fontsize=12)
        ax2.imshow(difference, cmap='inferno')
        ax2.set_xlabel('Wavenumber cm$^{-1}$', fontsize=12)
        ax2.set_xticks(ticks)
        ax2.set_xticklabels(wavenumber_samples)

        ax3.set_title(f'Groups ({num_groups})', fontsize=12)
        ax3.imshow(data, cmap='gray')
        ax3.imshow(full_mask, cmap=full_cmap, alpha=1)
        ax3.set_xlabel('Wavenumber cm$^{-1}$', fontsize=12)
        ax3.set_xticks(ticks)
        ax3.set_xticklabels(wavenumber_samples)

        plt.autoscale()
        plt.tight_layout()

        if params['molecule'] == 'BPT':
            path = f'./analysis/{params["c_ver"]}/detections'
        else:
            path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}/detections'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{path}/particle_{label_str}.png', dpi=200)
        plt.close()

    return


def group_converter(params, data, tracker, timesteps, width, label_str):
    """ Function to create and save feature vectors from each track Group.

    Args:
        params: Dict, The hyperparameter dictionary
        data: Array, The scan for the current Group
        tracker: class, the Tracker class containing all separated tracks
        timesteps: Int, the number of time steps in the input scan
        width: Int, the number of pixel-wavenumbers in the input scan
        label_str: Str, The particle number (e.g. '1045' from 'Particle_1045')
    """
    # Create a dictionary with each unique ID as the keys, and their associated tracks as the values
    id_dict = {}
    for track in tracker.tracks:
        append_value(id_dict, track.group_id, track)

    # Instantiate Groups object to store all Groups within one scan
    groups = Groups(particle_num=label_str)

    for group_id, tracks in zip(id_dict.keys(), id_dict.values()):
        # Create an empty feature vector (spectrum) for the current group feature vector dictionary (ID also included)
        vector = {'id': group_id, 'spectrum': np.zeros(width)}

        # Instantiate variables to store the earliest start and latest end times
        first_start, last_end = timesteps, 0  # PH values

        # Instantiate list of mean wavenumber positions for tracks, their respective Q3 widths*, and the number of times
        # each track has been zipped
        mean_centroids = []
        q3_widths = []
        num_zips = []
        for track in tracks:
            # # Round the trace bounds and find the min/max values
            # rounded_bounds = np.round(track.trace_bounds).astype('int')
            # rounded_bounds[rounded_bounds >= width - 1] = width - 1  # prevent values going beyond width of spectra
            # min_bound = np.min(rounded_bounds)
            # max_bound = np.max(rounded_bounds)
            #
            # # Calculate the normalised peak occurrence at each wavenumber-pixel position (i.e. time step sum) by...
            # # ...creating a column vector of trace bound wavenumber positions, then...
            # idx = np.arange(min_bound, max_bound + 1).reshape(-1, 1)  # reshape to column vector
            # # ...find, for each time step, which trace bounds are present within the min/max range, then...
            # num_matches = np.logical_and(rounded_bounds[:, 0] <= idx, idx <= rounded_bounds[:, 1])
            # # ...count, column-wise, the number of times each wavenumber is occupied, then normalise between 0 - 1
            # norm_matches = np.sum(num_matches, axis=1) / num_matches.shape[1]
            #
            # # Update the normalised values in the feature vector at the appropriate indices
            # # (Note: '+=' because another track *could* have bounds at the same indices. This will make the vector
            # # unnormalised, however, this will be corrected at a later stage)
            # vector['spectrum'][idx.T] += norm_matches

            # Update first_start if an earlier time exists
            if min(track.trace_time) < first_start:
                first_start = min(track.trace_time)

            # Update last_end if a later time exists
            if max(track.trace_time) > last_end:
                last_end = max(track.trace_time)

            # Calculate the centroid of the current track
            mean_centroids.append(int(np.round(np.mean(track.trace))))

            # Calculate the Q3 width of the current track (*Q3 instead of max to better account for outliers)
            q3_widths.append(int(np.round(np.quantile([np.diff(bound) for bound in track.trace_bounds], 0.75))))

            # Store the number of times the current track has been zipped
            num_zips.append(track.zip_count)

        # Calculate the group duration
        group_duration = last_end - first_start + 1

        # Convert to arrays, and sort w.r.t mean_centroids
        sort = np.argsort(mean_centroids)
        q3_widths = np.array(q3_widths)[sort]
        mean_centroids = np.array(mean_centroids)[sort]
        num_zips = np.array(num_zips)[sort]

        # Create additional keys for details of the group feature vector
        vector['duration'] = group_duration  # duration of current group
        vector['start'] = first_start  # start time for current group (used for future annotation)
        vector['end'] = last_end  # end time for current group (used for future annotation)
        vector['particle_num'] = label_str  # store particle number, in case future annotations require it
        vector['num_tracks'] = len(tracks)  # number of tracks (used in Event clustering)
        vector['mean_centroids'] = mean_centroids  # mean wavenumber position for each track
        vector['q3_widths'] = q3_widths  # Q3 width for each track
        vector['zip_counts'] = num_zips  # number of zips for each track
        vector['spectrum'] = np.mean(data[vector['start']:vector['end'] + 1], axis=0)

        # Append the current group feature vector to the list of vectors within the current Group object
        groups.vectors.append(vector)

        # # Save the Groups individually [NOT USED: This could potentially create & save 10k+ files]
        # path = f'{params["fpath"]}/data/groups/{params["cae_ver"]}'
        # Path(path).mkdir(parents=True, exist_ok=True)
        # with open(f'{path}/particle_{label_str}_id_{group_id}.pkl', 'wb') as w:
        #     pickle.dump(vector, w, pickle.HIGHEST_PROTOCOL)

    # Save the Groups stored found within this scan
    if params['molecule'] == 'BPT':
        path = f'./data/groups/{params["c_ver"]}'
    else:
        path = f'./data/BPT_new/groups/{params["c_ver_ft"]}'
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f'{path}/particle_{label_str}.pkl', 'wb') as w:
        pickle.dump(groups, w, pickle.HIGHEST_PROTOCOL)

    return


def particle_name(params, particle=None):
    """ Find and return the real name of the particle (intended for the BPT_new dataset)

    Args:
        params: Dict, The hyperparameter dictionary
        particle: List/Str, The global particle name (e.g. Particle_2418 or ['2418', ...])

    Returns:
        List of information matching the given global particle name
    """
    if isinstance(particle, str):  # If the format is 'Particle_xxxx' then convert to ['xxxx']
        particle = [particle[-4:]]

    with open('./data/BPT_new/particle_reference.txt', 'r') as r:
        lines = r.readlines()

        if params['particles'] is None and particle is None:  # use all particles if specified
            particle = np.arange(0, len(lines), 1)  # e.g. [0, 1, ..., 3666]

        particles = []
        for part in particle:
            # Retrieve the appropriate line by converting the global particle number to an int
            line = lines[int(part)]  # particle_reference.txt is sorted by global particle number

            # Return a formatted version of the particle information
            # - The \n and leading/trailing spaces are removed
            # - A list is return where each piece of information is a separate entry
            # -- e.g. ['Au NP on 1ML Pd mirror', 'Particle_68', 'kinetic_SERS_1', 'False']
            particles.append([ln.strip() for ln in line.rstrip().split('=')[-1].split(',')])

        if len(particles) == 1:  # output a single list if only one particle is converted
            particles = particles[0]

        return particles


def real_particles(params, particles=None, keep_strings=False):
    """ Converts a list of global particle numbers into a list of arbitrary labels, where each label represents the
    physical particle that each global particle number belongs to.

    For example:
        Particle_0000 = Au NP on 1ML Pd mirror, *Particle_1*, kinetic_SERS, 200, True
        Particle_0001 = Au NP on 1ML Pd mirror, *Particle_1*, kinetic_SERS_0, 200, True
        Particle_0002 = Au NP on 1ML Pd mirror, *Particle_10*, kinetic_SERS, 200, True
        Particle_0003 = Au NP on 1ML Pd mirror, *Particle_10*, kinetic_SERS_0, 200, True
        Particle_0004 = Au NP on 1ML Pd mirror, *Particle_10*, kinetic_SERS_1, 200, True
    Output would be:
        [0, 0, 1, 1, 1], based on the 'Particle_X' value (also separated by the experiment - i.e. Aa, APa and Ap)

    Args:
        params: Dict, The hyperparameter dictionary
        particles: List, Strings for each global particle number
        keep_strings: Bool If False: Only outputs the unique labels, If True: Also outputs the sample_power strings
            alongside the other output
    """
    # The original BPT dataset does not have 'scans within particles' (i.e. each scan is the entirety of the particle
    # that it belongs to), whereas scans in the BPT_new dataset are a part of the overall particle
    if params['molecule'] != 'BPT':
        # Convert particle numbers to their 'real names' found in the reference list
        particles = particle_name(params, particles)

        # Combine the first two parts of each particle name (e.g. Au NP on Au mirror + Particle_0)
        if keep_strings:
            strings = []
            for particle in particles:
                if particle[0] == 'Au NP on Au mirror':
                    ref = 'Aa'
                elif particle[0] == 'Au NP on 1ML Pd mirror':
                    ref = 'Ap'
                else:
                    ref = 'APa'
                strings.append(ref + '_' + particle[-2])

            particles = [particle[0] + '_' + particle[1] + '_' + particle[-2] for particle in particles]

        else:
            # particles = [particle[0] + '_' + particle[1] for particle in particles]  # OLD VERSION
            particles = [particle[0] + '_' + particle[1] + '_' + particle[-2] for particle in particles]

    else:
        if particles is None:  # use all BPT particles ('True' scans only, which are all contained in the test dataset)
            # Loop through each scan/particle (these are synonymous for the original BPT dataset)
            if keep_strings:
                particles = []
                strings = []
                for fpath in glob.glob('./data/test/*.npy'):
                    scan = fpath.split('/')[-1]

                    # Retrieve the particle number and append to list
                    particles.append('_'.join(scan.split('_')[1:3]))

                    # Retrieve the 'key' (i.e. BPT_power) and append to list
                    strings.append('BPT_' + scan.split('_')[3:4][0])
            else:
                particles = []
                for fpath in glob.glob('./data/test/*.npy'):
                    scan = fpath.split('/')[-1]

                    # Retrieve the particle number and append to list
                    particles.append('_'.join(scan.split('_')[1:3]))

    # Find the unique physical particles
    particles_unique = np.unique(particles)

    # Return the labels for each scan that belongs to each of the previously defined physical particles
    if keep_strings:
        return np.array([np.argwhere(particles_unique == particle).squeeze() for particle in particles]), strings
    else:
        return np.array([np.argwhere(particles_unique == particle).squeeze() for particle in particles])
