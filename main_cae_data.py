"""
@Created : 22/01/2020
@Edited  : 15/01/2023
@Author  : Alex Poppe
@File    : dataset.py
@Software: Pycharm
@Description:
Various functions that preprocess, split, and load the BPT SERS data
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import h5py
import glob
import csv
from scipy.interpolate import interp1d
from pathlib import Path


#######################################################################################################################
# Original BPT dataset
#######################################################################################################################
def bpt_dataset():
    """ Read in raw SERS dataset in an .h5 format. Each scan is preprocessed using a cubic spline interpolation. The
    'False' scans (those not containing transient events) are split between the training and validation datasets, and
    'True' scans (those containing transient events) are used as the testing dataset

    NOTE:
    The data is saved as individual csv/npy files so that they can be loaded using a system of identifying labels. This
    also means that the batch size should be some number wholly divisible by the number of time steps (i.e. 1000 time
    steps in the BPT scans, so recommended batch sizes are: 1000, *500*, 250, 200, 100, etc.)
    """
    print('>> Forming train/valid/test datasets')

    # Open raw .h5 dataset
    dataset = h5py.File('./data/BPT_Dataset_For_Kent.h5', 'r')

    # Create new wavenumber scale (step size = wavenumber resolution in pixels)
    wavenumber_scale = np.arange(268, 1611, 2.625)

    # Form a list of flags for splitting 'False' scans into the train/valid datasets (3:1 ratio)
    choose_train, choose_valid = ['train'], ['valid']
    choose_dataset = 3 * choose_train + 1 * choose_valid

    # Create the directories for the train/valid/test datasets if they do not yet exist
    Path('./data/train/').mkdir(parents=True, exist_ok=True)
    Path('./data/valid/').mkdir(parents=True, exist_ok=True)
    Path('./data/test/').mkdir(parents=True, exist_ok=True)

    # Create csv writers for spectrum labels and spectrum names for all three datasets
    with open(f'./data/train/train_labels.csv', 'w', newline='') as trainlabelcsv, \
            open(f'./data/train/train_specs.csv', 'w', newline='') as trainspeccsv, \
            open(f'./data/valid/valid_labels.csv', 'w', newline='') as validlabelcsv, \
            open(f'./data/valid/valid_specs.csv', 'w', newline='') as validspeccsv, \
            open(f'./data/test/test_labels.csv', 'w', newline='') as testlabelcsv, \
            open(f'./data/test/test_specs.csv', 'w', newline='') as testspeccsv:
        trainlabelwriter = csv.writer(trainlabelcsv)
        trainspecwriter = csv.writer(trainspeccsv)
        validlabelwriter = csv.writer(validlabelcsv)
        validspecwriter = csv.writer(validspeccsv)
        testlabelwriter = csv.writer(testlabelcsv)
        testspecwriter = csv.writer(testspeccsv)

        for particle in dataset.keys():
            # Obtain the label information (already have particle from iterator)
            power = dataset[particle].attrs['Laser_Power']
            pico = dataset[particle].attrs['Contains_Picocavity']
            idx = particle.index('_')  # Add zeroes before the particle number if < 1000 to make the number str len(4)
            particle_str = particle[:idx + 1] + '0' * (13 - len(particle)) + particle[1 + idx:]

            # Place 'True' scans within the testing dataset
            if pico:
                flags = 'test'
                label = [flags, particle_str, power, pico]

            # Randomly assign all pico == False scans between the train/valid datasets
            else:
                flags = np.random.choice(choose_dataset)
                label = [flags, particle_str, power, pico]

            # Obtain the wavenumber information
            wavenumber = dataset[particle].attrs['wavenumber_axis']

            # Reduce the scan by 300 (i.e. remove average background count)
            scan = dataset[particle][:] - 300

            # Crop the wavenumber and scans to remove the anti-stokes and notch filter regions
            # (This should be redundant as the interpolation would remove those wavenumbers. It was done for speed)
            wavenumber = wavenumber[845:]
            scan = scan[:, 845:]

            # Create interpolation function and interpolate the scans into the given wavenumber scale
            scan_reshape = interp1d(wavenumber[:], scan, kind='cubic', fill_value='extrapolate', axis=-1)(
                wavenumber_scale)

            # Zero out the regions for each scan without data
            # (a fill_value of 0 could be set for interp1d() but it didn't work? so I manually set outliers to 0)
            scan_reshape[:, wavenumber_scale < wavenumber[0]] = 0
            scan_reshape[:, wavenumber_scale > wavenumber[-1]] = 0

            # Linearly rescale the data between [0, 1]
            scan_reshape = (scan_reshape - np.min(scan_reshape)) / (np.max(scan_reshape) - np.min(scan_reshape))

            # Save the interpolated scans as npy files in their respective directories
            # - Write directories of each scan to respective csv (name_spec is the *exact* filepath to each npy)
            # - Write labels to respective csv files
            if flags == 'train':
                np.save(f'./data/train/scan_{particle_str}_{power}_{pico}.npy', scan_reshape)
                name_spec = [f'./data/train/scan_{particle_str}_{power}_{pico}.npy']
                trainspecwriter.writerow(name_spec)
                trainlabelwriter.writerow(label)

            elif flags == 'valid':
                np.save(f'./data/valid/scan_{particle_str}_{power}_{pico}.npy', scan_reshape)
                name_spec = [f'./data/valid/scan_{particle_str}_{power}_{pico}.npy']
                validspecwriter.writerow(name_spec)
                validlabelwriter.writerow(label)

            else:
                np.save(f'./data/test/scan_{particle_str}_{power}_{pico}.npy', scan_reshape)
                name_spec = [f'./data/test/scan_{particle_str}_{power}_{pico}.npy']
                testspecwriter.writerow(name_spec)
                testlabelwriter.writerow(label)
    return


#######################################################################################################################
# New BPT dataset
#######################################################################################################################
# Create wavenumber arrays for each dataset/power combo
def create_wavenumbers():
    """ Creates 12 npy files, one for each dataset/power combo (e.g. Aa_100.npy)
    """
    # Instantiate lists
    APa_50 = []
    APa_100 = []
    APa_150 = []
    APa_200 = []

    Ap_50 = []
    Ap_100 = []
    Ap_150 = []
    Ap_200 = []

    Aa_50 = []
    Aa_100 = []
    Aa_150 = []
    Aa_200 = []

    # Read the csv containing all wavenumber information
    with open(f'./data/BPT_new/wavenumber_info.csv', 'r') as r:
        lines = r.readlines()[2:]

        for line in lines:
            unpack = [float(i) for i in line.split(',')]

            # Au_Pd NP on Au mirror
            APa_50.append(unpack[0])
            APa_100.append(unpack[1])
            APa_150.append(unpack[2])
            APa_200.append(unpack[3])

            # Au NP on 1ML Pd mirror
            Ap_50.append(unpack[4])
            Ap_100.append(unpack[5])
            Ap_150.append(unpack[6])
            Ap_200.append(unpack[7])

            # Au NP on Au mirror
            Aa_50.append(unpack[8])
            Aa_100.append(unpack[9])
            Aa_150.append(unpack[10])
            Aa_200.append(unpack[11])

        APa_50 = np.array(APa_50)
        APa_100 = np.array(APa_100)
        APa_150 = np.array(APa_150)
        APa_200 = np.array(APa_200)
        Ap_50 = np.array(Ap_50)
        Ap_100 = np.array(Ap_100)
        Ap_150 = np.array(Ap_150)
        Ap_200 = np.array(Ap_200)
        Aa_50 = np.array(Aa_50)
        Aa_100 = np.array(Aa_100)
        Aa_150 = np.array(Aa_150)
        Aa_200 = np.array(Aa_200)

        np.save('./data/BPT_new/datasets/APa_50.npy', APa_50)
        np.save('./data/BPT_new/datasets/APa_100.npy', APa_100)
        np.save('./data/BPT_new/datasets/APa_150.npy', APa_150)
        np.save('./data/BPT_new/datasets/APa_200.npy', APa_200)
        np.save('./data/BPT_new/datasets/Ap_50.npy', Ap_50)
        np.save('./data/BPT_new/datasets/Ap_100.npy', Ap_100)
        np.save('./data/BPT_new/datasets/Ap_150.npy', Ap_150)
        np.save('./data/BPT_new/datasets/Ap_200.npy', Ap_200)
        np.save('./data/BPT_new/datasets/Aa_50.npy', Aa_50)
        np.save('./data/BPT_new/datasets/Aa_100.npy', Aa_100)
        np.save('./data/BPT_new/datasets/Aa_150.npy', Aa_150)
        np.save('./data/BPT_new/datasets/Aa_200.npy', Aa_200)

    return


# New BPT dataset functions
def new_bpt_dataset():
    """ Read in raw SERS dataset in an .h5 format. Each scan is preprocessed using a cubic spline interpolation. The
    'False' scans (those not containing transient events) are split between the training and validation datasets, and
    'True' scans (those containing transient events) are used as the testing dataset

    NOTE:
    The data is saved as individual csv/npy files so that they can be loaded using a system of identifying labels. This
    also means that the batch size should be some number wholly divisible by the number of time steps (i.e. 1000 time
    steps in the BPT scans, so recommended batch sizes are: 1000, *500*, 250, 200, 100, etc.)
    """
    print('>> Forming train/valid/test datasets')

    # <editor-fold desc="---=== [+] Load Wavenumber Information ===---">
    # Create new wavenumber scale (step size = wavenumber resolution in pixels)
    # (NOTE: This is the same wavenumber scale that was used in the old BPT dataset (in order to combine them))
    wavenumber_scale = np.arange(268, 1611, 2.625)

    # Load in wavenumber arrays corresponding to each dataset/power combo into a dictionary
    wn_dict = {'APa_50': np.load('./data/BPT_new/datasets/APa_50.npy'),
               'APa_100': np.load('./data/BPT_new/datasets/APa_100.npy'),
               'APa_150': np.load('./data/BPT_new/datasets/APa_150.npy'),
               'APa_200': np.load('./data/BPT_new/datasets/APa_200.npy'),
               'Ap_50': np.load('./data/BPT_new/datasets/Ap_50.npy'),
               'Ap_100': np.load('./data/BPT_new/datasets/Ap_100.npy'),
               'Ap_150': np.load('./data/BPT_new/datasets/Ap_150.npy'),
               'Ap_200': np.load('./data/BPT_new/datasets/Ap_200.npy'),
               'Aa_50': np.load('./data/BPT_new/datasets/Aa_50.npy'),
               'Aa_100': np.load('./data/BPT_new/datasets/Aa_100.npy'),
               'Aa_150': np.load('./data/BPT_new/datasets/Aa_150.npy'),
               'Aa_200': np.load('./data/BPT_new/datasets/Aa_200.npy')}
    # </editor-fold>

    # Form a list of flags for splitting 'False' scans into the train/valid datasets (3:1 ratio)
    choose_train, choose_valid = ['train'], ['valid']
    choose_dataset = 3 * choose_train + 1 * choose_valid

    # Create the directories for the train/valid/test datasets if they do not yet exist
    Path('./data/BPT_new/train/').mkdir(parents=True, exist_ok=True)
    Path('./data/BPT_new/valid/').mkdir(parents=True, exist_ok=True)
    Path('./data/BPT_new/test/').mkdir(parents=True, exist_ok=True)

    # Create csv writers for spectrum labels and spectrum names for all three datasets, as well as a writer for a
    # global reference to each particle (as all particles will be named using a global counter)
    with open(f'./data/BPT_new/train/train_labels.csv', 'w', newline='') as trainlabelcsv, \
            open(f'./data/BPT_new/train/train_specs.csv', 'w', newline='') as trainspeccsv, \
            open(f'./data/BPT_new/valid/valid_labels.csv', 'w', newline='') as validlabelcsv, \
            open(f'./data/BPT_new/valid/valid_specs.csv', 'w', newline='') as validspeccsv, \
            open(f'./data/BPT_new/test/test_labels.csv', 'w', newline='') as testlabelcsv, \
            open(f'./data/BPT_new/test/test_specs.csv', 'w', newline='') as testspeccsv, \
            open('./data/BPT_new/particle_reference.txt', 'w') as particlereferencewriter:
        trainlabelwriter = csv.writer(trainlabelcsv)
        trainspecwriter = csv.writer(trainspeccsv)
        validlabelwriter = csv.writer(validlabelcsv)
        validspecwriter = csv.writer(validspeccsv)
        testlabelwriter = csv.writer(testlabelcsv)
        testspecwriter = csv.writer(testspeccsv)

        count = 0  # instantiate the global counter for new particle IDs

        # Cycle through all raw .h5 datasets
        # (NOTE: The 'Au NP on 1ML Pd mirror @ 150 mW' dataset is skipped because its files are corrupted(?)... they
        # have infinite loops, which prevents the data from being saved/modified/used)
        filepaths = glob.glob('./data/BPT_new/datasets/*/*.h5')[0:1] + glob.glob('./data/BPT_new/datasets/*/*.h5')[2:]
        for filepath in filepaths:
            with h5py.File(filepath, 'r') as dataset:  # this also works: dataset = h5py.File(filepath, 'r')
                print('--------------------')
                print(filepath, '\n', dataset)

                # If a file contains the 'ParticleScannerScan_1' key, then the data is contained within it,
                # otherwise the data is contained within 'ParticleScannerScan_0'
                if 'ParticleScannerScan_1' in dataset:
                    key = 'ParticleScannerScan_1'
                else:
                    key = 'ParticleScannerScan_0'

                for k0, v0 in dataset[key].items():
                    if 'Particle' not in k0:  # do not include 'Tile' data
                        print('ding')
                        continue

                    for k1, v1 in v0.items():
                        if 'sers' not in k1.lower():  # skip anything except SERS scans
                            continue

                        if v1 is None:  # skip empty keys
                            continue

                        # <editor-fold desc="---=== [+] Create Naming Prefix Based on Experiment Information ===---">
                        # Create the string used to identify the current particle
                        exp_info = filepath.split('/')[-2]  # experimental setup info (e.g. Au NP on Au mirror)
                        if exp_info == 'Au NP on 1ML Pd mirror':
                            exp = 'Ap'
                        elif exp_info == 'Au NP on Au mirror':
                            exp = 'Aa'
                        else:  # exp_info == 'Au_Pd NP on Au mirror'
                            exp = 'APa'
                        # </editor-fold>

                        # Check if data is at this level, or if we need to go deeper
                        if 'kinetic' in k1.lower():
                            # <editor-fold desc="---=== [+] Retrieve Label Information ===---">
                            # Obtain the label information (already have particle number from global counter)
                            try:
                                power = int(v1.attrs['Power'] * 1000)  # convert from mW to uW
                                pico = v1.attrs['Picocavity']
                            except (KeyError, OSError):  # skip if unlabelled/no power info/labelling error
                                continue

                            # wavenumber = v1.attrs['wavelengths']
                            particle_str = 'Particle_' + '0' * (4 - len(str(count))) + str(count)

                            # Place 'True' scans within the testing dataset
                            if pico == 'True':
                                flags = 'test'

                            # Randomly assign all pico == False scans between train/valid
                            else:
                                flags = np.random.choice(choose_dataset)

                            # Create list of labels
                            label = [flags, particle_str, power, pico]
                            # </editor-fold>

                            # <editor-fold desc="---=== [+] Retrieve and Interpolate the Scan ===---">
                            # Obtain the wavenumber information
                            wavenumber = wn_dict[f'{exp}_{power}']

                            # Store the scan in an array ready for preprocessing
                            scan = v1[:]
                            if v1.shape != (500, 1600):  # skip the array if the shape is not (500, 1600)
                                continue

                            if exp == 'APa':  # reverse columns for the 'Au_Pd NP on Au mirror' dataset
                                scan = scan[:, ::-1]

                            # Remove background count
                            scan -= 300

                            # Create interpolation function and interpolate scans into given wavenumber scale
                            scan = interp1d(wavenumber, scan, kind='cubic', fill_value='extrapolate', axis=-1)(
                                wavenumber_scale)

                            # Zero out the regions for each scan without data
                            # (fill_value of 0 could be set for interp1d() but it didn't work? so I manually set it)
                            scan[:, wavenumber_scale < wavenumber[0]] = 0
                            scan[:, wavenumber_scale > wavenumber[-1]] = 0

                            if np.max(scan) == np.min(scan):  # skip empty scans
                                continue

                            # Linearly rescale the data between [0, 1]
                            scan = (scan - np.min(scan)) / (np.max(scan) - np.min(scan))
                            # </editor-fold>

                            # <editor-fold desc="---=== [+] Save Data and Label Information ===---">
                            # Write the particle reference to file
                            particlereferencewriter.write(f'{particle_str} = {exp_info}, {k0}, {k1}, {power}, {pico}\n')
                            count += 1  # increase global counter

                            # Save the interpolated scans as npy files in their respective directories
                            # - Write directories of each scan to respective csv (name_spec is *exact* filepath)
                            # - Write labels to respective csv files
                            if flags == 'train':
                                np.save(f'./data/BPT_new/train/{exp}_{particle_str}_{power}_{pico}.npy', scan)
                                name_spec = [f'./data/BPT_new/train/{exp}_{particle_str}_{power}_{pico}.npy']
                                trainspecwriter.writerow(name_spec)
                                trainlabelwriter.writerow(label)

                            elif flags == 'valid':
                                np.save(f'./data/BPT_new/valid/{exp}_{particle_str}_{power}_{pico}.npy', scan)
                                name_spec = [f'./data/BPT_new/valid/{exp}_{particle_str}_{power}_{pico}.npy']
                                validspecwriter.writerow(name_spec)
                                validlabelwriter.writerow(label)

                            else:
                                np.save(f'./data/BPT_new/test/{exp}_{particle_str}_{power}_{pico}.npy', scan)
                                name_spec = [f'./data/BPT_new/test/{exp}_{particle_str}_{power}_{pico}.npy']
                                testspecwriter.writerow(name_spec)
                                testlabelwriter.writerow(label)
                            # </editor-fold>

                        else:
                            for k2, v2 in v1.items():
                                # <editor-fold desc="---=== [+] Retrieve Label Information ===---">
                                # Obtain the label information (already have particle number from global counter)
                                try:
                                    power = int(v2.attrs['Power'] * 1000)  # convert from mW to uW
                                    pico = v2.attrs['Picocavity']
                                except (KeyError, OSError):  # skip if unlabelled/no power info/labelling error
                                    continue

                                # wavenumber = v1.attrs['wavelengths']
                                particle_str = 'Particle_' + '0' * (4 - len(str(count))) + str(count)

                                # Place 'True' scans within the testing dataset
                                if pico == 'True':
                                    flags = 'test'

                                # Randomly assign all pico == False scans between train/valid
                                else:
                                    flags = np.random.choice(choose_dataset)

                                # Create list of labels
                                label = [flags, particle_str, power, pico]
                                # </editor-fold>

                                # <editor-fold desc="---=== [+] Retrieve and Interpolate the Scan ===---">
                                # Obtain the wavenumber information
                                wavenumber = wn_dict[f'{exp}_{power}']

                                # Store the scan in an array ready for preprocessing
                                scan = v2[:]
                                if v2.shape != (500, 1600):  # skip the array if the shape is not (500, 1600)
                                    continue

                                if exp == 'APa':  # reverse columns for the 'Au_Pd NP on Au mirror' dataset
                                    scan = scan[:, ::-1]

                                # Remove background count
                                scan -= 300

                                # Create interpolation function and interpolate scans into given wavenumber scale
                                scan = interp1d(wavenumber, scan, kind='cubic', fill_value='extrapolate', axis=-1)(
                                    wavenumber_scale)

                                # Zero out the regions for each scan without data
                                # (fill_value of 0 could be set for interp1d() but it didn't work? so I manually set it)
                                scan[:, wavenumber_scale < wavenumber[0]] = 0
                                scan[:, wavenumber_scale > wavenumber[-1]] = 0

                                if np.max(scan) == np.min(scan):  # skip empty scans
                                    continue

                                # Linearly rescale the data between [0, 1]
                                scan = (scan - np.min(scan)) / (np.max(scan) - np.min(scan))
                                # </editor-fold>

                                # <editor-fold desc="---=== [+] Save Data and Label Information ===---">
                                # Write the particle reference to file
                                particlereferencewriter.write(
                                    f'{particle_str} = {exp_info}, {k0}, {k1}, {k2}, {power}, {pico}\n')
                                count += 1  # increase global counter

                                # Save the interpolated scans as npy files in their respective directories
                                # - Write directories of each scan to respective csv (name_spec is *exact* filepath)
                                # - Write labels to respective csv files
                                if flags == 'train':
                                    np.save(f'./data/BPT_new/train/{exp}_{particle_str}_{power}_{pico}.npy', scan)
                                    name_spec = [f'./data/BPT_new/train/{exp}_{particle_str}_{power}_{pico}.npy']
                                    trainspecwriter.writerow(name_spec)
                                    trainlabelwriter.writerow(label)

                                elif flags == 'valid':
                                    np.save(f'./data/BPT_new/valid/{exp}_{particle_str}_{power}_{pico}.npy', scan)
                                    name_spec = [f'./data/BPT_new/valid/{exp}_{particle_str}_{power}_{pico}.npy']
                                    validspecwriter.writerow(name_spec)
                                    validlabelwriter.writerow(label)

                                else:
                                    np.save(f'./data/BPT_new/test/{exp}_{particle_str}_{power}_{pico}.npy', scan)
                                    name_spec = [f'./data/BPT_new/test/{exp}_{particle_str}_{power}_{pico}.npy']
                                    testspecwriter.writerow(name_spec)
                                    testlabelwriter.writerow(label)
                                # </editor-fold>
                # exit()
        # exit()
    return


#######################################################################################################################
# Dataset and scan selection functions
#######################################################################################################################
# <editor-fold desc="---=== [+] Functions to load selected datasets (train/valid/test) ===---">
def load_scan(name):
    data = np.load(name.numpy()[0])
    return data.astype(np.float32)


def load_dataset(params, data_type):
    """ Load in the specified dataset as a sliced tensorflow dataset

    Args:
        params: Dict, The hyperparameter dictionary
        data_type: Str, The chosen dataset. options: train, valid, test
    """
    # Define filepath extension necessary to locate the correct scans
    if params['molecule'] == 'BPT':
        ext = ''
    else:
        ext = f'{params["molecule"]}/'

    with open(f'./data/{ext}{data_type}/{data_type}_specs.csv', 'r', newline='') as speccsv:
        name_spec = list(csv.reader(speccsv, delimiter=','))
        num_spectra = len(name_spec)
        dataset = (tf.data.Dataset.from_tensor_slices(name_spec).shuffle(num_spectra).map(
            lambda name: tf.py_function(load_scan, [name], [tf.float32])).map(
            lambda data: tf.expand_dims(data, axis=-1)))
    return dataset, num_spectra


# </editor-fold>


# <editor-fold desc="---=== [+] Functions to load specific scans (e.g. Particle_1045) ===---">
# Function to convert name_spec entries to respective numpy arrays
def load_scan_eval(name, label):
    data = np.load(name.numpy()[0])
    return data.astype(np.float32), label


# Function to expand the dimensions of each respective numpy array (separate function required to pass on label data)
def expand_eval(data, label):
    data_expand = tf.expand_dims(data, axis=-1)
    return data_expand, label


# Search function for selecting user-specified scans
def select_scan(params, particles=None, powers=None, picos=None, flags=None, exps=None, print_info=False):
    """ Function to create a dataset from user-specified scans

    Args:
        params: Dict, The hyperparameter dictionary
        particles: List of Str, A list of chosen particles. E.g. ['Particle_0506', 'Particle_1045']
        powers: List of Str, A list of chosen powers. E.g. ['50', '100', '150', '200']
        picos: List of Str, A list of chosen picocavity scans. E.g. ['True']
        flags: List of Str, A list of chosen datasets. E.g. ['test', 'valid']
        exps: List of Str or None, Chosen experiment to draw scans from. E.g. 'Aa', 'Ap', 'APa', or None (assumes all)
        print_info: Bool, If True: Displays the information for the retrieved scans, if False: Doesn't
    """
    # Define filepath extension necessary to locate the correct scans
    if params['molecule'] == 'BPT':
        ext = ''
    else:
        ext = f'{params["molecule"]}/'

    # Open spec (full paths to .npy files) and label (tables of labels) csv files for the train/valid/test datasets
    with open(f'./data/{ext}train/train_specs.csv', 'r', newline='') as trainspeccsv, open(
            f'./data/{ext}train/train_labels.csv', 'r', newline='') as trainlabelcsv:
        train_name_spec = list(csv.reader(trainspeccsv, delimiter=','))
        train_name_label = list(csv.reader(trainlabelcsv, delimiter=','))
    with open(f'./data/{ext}valid/valid_specs.csv', 'r', newline='') as validspeccsv, open(
            f'./data/{ext}valid/valid_labels.csv', 'r', newline='') as validlabelcsv:
        valid_name_spec = list(csv.reader(validspeccsv, delimiter=','))
        valid_name_label = list(csv.reader(validlabelcsv, delimiter=','))
    with open(f'./data/{ext}test/test_specs.csv', 'r', newline='') as testspeccsv, open(
            f'./data/{ext}test/test_labels.csv', 'r', newline='') as testlabelcsv:
        test_name_spec = list(csv.reader(testspeccsv, delimiter=','))
        test_name_label = list(csv.reader(testlabelcsv, delimiter=','))

    # Concatenate all train/valid/test lists for both spec and label csv data
    name_spec = train_name_spec + valid_name_spec + test_name_spec
    name_label = train_name_label + valid_name_label + test_name_label

    # Create lists containing identifying information
    allflags = []
    allparticles = []
    allpowers = []
    allpicos = []
    for i in range(len(name_spec)):
        allflags_temp, allparticles_temp, allpowers_temp, allpicos_temp = name_label[i]
        allflags.append(allflags_temp)
        allparticles.append(allparticles_temp)
        allpowers.append(allpowers_temp)
        allpicos.append(allpicos_temp)

    # Any unspecified label (i.e. 'None') selects all of that label
    if flags is None:
        flags = np.unique(allflags)
    particles_all = None
    if particles is None:
        particles = np.unique(allparticles)
        particles_all = '--== All Particles ==--'
    if powers is None:
        powers = np.unique(allpowers)
    if picos is None:
        picos = np.unique(allpicos)
    if exps[0] is None:
        if params['molecule'] == 'BPT':
            exps = ['']
        else:
            exps = ['Aa_', 'Ap_', 'APa_']
    else:  # do some formatting...
        if params['molecule'] != 'BPT':
            for i in range(len(exps)):
                exps[i] += '_'

    # Instantiate lists of chosen name_spec/name_label pairs
    select_specs = []
    select_labels = []

    matches = 0  # Total amount of scans
    n_train, n_valid, n_test = 0, 0, 0  # Amount of scans from each dataset

    # Loop through each selected scan and store if it matches
    for flag in flags:
        for particle in particles:
            for power in powers:
                for pico in picos:
                    for exp in exps:
                        # Create directory to npy file...
                        select_spec_format = f'{exp}{particle}_{power}_{pico}.npy'

                        # ... and store it if it exists within that directory ...
                        select_specs_temp = [s for spec in name_spec for s in spec if select_spec_format in s]

                        if select_specs_temp:
                            if flag not in select_specs_temp[0]:  # ... as long as it exists within chosen dataset
                                continue  # normally done in 'select_spec_format =' but prefix differs (Ap/Aa/APa)
                            matches += 1
                            select_specs.append(select_specs_temp)
                            select_labels_temp = [flag, particle, power, pico]
                            select_labels.append(select_labels_temp)

    for i in select_labels:
        if i[0] == 'train':  # Count the amounts of each dataset belonging to the chosen scans
            n_train += 1
        elif i[0] == 'valid':
            n_valid += 1
        else:
            n_test += 1

    ###################################################################################################################
    # Try to keep this code on one line, as the current TF version produces redundant warnings when lambda functions
    # appear on multiple lines when applied to TF code
    dataset = (tf.data.Dataset.from_tensor_slices((select_specs, select_labels)).map(lambda name, label: tf.py_function(load_scan_eval, [name, label], [tf.float32, tf.string])).map(lambda data, label: tf.py_function(expand_eval, [data, label], [tf.float32, tf.string])))
    ###################################################################################################################

    if print_info:
        # Used for print formatting
        if particles_all is not None:
            particles = particles_all

        print(
            f'\n{matches} matches found for chosen labels:\n  Molecule:  {params["molecule"]}\n  Dataset:   {flags} <-- ({n_test}/{n_train}/{n_valid}) split\n  '
            f'Particles: {particles}\n  Powers:    {powers}\n  Picos:     {picos}')

    return dataset


# </editor-fold>


if __name__ == '__main__':
    # # Create train/valid/test folders in './data' directory from which datasets or specific scans are called
    # bpt_dataset()

    # create_wavenumbers()

    # # Create train/valid/test folders in './data/BPT_new' directory from which datasets or specific scans are called
    # new_bpt_dataset()

    exit()
