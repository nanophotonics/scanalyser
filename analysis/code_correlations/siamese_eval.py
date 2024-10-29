"""
@Created : 04/03/2021
@Edited  : 01/07/2022
@Author  : Alex Poppe
@File    : siamese_cnn_eval.py
@Software: Pycharm
@Description:
This code evaluates the given pre-trained or fine-tuned Siamese-CNN model
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import csv
from sklearn.metrics import roc_curve, auc
from itertools import combinations
from main_siamese_data import load_group_test, kfold_cv
from nn.models.siamese_cnn import siamese
from utils import wavenumber_range


# <editor-fold desc="---=== [+] Configure GPU ===---">
def enable_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(f'{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)')
        except RuntimeError as e:
            print(e)


# </editor-fold>
enable_gpu()


def roc_auc(params, finetune=False, k=0):
    """ Generate an RoC (Receiver operating Characteristic) curve with its associated AUC (area under curve) value

    Args:
        params: Dict, The hyperparameter dictionary
        finetune: Bool, If True: Evaluates a fine-tuned model, if False: Evaluates the pre-trained model
        k: Int, The fine-tuned model to evaluate (only used if finetune==True), i.e. k in k-fold cross validation
    """
    if finetune:  # insert extension to the checkpoint filepath to load the current fine-tuned model
        fdir = f'{params["s_ver_ft"]}/iter{k}/'
    else:
        fdir = ''

    # Define optimiser function and instantiate Siamese CNN model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['s_learning_rate'], clipnorm=True)
    model = siamese(params)

    # Data is processed through the entire model but the model is loaded in through each part separately
    scnn = model.layers[0]
    sdense = model.layers[1]

    # Create the model checkpoints/managers
    scnn_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=scnn)
    scnn_mngr = tf.train.CheckpointManager(scnn_ckpt,
                                           directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/{fdir}scnn',
                                           max_to_keep=2)
    sdense_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=sdense)
    sdense_mngr = tf.train.CheckpointManager(sdense_ckpt,
                                             directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/{fdir}sdense',
                                             max_to_keep=2)

    # Load in the trained model
    scnn_ckpt.restore(scnn_mngr.latest_checkpoint).expect_partial()
    sdense_ckpt.restore(sdense_mngr.latest_checkpoint).expect_partial()
    if scnn_mngr.latest_checkpoint and sdense_mngr.latest_checkpoint:
        # print('Evaluating the accuracy of the model...')
        # print(f"CNN: {scnn_mngr.latest_checkpoint}")
        # print(f"FC: {sdense_mngr.latest_checkpoint}")
        pass
    else:
        print('Could not find model inside one or both directories!\nExiting...')
        exit()

    # Instantiate lists to contain all predicted and true labels
    pred_labels = []
    true_labels = []

    # Load in the chosen dataset
    if finetune:
        # Collect the training and testing datasets for the current partition of the fine-tuning dataset
        _, test_data, _, test_labels, _, _ = kfold_cv(params=params, partition=k)
        # Convert to TF dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).shuffle(test_data.shape[0]).batch(
            params['s_batch_size'], drop_remainder=False)

        for data, label in test_dataset.as_numpy_iterator():
            # Extend the labels list
            true_labels.extend(label)

            # Split the data pairs into their constituents - one for each network arm
            data_a = data[:, 0]
            data_b = data[:, 1]

            # Produce the model predictions
            pred = model([data_a, data_b], training=False)

            # Convert the predictions from logits to probabilities, and extend the list
            pred_labels.extend(tf.math.sigmoid(pred).numpy())
    else:
        for data, label, _ in load_group_test(params):
            # Extend the labels to list
            true_labels.extend(label)

            # Split the data into two sets of pairs - one for each network arm
            data_a = data[:, 0]
            data_b = data[:, 1]

            # Produce the model predictions
            pred = model([data_a, data_b], training=False)

            # Convert the predictions from logits to probabilities, and extend the list
            pred_labels.extend(tf.math.sigmoid(pred).numpy())

    # Convert the lists to arrays
    pred_labels = np.array(pred_labels)
    true_labels = np.array(true_labels)

    # Define remaining parts of filepath
    if finetune:  # change extension such that all RoC curves for fine-tuned model are saved to same directory
        fdir1 = f'{params["s_ver_ft"]}/'
        fdir2 = f'_partition{k}'
    else:
        fdir1 = ''
        fdir2 = ''

    # Keep only the values with true labels of 0/1
    ones = true_labels == 1
    zeroes = true_labels == 0
    true_ones = true_labels[ones]
    pred_ones = pred_labels[ones]
    true_zeroes = true_labels[zeroes]
    pred_zeroes = pred_labels[zeroes]
    true_labels = np.concatenate((true_ones, true_zeroes), axis=0)  # combine the two
    pred_labels = np.concatenate((pred_ones, pred_zeroes), axis=0)  # combine the two

    # <editor-fold desc="---=== [+] Plot ROC Curve w/ AUC Value ===---">
    # Calculate the true and false positives rates, and the associated AUC, for each threshold
    fpr, tpr, thresholds = roc_curve(y_true=true_labels, y_score=pred_labels, pos_label=1)
    area = auc(fpr, tpr)

    # Calculate the Youden's J statistic (i.e. the threshold that produces the best balance between TRP and FPR)
    j = thresholds[np.argmax(tpr - fpr)]

    # Produce an ROC curve for the dataset output
    plt.figure(figsize=(8, 6))
    plt.title(f'ROC Curve', fontsize=12)
    plt.plot(fpr, tpr, 'orange', label=f'AUC = {area:.4f}', lw=2)
    plt.plot([0, 1], [0, 1], 'c--', lw=2)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    plt.legend(loc='lower right')
    plt.savefig(f'./analysis/{params["s_ver"]}/{fdir1}roc_curve{fdir2}.png')
    plt.close()
    # </editor-fold>

    # Calculate the accuracy metrics, store in a list, then output later
    tp = np.count_nonzero(pred_labels[np.argwhere(true_labels == 1)] >= 0.5)  # true positive
    fn = np.count_nonzero(pred_labels[np.argwhere(true_labels == 1)] < 0.5)  # false negative
    tn = np.count_nonzero(pred_labels[np.argwhere(true_labels == 0)] < 0.5)  # true negative
    fp = np.count_nonzero(pred_labels[np.argwhere(true_labels == 0)] >= 0.5)  # false positive
    metrics = [tp, fn, tn, fp, area, j]  # also store the AUC and J-statistic values

    return metrics


def model_performance(params, metrics, finetune=False):
    """ Write calculated model performance metrics to file. The performance metrics are: accuracy, precision,
    sensitivity, specificity, and F1-score.

    Args:
        params: Dict, The hyperparameter dictionary
        metrics: List, Performance metrics (TP, FN, TN, FP)
        finetune: Bool, If True: Evaluates a fine-tuned model, if False: Evaluates the pre-trained model
    """
    # Define remaining parts of filepath
    if finetune:  # change extension such that all RoC curves for fine-tuned model are saved to same directory
        fdir = f'{params["s_ver_ft"]}/average_'

        # rearrange input metrics list
        metrics = np.sum(np.array(metrics), axis=0)  # sum each tp/fn/tn/fp value from each model iteration

        # Average the AUC by the number of partitions (i.e. the k-value in k-fold cross validation)
        metrics[-2] /= params["s_kfold"]

        # Convert back to list for unpacking
        metrics = metrics.tolist()

    else:
        fdir = ''

    # Unpack individual values from metrics list
    tp, fn, tn, fp, area, _ = metrics  # (do not need threshold value [i.e. last value])

    # Calculate the normalised accuracy
    accuracy = (tp + tn) / (tp + fn + tn + fp)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)  # i.e. how well the model identifies positives from all positive samples (aka recall)
    specificity = tn / (tn + fp)  # i.e. how well the model identifies negatives from all negative samples
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)  # i.e. harmonic mean of precision and recall

    # Write accuracy metrics to file
    with open(f'./analysis/{params["s_ver"]}/{fdir}performance.txt', 'w') as w:
        w.write(f'Accuracy = {100 * accuracy:.2f}%\n')
        w.write(f'Precision = {100 * precision:.2f}%\n')
        w.write(f'Sensitivity (TPR/Recall) = {100 * sensitivity:.2f}%\n')
        w.write(f'Specificity (1 - FPR) = {100 * specificity:.2f}%\n')
        w.write(f'F1-Score = {100 * f1:.2f}%\n')
        w.write(f'Area Under Curve (AUC) = {area:.4f}')

    return


def correlation_heatmap(params, finetune=False, k=0, threshold=None):
    """ Create a heatmap and save a csv of wavenumber correlations via correlation predictions from a fine-tuned
    Siamese-CNN. The data is defined by every time a correlation change occurs

    Args:
        params: Dict, The hyperparameter dictionary
        finetune: Bool, If True: Evaluates a fine-tuned model, if False: Evaluates the pre-trained model
        k: Int, The model iteration (only used if finetune == True)
        threshold: Float or None, The threshold at or above which a prediction is set to 1, rather than 0 (If Float:
            uses Youden's J statistic found from ROC curve [i.e. optimal threshold value]; if None: defaults to 0.5)
    """
    # Load in the group_test dataset, and the associated 'unsupervised labels'
    data = np.load(f'./data/snippets/{params["c_ver"]}/real_pairs/test_images.npy')
    unsupervised_labels = np.load(f'./data/snippets/{params["c_ver"]}/real_pairs/test_labels_unsupervised.npy')

    # Define filepath extensions
    if finetune:
        fdir1 = f'{params["s_ver_ft"]}/'
        fdir2 = f'iter{k}/'
    else:
        fdir1 = ''
        fdir2 = ''

    # <editor-fold desc="---=== [+] Retrieve Valid Data Pairs ===---">
    # Retrieve the names of all unique particles
    particles = np.unique(unsupervised_labels[:, 4])

    # Instantiate lists to contain all valid combinations of sub-image pairs and their associated unsupervised labels
    pairs = []
    labels = []

    # Loop through the data, one particle at a time
    for particle in particles:
        # Retrieve the data and unsupervised label information for the current particle
        particle_idxs = np.argwhere(unsupervised_labels[:, 4] == particle).squeeze()
        particle_data = data[particle_idxs]
        particle_unsupervised = unsupervised_labels[particle_idxs]

        # Retrieve the indices of each unique ID
        id_diff = np.diff(particle_unsupervised[:, 0].astype('int'))
        new_ids = np.argwhere(id_diff != 0).squeeze() + 1  # +1 adjusts the indexing
        if new_ids.ndim == 0:  # convert to a list if int64 to become iterable for loop
            new_ids = [new_ids]

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
            # Retrieve the next group
            if j == len(splits) - 1:
                group_data = particle_data[splits[j]:]
                group_unsupervised = particle_unsupervised[splits[j]:]
            else:
                group_data = particle_data[splits[j]:splits[j + 1]]
                group_unsupervised = particle_unsupervised[splits[j]:splits[j + 1]]

            # Cycle through all combinations of pairs at each time step
            subs = np.unique(group_unsupervised[:, 1])
            for sub in subs:
                timestep_idx = np.argwhere(group_unsupervised[:, 1] == sub).squeeze()
                sub_data = group_data[timestep_idx]
                sub_unsupervised = group_unsupervised[timestep_idx]

                # Create every combination of pairs for the current sub-images
                idx = np.arange(sub_data.shape[0])
                combs = np.array(list(combinations(idx, 2)))
                combs_data = sub_data[combs]
                combs_unsupervised = sub_unsupervised[combs]

                pairs.extend(combs_data)
                labels.extend(combs_unsupervised)

    # Convert to arrays
    pairs = np.array(pairs)
    labels = np.array(labels)
    # </editor-fold>

    # Instantiate a list to contain sub-lists of each Track index (e.g. [[0, 9, 15, 23], [3, 11, 17, 20], ...])
    track_idxs = []

    # Loop through each unique particle in the dataset
    for particle in particles:
        # Retrieve all pairs associated with the current particle
        particle_idxs = np.argwhere(labels[:, 0, 4] == particle).squeeze()
        particle_labels = labels[particle_idxs]

        # Find every unique combination of tracks, in terms of pixel positions
        if particle_labels.ndim == 2:
            track_idxs.append([particle_idxs.tolist()])
        else:
            all_pairs = np.unique(np.sort(particle_labels[:, :, 2]), axis=0)  # sort else [A, B] & [B, A] is possible

            # Loop through each combination (i.e. each Track pair set)
            for pair in all_pairs:
                pair_idxs = np.argwhere(np.logical_and(np.sort(particle_labels[:, :, 2])[:, 0] == pair[0],
                                                       np.sort(particle_labels[:, :, 2])[:, 1] == pair[1])).squeeze()
                track_idxs.append(list(particle_idxs[pair_idxs]))

    # <editor-fold desc="---=== [+] Load Model & Make Correlation Predictions ===---">
    # Define optimiser function and instantiate Siamese CNN model
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['s_learning_rate'], clipnorm=True)
    model = siamese(params)

    # Data is processed through the entire model but the model is loaded in through each part separately
    scnn = model.layers[0]
    sdense = model.layers[1]

    # Create the model checkpoints/managers
    scnn_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=scnn)
    scnn_mngr = tf.train.CheckpointManager(
        scnn_ckpt,
        directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/{fdir1}{fdir2}scnn',
        max_to_keep=2)
    sdense_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=sdense)
    sdense_mngr = tf.train.CheckpointManager(
        sdense_ckpt,
        directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/{fdir1}{fdir2}sdense',
        max_to_keep=2)

    # Load in the trained model
    scnn_ckpt.restore(scnn_mngr.latest_checkpoint).expect_partial()
    sdense_ckpt.restore(sdense_mngr.latest_checkpoint).expect_partial()
    if scnn_mngr.latest_checkpoint and sdense_mngr.latest_checkpoint:
        # print(f"Restored CNN arms: {scnn_mngr.latest_checkpoint}")
        # print(f"Restored FC layer: {sdense_mngr.latest_checkpoint}")
        pass
    else:
        print('Please provide both parts of the model!\nExiting...')
        exit()

    # Instantiate a list of correlation predictions
    preds = []

    # Create a tensorflow dataset to model it easier to input into the model
    dataset = tf.data.Dataset.from_tensor_slices((pairs, labels)).batch(params["s_batch_size"], drop_remainder=False)

    # Load in the chosen dataset
    for data, label in dataset.as_numpy_iterator():  # use defaults
        # Split the data into two sets of pairs - one for each network arm
        data_a = data[:, 0]
        data_b = data[:, 1]

        # Produce the model predictions
        pred = model([data_a, data_b], training=False)

        # Convert the predictions from logits to binary labels, and extend the list
        probs = tf.math.sigmoid(pred).numpy().squeeze()
        preds.extend(np.where(probs >= threshold, 1, 0))  # i.e. if >= thresh then 1, else 0

    # Convert to array
    preds = np.array(preds)
    # </editor-fold>

    # Define the wavenumber range
    wn_range = wavenumber_range()

    # Instantiate lists to store x-y data for each pair, stored in separate lists based on correlation value
    pos_data = []
    neg_data = []

    # Instantiate a csv file which will contain the mean wavenumber positions for each Track in a pair, as well
    # as the start and stop time steps, and their corresponding correlation value
    with open(f'./analysis/{params["s_ver"]}/{fdir1}correlation_change.csv', 'w') as outcsv:
        writer = csv.writer(outcsv)
        # Write a header
        writer.writerow(['wn_1', 'wn_2', 'start', 'stop', 'corr'])

        # Cycle through all snippet pairs
        for idx1 in track_idxs:
            # Retrieve the predictions and labels associated with the current track
            current_preds = preds[idx1]
            current_labels = labels[idx1]

            # Find the indices of each correlation change
            switch_idxs = np.argwhere(np.diff(current_preds) != 0).squeeze() + 1

            # Convert to array if there is specifically one correlation change
            if isinstance(switch_idxs, np.integer):
                switch_idxs = np.array([switch_idxs])

            # Retrieve the last indices for each of the correlation windows
            end_idxs = np.concatenate((switch_idxs - 1, np.array([len(current_preds) - 1])), axis=0)

            # Add zero to the start of the switch indices
            switch_idxs = np.insert(switch_idxs, 0, 0)

            # Increment the affinity matrices with the first pair at each correlation change
            for idx2, idx3 in zip(switch_idxs, end_idxs):
                # Write the coordinates of each 'correlation set' and the associated correlation to file
                writer.writerow([f'{wn_range[current_labels[switch_idxs[0], 0, 2].astype(int)]:.2f}',  # wn_1
                                 f'{wn_range[current_labels[switch_idxs[0], 1, 2].astype(int)]:.2f}',  # wn_2
                                 int(current_labels[idx2, 0, 3]),  # start
                                 int(current_labels[idx3, 0, 3]) + 99,  # end
                                 current_preds[idx2]])  # correlation value

                # Convert wavenumbers (in pixels) from strings to floats
                wn_1 = current_labels[idx2, 0, 2].astype(int)
                wn_2 = current_labels[idx2, 1, 2].astype(int)

                # Append [wn_1, wn_2, correlation] to data dictioanary
                # - The larger wavenumber goes first (places all data points in the lower triangle)
                if wn_1 >= wn_2:
                    if current_preds[idx2] == 1:
                        pos_data.append([wn_1, wn_2])
                    else:
                        neg_data.append([wn_1, wn_2])
                else:
                    if current_preds[idx2] == 1:
                        pos_data.append([wn_2, wn_1])
                    else:
                        neg_data.append([wn_2, wn_1])

    # <editor-fold desc="---=== [+] Define Tick Locations and Values for Figure of Affinity Matrix ===---">
    # Define the wavenumber range used for the figure (i.e. 300, 400, ..., 1600)
    wavenumber_samples = np.arange(np.ceil(wn_range[0] / 100) * 100,
                                   np.ceil(wn_range[-1] / 100) * 100,
                                   100).astype('int')

    # Instantiate a list to contain the tick locations of the wavenumber range
    ticks = []
    for sample in wavenumber_samples:
        # Subtract the current sample wavenumber from the wavenumber range, and find the two closest wavenumbers
        lr = np.abs(wn_range - sample)
        a, b = np.partition(lr, 1)[:2]  # find the smallest two values

        # Find the indices of the two closest wavenumbers
        left = np.argwhere(lr == a).squeeze()
        right = np.argwhere(lr == b).squeeze()

        # Calculate the position of the sample wavenumber in pixel-scale
        # Parts of this equation...
        # a / (wn_r - wn_l): The fractional distance of the sample between the two closest wavenumbers
        # * (right - left): Define the fraction on the new scale
        # + left: Add back on the offset (originally removed via '- wn_l')
        ticks.append(a / (wn_range[right] - wn_range[left]) * (right - left) + left)
    # </editor-fold>

    # Convert to arrays
    pos_data = np.array(pos_data)
    neg_data = np.array(neg_data)

    # Plot affinity matrix
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Overlaid correlations
    ax.set_title(f'Peak Correlations (iteration {k})', fontsize=12)

    # Plot data (+ve = circles, -ve = hollow diamonds)
    ax.plot(pos_data[:, 0], pos_data[:, 1], 'ro',
            markersize=8, markeredgewidth=3, markerfacecolor='none',
            alpha=0.7, zorder=100)
    ax.plot(neg_data[:, 0], neg_data[:, 1], 'bd',
            markersize=8, markeredgewidth=3, markerfacecolor='none',
            alpha=0.7, zorder=1000)

    # Plot axis ticks/labels and gridlines
    ax.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
    ax.set_xticks(ticks)
    ax.set_xticklabels(wavenumber_samples)
    ax.set_ylabel('Wavenumber (cm^-1)', fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(wavenumber_samples)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(zorder=1)

    # Plot diagonal line
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.925)

    plt.savefig(f'./analysis/{params["s_ver"]}/{fdir1}peak_correlations.png')

    return
