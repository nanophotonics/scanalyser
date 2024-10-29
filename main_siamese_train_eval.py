"""
@Created : 08/06/2021
@Edited  : 10/06/2022
@Author  : Alex Poppe
@File    : siamese_train.py
@Software: Pycharm
@Description:
Pre-trains and fine-tunes the Siamese-CNN to predict peak correlations on Groups extracted from the BPT SERS spectra
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from pathlib import Path
from main_siamese_data import generate_pairs, load_inference, kfold_cv
from analysis.code_correlations.siamese_eval import roc_auc, model_performance, correlation_heatmap
from nn.models.siamese_cnn import siamese
from utils import hyperparams_setup


# <editor-fold desc="---=== [+] Configure GPU ===---">
def enable_gpu(memory_limit=None):
    """ Enables the GPU

    Args:
        memory_limit: Int or None, If Int: Sets a memory cap on a virtual GPU, If None: Sets memory growth to be True
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_limit is None:
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    tf.config.experimental.set_virtual_device_configuration(gpu, [
                        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(f'{len(gpus)} Physical GPU(s), {len(logical_gpus)} Logical GPU(s)')
        except RuntimeError as e:
            print(e)


# </editor-fold>
enable_gpu(memory_limit=None)

with tf.device('/device:GPU:0'):
    def pre_train(params, save_figs=False):
        """ Train and validate the Siamese-CNN model using synthesised Track 'self-pairs'

        Args:
            params: Dict, The hyperparameter dictionary
            save_figs: Bool, If True: Saves RoC curve to file
        """
        # <editor-fold desc="---=== [+] Instantiate Model, Loss Function, and Checkpoint Managers ===---">
        # Define the loss function with labels as logits (i.e. no activation function on the output layer)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Define optimiser function and instantiate models
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['s_learning_rate'], clipnorm=True)
        model = siamese(params)  # train on the whole siamese CNN, but...

        # Load the partially-trained model if it exists
        scnn = model.layers[0]  # ...save the trainable parameters to the CNN...
        scnn_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=scnn)
        scnn_mngr = tf.train.CheckpointManager(scnn_ckpt,
                                               directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/scnn',
                                               max_to_keep=2)

        sdense = model.layers[1]  # ...and the FC layer, so that they can be run separately
        sdense_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=sdense)
        sdense_mngr = tf.train.CheckpointManager(sdense_ckpt,
                                                 directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/sdense',
                                                 max_to_keep=2)

        # Restore CNN and FC weights and biases
        scnn_ckpt.restore(scnn_mngr.latest_checkpoint).expect_partial()
        sdense_ckpt.restore(sdense_mngr.latest_checkpoint).expect_partial()
        if scnn_mngr.latest_checkpoint and sdense_mngr.latest_checkpoint:
            # print(f"Restored CNN from {scnn_mngr.latest_checkpoint}")
            # print(f"Restored FC layer from {sdense_mngr.latest_checkpoint}")
            init_epoch = int(scnn_ckpt.save_counter)  # Store epoch value for current training progress
        else:
            init_epoch = 0
            if params['s_record']:
                print('No checkpoint found! Initialising from scratch...')
            else:
                print('Attempting to train/evaluate new pre-trained model without storing trained values!')
                print('Try setting "s_record True" inside the chosen config')
                print('Or make sure the pre-trained version name ("s_ver") is spelled correctly\nExiting...')
                exit()

        if params['s_record']:
            # Create summary writers for losses
            train_log_dir = f'./nn/logs/siamese_cnn/{params["s_ver"]}/train'
            valid_log_dir = f'./nn/logs/siamese_cnn/{params["s_ver"]}/valid'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        # </editor-fold>

        with tf.device('/device:CPU:0'):
            # Load in the validation dataset
            valid_dataset = load_inference(params, dataset='valid')

        for epoch in range(init_epoch, params['s_epochs']):
            epoch_train_loss = 0  # initialise epoch training loss
            epoch_valid_loss = 0  # initialise epoch validation loss

            # Print current epoch/epochs remaining
            current_epoch = epoch + 1
            print(f"--== Epoch {current_epoch}/{params['s_epochs']} ==--")

            if current_epoch % 10 == 0:  # calculate accuracies every 10 epochs for computational speed
                epoch_train_acc = 0  # initialise epoch training accuracy
                epoch_valid_acc = 0  # initialise epoch validation accuracy

            # <editor-fold desc="---=== [+] Training ===---">
            # Train the network by stepping through each batch of the dataset
            for i, (train_data, train_labels) in enumerate(generate_pairs(params)):
                # Split the data pairs into their constituents - one for each network arm
                train_data_a = train_data[:, 0]
                train_data_b = train_data[:, 1]
                with tf.GradientTape() as tape:
                    train_pred = model([train_data_a, train_data_b], training=True)
                    train_loss = bce(train_labels, train_pred)
                gradients = tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_train_loss += train_loss

                if current_epoch % 10 == 0:  # calculate training accuracy every 10 epochs for computational speed
                    # Convert the training prediction logits to fit a probability distribution
                    train_sigmoid = tf.math.sigmoid(train_pred).numpy()

                    # Calculate the training accuracy
                    tp = np.count_nonzero(train_sigmoid[np.argwhere(train_labels[:, 0] == 1)] >= 0.5)  # true positive
                    fn = np.count_nonzero(train_sigmoid[np.argwhere(train_labels[:, 0] == 1)] < 0.5)  # false negative
                    tn = np.count_nonzero(train_sigmoid[np.argwhere(train_labels[:, 0] == 0)] < 0.5)  # true negative
                    fp = train_labels.shape[0] - tp - fn - tn  # false negative
                    epoch_train_acc += (tp + tn) / (tp + fn + tn + fp)  # equation for accuracy

            epoch_train_loss /= (i + 1)  # correct for dataset size differences
            if current_epoch % 10 == 0:
                epoch_train_acc /= (i + 1)

            if params['s_record']:
                # Save the training loss and accuracy for the current epoch
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_train_loss, step=current_epoch)
                    if current_epoch % 10 == 0:
                        tf.summary.scalar('acc', epoch_train_acc, step=current_epoch)

                # Save current weights and biases
                scnn_mngr.save()
                sdense_mngr.save()
            # </editor-fold>

            # <editor-fold desc="---=== [+] Validation ===---">
            # Cycle through the training dataset
            for i, (valid_data, valid_labels) in enumerate(valid_dataset.as_numpy_iterator()):
                # Split the data pairs into their constituents - one for each network arm
                valid_data_a = valid_data[:, 0]
                valid_data_b = valid_data[:, 1]
                valid_pred = model([valid_data_a, valid_data_b], training=False)
                epoch_valid_loss += bce(valid_labels, valid_pred)

                if current_epoch % 10 == 0:
                    epoch_valid_acc = 0  # initialise epoch validation accuracy

                    # Convert the validation prediction logits to fit a probability distribution
                    valid_sigmoid = tf.math.sigmoid(valid_pred).numpy()

                    # Calculate the validation accuracy
                    tp = np.count_nonzero(
                        valid_sigmoid[np.argwhere(valid_labels[:, 0] == 1)] >= 0.5)  # true positive
                    fn = np.count_nonzero(
                        valid_sigmoid[np.argwhere(valid_labels[:, 0] == 1)] < 0.5)  # false negative
                    tn = np.count_nonzero(
                        valid_sigmoid[np.argwhere(valid_labels[:, 0] == 0)] < 0.5)  # true negative
                    fp = valid_labels.shape[0] - tp - fn - tn  # false negative (quicker)
                    epoch_valid_acc += (tp + tn) / (tp + fn + tn + fp)  # equation for accuracy

            epoch_valid_loss /= (i + 1)  # correct for dataset size differences
            if current_epoch % 10 == 0:
                epoch_valid_acc /= (i + 1)

            if params['s_record']:
                # Save the validation loss and accuracy for the current epoch
                with valid_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_valid_loss, step=current_epoch)
                    if current_epoch % 10 == 0:
                        tf.summary.scalar('acc', epoch_valid_acc, step=current_epoch)
            # </editor-fold>

        if save_figs:
            # Create the default filepath if it does not already exist
            path = f'./analysis/{params["s_ver"]}/'
            Path(path).mkdir(parents=True, exist_ok=True)

            print('>> Generating ROC curves of pre-trained model')
            metrics = roc_auc(params)

            print('>> Calculating average performance of pre-trained model')
            model_performance(params, metrics)

            # Retrieve the Youden's J statistic value (i.e. best threshold)
            j = metrics[-1]

            print('>> Generating correlation heatmap for pre-trained model')
            correlation_heatmap(params, threshold=j)

        return


    def fine_tune(params, save_figs=False):
        """ Fine-tune the pre-trained Siamese-CNN model using manually labelled Track pairs

        Args:
            params: Dict, The hyperparameter dictionary
            save_figs: Bool, If True: Saves RoC curve to file
        """
        # <editor-fold desc="---=== [+] Instantiate Pre-Trained Model, Loss Function, and Checkpoint Managers ===---">
        # Define the loss function with labels as logits (i.e. no activation function on the output layer)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # Define optimiser function and instantiate Siamese CNN model
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['s_learning_rate'], clipnorm=True)
        model = siamese(params)  # train on the whole siamese CNN...

        # Load the partially-trained model if it exists
        scnn = model.layers[0]  # ...but save the trainable parameters to the CNN...
        scnn_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=scnn)
        scnn_mngr = tf.train.CheckpointManager(
            scnn_ckpt,
            directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/scnn',
            max_to_keep=2)

        sdense = model.layers[1]  # ...and the FC layer, so that they can be run separately
        sdense_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=sdense)
        sdense_mngr = tf.train.CheckpointManager(
            sdense_ckpt,
            directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/sdense',
            max_to_keep=2)

        # Store filepaths for latest pre-trained model, used to restore model after each iteration of fine-tuning
        scnn_pretrained = scnn_mngr.latest_checkpoint
        sdense_pretrained = sdense_mngr.latest_checkpoint
        if scnn_pretrained and sdense_pretrained:
            # print(f"Restored Siamese CNN from {scnn_pretrained}")
            # print(f"Restored FC layer from {sdense_pretrained}")
            pass
        else:
            print("No checkpoint found for the pre-trained model! Exiting...")
            exit()
        # </editor-fold>

        # Define the filepath to save the RoC curve and AUC values to
        path = f'./analysis/{params["s_ver"]}/{params["s_ver_ft"]}'
        Path(path).mkdir(parents=True, exist_ok=True)

        # Fine-tune one model for each partition in the k-fold cross validation process
        for k in range(params['s_kfold']):
            # # Print current iteration/iterations remaining
            # print(f"---=== Iteration {k}/{params['s_kfold'] - 1} ===---")

            # Define optimiser function and instantiate Siamese CNN model
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['s_learning_rate_ft'], clipnorm=True)
            model = siamese(params)  # train on the whole siamese CNN...

            # Create checkpoints/managers for each part of the model
            scnn = model.layers[0]  # ...but save the trainable parameters to the CNN...
            scnn_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=scnn)
            scnn_mngr = tf.train.CheckpointManager(
                scnn_ckpt,
                directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/{params["s_ver_ft"]}/iter{k}/scnn',
                max_to_keep=2)

            sdense = model.layers[1]  # ...and the FC layer, so that they can be run separately
            sdense_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=sdense)
            sdense_mngr = tf.train.CheckpointManager(
                sdense_ckpt,
                directory=f'./nn/checkpoints/siamese_cnn/{params["s_ver"]}/{params["s_ver_ft"]}/iter{k}/sdense',
                max_to_keep=2)

            if scnn_mngr.latest_checkpoint and sdense_mngr.latest_checkpoint:
                # Load in the partially-trained fine-tuned model if it exists...
                scnn_ckpt.restore(scnn_mngr.latest_checkpoint).expect_partial()
                sdense_ckpt.restore(sdense_mngr.latest_checkpoint).expect_partial()
                init_epoch = int(scnn_ckpt.save_counter)  # restore last epoch to resume training at correct epoch
                # print(f"Restored Fine-Tuned CNN from {scnn_mngr.latest_checkpoint}")
                # print(f"Restored Fine-Tuned FC layer from {sdense_mngr.latest_checkpoint}")
                if init_epoch == params['s_epochs'] + params['s_epochs_ft']:
                    # print(f"Model 'iter{k}' is already trained. Skipping...")
                    continue
                init_epoch -= 1000
            else:
                init_epoch = 0
                if params['s_record_ft']:
                    # ...Else load in the pre-trained model and start fine-tuning
                    print("No fine-tuned checkpoint found! Initialising from the pre-trained model...")
                    scnn_ckpt.restore(scnn_pretrained).expect_partial()
                    sdense_ckpt.restore(sdense_pretrained).expect_partial()
                else:
                    print('Attempting to train/evaluate new fine-tuned model without storing trained values!')
                    print('Try setting "s_record_ft True" inside the chosen config')
                    print('Or make sure the fine-tuned version name ("s_ver_ft") is spelled correctly\nExiting...')
                    exit()

            if params['s_record_ft']:
                # Create summary writer for the training loss (there is no validation loss for CV)
                train_log_dir = f'./nn/logs/siamese_cnn/{params["s_ver"]}/{params["s_ver_ft"]}/iter{k}/train'
                train_summary_writer = tf.summary.create_file_writer(train_log_dir)

            # Collect the training and testing datasets for the current partition of the fine-tuning dataset
            train_data, _, train_labels, _, _, _ = kfold_cv(params=params, partition=k)

            # Convert the arrays to TF datasets
            # (Note: remainders are not dropped because instance norm is being used in the model)
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(
                train_data.shape[0]).batch(params['s_batch_size'], drop_remainder=False)

            for epoch in range(init_epoch, params['s_epochs_ft']):
                # Print current epoch/epochs remaining
                current_epoch = epoch + 1
                print(f"--== Epoch {current_epoch}/{params['s_epochs_ft']} ==--")

                # <editor-fold desc="---=== [+] Training ===---">
                epoch_train_loss = 0  # initialise epoch training loss
                epoch_train_acc = 0  # initialise epoch training accuracy
                # Cycle through the training dataset
                for i, (train_data, train_label) in enumerate(train_dataset.as_numpy_iterator()):
                    # Split the data pairs into their constituents - one for each network arm
                    train_data_a = train_data[:, 0]
                    train_data_b = train_data[:, 1]
                    with tf.GradientTape() as tape:
                        train_pred = model([train_data_a, train_data_b], training=True)
                        train_loss = bce(train_label, train_pred)
                    gradients = tape.gradient(train_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    epoch_train_loss += train_loss

                    # Convert the training prediction logits to fit a probability distribution
                    train_sigmoid = tf.math.sigmoid(train_pred).numpy()

                    # Calculate the training accuracy
                    tp = np.count_nonzero(
                        train_sigmoid[np.argwhere(train_label[:, 0] == 1)] >= 0.5)  # true positive
                    fn = np.count_nonzero(
                        train_sigmoid[np.argwhere(train_label[:, 0] == 1)] < 0.5)  # false negative
                    tn = np.count_nonzero(train_sigmoid[np.argwhere(train_label[:, 0] == 0)] < 0.5)  # true negative
                    fp = train_label.shape[0] - tp - fn - tn  # false negative (quicker)
                    epoch_train_acc += (tp + tn) / (tp + fn + tn + fp)  # equation for accuracy

                # Correct for dataset size differences
                epoch_train_loss /= (i + 1)
                epoch_train_acc /= (i + 1)

                if params['s_record_ft']:
                    # Save the training epoch loss
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', epoch_train_loss, step=current_epoch)
                        tf.summary.scalar('acc', epoch_train_acc, step=current_epoch)

                    # Save current Siamese CNN model weights and biases
                    scnn_mngr.save()
                    sdense_mngr.save()
                # </editor-fold>

            print(f'\n>> Fine-tuning iteration {k} complete!')

        if save_figs:
            # Create the default filepath if it does not already exist
            path = f'./analysis/{params["s_ver"]}/{params["s_ver_ft"]}/'
            Path(path).mkdir(parents=True, exist_ok=True)

            print('>> Generating ROC curves of fine-tuned models')
            metrics = []
            for k in range(params['s_kfold']):
                metrics.append(roc_auc(params=params, finetune=True, k=k))

            print('>> Calculating average performance of fine-tuned models')
            model_performance(params, metrics, finetune=True)

            # Fine the partition with the best AUC score
            k = np.argmax(np.array(metrics)[:, -2])

            # Retrieve the Youden's J statistic value for the best partition (i.e. best threshold)
            j = metrics[k][-1]

            print('>> Generating correlation heatmap for best fine-tuned model')
            correlation_heatmap(params, finetune=True, k=k, threshold=j)

        return

if __name__ == '__main__':
    # Load chosen hyperparameters
    hyperparams = hyperparams_setup()

    # Pre-train the Siamese-CNN, storing parameters inside ./nn/checkpoints/siamese_cnn/s_ver, and logs inside
    # ./nn/logs/siamese_cnn/s_ver
    # (NOTE: Setting save_figs=True saves 4 files to ./analysis/s_ver/)
    pre_train(params=hyperparams, save_figs=True)

    # Fine-tune the Siamese-CNN, storing parameters inside ./nn/checkpoints/siamese_cnn/s_ver/s_ver_ft, and logs
    # inside ./nn/logs/siamese_cnn/s_ver/s_ver_ft
    # (NOTE: Setting save_figs=True saves k figures, plus three more files to ./analysis/s_ver/s_ver_ft)
    if not (hyperparams['s_record'] and not hyperparams['s_record_ft']):
        fine_tune(params=hyperparams, save_figs=True)

    # If you want to view loss curves for both pre-training and fine-tuning, then run the following on the command line:
    # tensorboard --logdir="./nn/logs/siamese/"
    # NOTE: You may need to alter the preceding directory as necessary.
    # This tensorboard can be view in a browser using the default URL: localhost:6006/
