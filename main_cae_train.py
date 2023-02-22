"""
@Created : 14/04/2021
@Edited  : 15/01/2023
@Author  : Alex Poppe
@File    : cae_train.py
@Software: Pycharm
@Description:
Trains the CAE to reconstruct the resting state of the BPT SERS spectra
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from pathlib import Path
from nn.models.cae import Autoencoder
from main_cae_data import load_dataset
from utils import hyperparams_setup


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

with tf.device('/device:GPU:0'):
    def train_cae(params):
        """ Train and validate the CAE model using the BPT dataset

        Args:
            params: Dict, The hyperparameter dictionary
        """
        # Define the MSE (i.e. L2) loss function
        mse_loss = tf.keras.losses.MeanSquaredError()

        # Define optimiser function and instantiate models
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
        model = Autoencoder(params)  # Train on the CAE, but...

        encoder = model.layers[0]  # ...save weights and checkpoints to the encoder...
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        encoder_mngr = tf.train.CheckpointManager(encoder_ckpt,
                                                  directory=f'./nn/checkpoints/cae/{params["c_ver"]}/encoder',
                                                  max_to_keep=2)

        decoder = model.layers[1]  # ...and the decoder, so that they can be run separately
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
        decoder_mngr = tf.train.CheckpointManager(decoder_ckpt,
                                                  directory=f'./nn/checkpoints/cae/{params["c_ver"]}/decoder',
                                                  max_to_keep=2)

        # Restore encoder and decoder weights and biases
        encoder_ckpt.restore(encoder_mngr.latest_checkpoint).expect_partial()
        decoder_ckpt.restore(decoder_mngr.latest_checkpoint).expect_partial()
        if encoder_mngr.latest_checkpoint and decoder_mngr.latest_checkpoint:
            print(f"Restored encoder from {encoder_mngr.latest_checkpoint}")
            print(f"Restored decoder from {decoder_mngr.latest_checkpoint}")
            init_epoch = int(encoder_ckpt.save_counter)  # Store epoch value for current training progress
        else:
            init_epoch = 0
            if params['c_record']:
                print('No checkpoint found! Initialising from scratch...')
            else:
                print('Attempting to train/evaluate new model without storing trained values!')
                print('Try setting "c_record True" inside the chosen config')
                print('Or make sure the version name ("c_ver") is spelled correctly\nExiting...')
                exit()

        if params['c_record']:
            # Create summary writers for losses
            train_log_dir = f'./nn/logs/cae/{params["c_ver"]}/train'
            valid_log_dir = f'./nn/logs/cae/{params["c_ver"]}/valid'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        with tf.device('/device:CPU:0'):
            # Load in the validation dataset and its length
            valid_dataset, valid_len = load_dataset(params=params, data_type='valid')

        # Calculate the number of batches per scan based on the batch size (there are 1000 total spectra per scan)
        n_batches = 1000 // params['c_batch_size']

        for epoch in range(init_epoch, params['c_epochs']):
            epoch_train_loss = 0  # Initialise epoch training loss
            epoch_valid_loss = 0  # Initialise epoch validation loss

            # Reload the training dataset (shuffles it)
            with tf.device('/device:CPU:0'):
                train_dataset, train_len = load_dataset(params=params, data_type='train')

            # Print current epoch/epochs remaining
            current_epoch = epoch + 1
            print(f"---=== Epoch {current_epoch}/{params['c_epochs']} ===---")

            # <editor-fold desc="---=== [+] Training ===---">
            # Train the network by stepping through each batch of the dataset
            for train_data in train_dataset.as_numpy_iterator():
                for i in range(n_batches):
                    train_data_temp = train_data[params['c_batch_size'] * i:params['c_batch_size'] * (i + 1)]
                    with tf.GradientTape() as tape:
                        train_pred = model(train_data_temp, training=True)
                        loss = mse_loss(train_data_temp, train_pred)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    epoch_train_loss += loss
            epoch_train_loss /= train_len  # correct for dataset size differences

            if params['c_record']:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_train_loss, step=current_epoch)  # Save training loss every epoch

                # Save current weights and biases
                encoder_mngr.save()
                decoder_mngr.save()
            # </editor-fold>

            # <editor-fold desc="---=== [+] Validation ===---">
            # Calculate and save the epoch validation loss
            for valid_data in valid_dataset.as_numpy_iterator():
                for i in range(n_batches):
                    valid_data_temp = valid_data[params['c_batch_size'] * i:params['c_batch_size'] * (i + 1)]
                    valid_pred = model(valid_data_temp, training=False)
                    epoch_valid_loss += mse_loss(valid_pred, valid_data_temp)
            epoch_valid_loss /= valid_len  # correct for dataset size differences

            if params['c_record']:
                # Save the validation loss for the current epoch
                with valid_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_valid_loss, step=current_epoch)  # Save validation loss every epoch
            # </editor-fold>

        print('\nModel Trained!')

        return


    def fine_tune(params):
        """ Fine-tune the pre-trained CAE model using the new BPT dataset

        Args:
            params: Dict, The hyperparameter dictionary
        """
        # <editor-fold desc="---=== [+] Instantiate Pre-Trained Model, Loss Function, and Checkpoint Managers ===---">
        # Define the MSE (i.e. L2) loss function
        mse_loss = tf.keras.losses.MeanSquaredError()

        # Define optimiser function and instantiate CAE model
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=True)
        model = Autoencoder(params)  # train on the CAE, but...

        # Load the partially-trained model if it exists
        encoder = model.layers[0]  # ...save the trainable parameters to the encoder...
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        encoder_mngr = tf.train.CheckpointManager(encoder_ckpt,
                                                  directory=f'./nn/checkpoints/cae/{params["c_ver"]}/encoder',
                                                  max_to_keep=2)

        decoder = model.layers[1]  # ...and the decoder, so that they can be run separately
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
        decoder_mngr = tf.train.CheckpointManager(decoder_ckpt,
                                                  directory=f'./nn/checkpoints/cae/{params["c_ver"]}/decoder',
                                                  max_to_keep=2)

        # Store filepaths for latest pre-trained model, used to restore model after each iteration of fine-tuning
        encoder_pretrained = encoder_mngr.latest_checkpoint
        decoder_pretrained = decoder_mngr.latest_checkpoint
        if encoder_pretrained and decoder_pretrained:
            # print(f"Restored Encoder from {encoder_pretrained}")
            # print(f"Restored Decoder from {decoder_pretrained}")
            pass
        else:
            print("No checkpoint found for the pre-trained model! Exiting...")
            exit()
        # </editor-fold>

        # Define the filepath to save the RoC curve and AUC values to
        path = f'./analysis/{params["c_ver"]}/{params["c_ver_ft"]}'
        Path(path).mkdir(parents=True, exist_ok=True)

        # Define optimiser function and instantiate the CAE model
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate_ft'], clipnorm=True)
        model = Autoencoder(params)  # train on the whole CAE...

        # Create checkpoints/managers for each part of the model
        encoder = model.layers[0]  # ...but save the trainable parameters to the encoder...
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        encoder_mngr = tf.train.CheckpointManager(
            encoder_ckpt,
            directory=f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/encoder',
            max_to_keep=2)

        decoder = model.layers[1]  # ...and the decoder, so that they can be run separately
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
        decoder_mngr = tf.train.CheckpointManager(
            decoder_ckpt,
            directory=f'./nn/checkpoints/cae/{params["c_ver"]}/{params["c_ver_ft"]}/decoder',
            max_to_keep=2)

        if encoder_mngr.latest_checkpoint and decoder_mngr.latest_checkpoint:
            # Load in the partially-trained fine-tuned model if it exists...
            encoder_ckpt.restore(encoder_mngr.latest_checkpoint).expect_partial()
            decoder_ckpt.restore(decoder_mngr.latest_checkpoint).expect_partial()
            init_epoch = int(encoder_ckpt.save_counter)  # restore last epoch to resume training at correct epoch
            # print(f"Restored Fine-Tuned Encoder from {encoder_mngr.latest_checkpoint}")
            # print(f"Restored Fine-Tuned Decoder from {decoder_mngr.latest_checkpoint}")
            init_epoch -= params['c_epochs']
        else:
            init_epoch = 0
            if params['c_record_ft']:
                # ...Else load in the pre-trained model and start fine-tuning
                print("No fine-tuned checkpoint found! Initialising from the pre-trained model...")
                encoder_ckpt.restore(encoder_pretrained).expect_partial()
                decoder_ckpt.restore(decoder_pretrained).expect_partial()
            else:
                print('Attempting to train/evaluate new fine-tuned model without storing trained values!')
                print('Try setting "c_record_ft True" inside the chosen config')
                print('Or make sure the fine-tuned version name ("c_ver_ft") is spelled correctly\nExiting...')
                exit()

        if params['c_record_ft']:
            # Create summary writer for the training loss (there is no validation loss for CV)
            train_log_dir = f'./nn/logs/cae/{params["c_ver"]}/{params["c_ver_ft"]}/train'
            valid_log_dir = f'./nn/logs/cae/{params["c_ver"]}/{params["c_ver_ft"]}/valid'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

        with tf.device('/device:CPU:0'):
            # Load in the validation dataset and its length
            valid_dataset, valid_len = load_dataset(params=params, data_type='valid')

        for epoch in range(init_epoch, params['c_epochs_ft']):
            epoch_train_loss = 0  # initialise epoch training loss
            epoch_valid_loss = 0  # initialise epoch training accuracy

            # Reload the training dataset (shuffles it)
            with tf.device('/device:CPU:0'):
                train_dataset, train_len = load_dataset(params=params, data_type='train')

            # Print current epoch/epochs remaining
            current_epoch = epoch + 1
            print(f"--== Epoch {current_epoch}/{params['c_epochs_ft']} ==--")

            # <editor-fold desc="---=== [+] Training ===---">
            # Cycle through the training dataset
            for train_data in train_dataset.as_numpy_iterator():
                with tf.GradientTape() as tape:
                    train_pred = model(train_data, training=True)
                    train_loss = mse_loss(train_data, train_pred)
                gradients = tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                epoch_train_loss += train_loss
            epoch_train_loss /= train_len  # correct for dataset size differences

            if params['c_record_ft']:
                # Save the training epoch loss
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_train_loss, step=current_epoch)

                # Save current model weights and biases
                encoder_mngr.save()
                decoder_mngr.save()
            # </editor-fold>

            # <editor-fold desc="---=== [+] Validation ===---">
            # Calculate and save the epoch validation loss
            for valid_data in valid_dataset.as_numpy_iterator():
                valid_pred = model(valid_data, training=False)
                epoch_valid_loss += mse_loss(valid_data, valid_pred)
            epoch_valid_loss /= valid_len  # correct for dataset size differences

            if params['c_record_ft']:
                # Save the validation loss for the current epoch
                with valid_summary_writer.as_default():
                    tf.summary.scalar('loss', epoch_valid_loss, step=current_epoch)  # Save validation loss every epoch
            # </editor-fold>

        return

if __name__ == '__main__':
    # Load chosen hyperparameters
    hyperparams = hyperparams_setup()

    # Train the CAE, storing parameters inside ./nn/checkpoints/cae/c_ver, and logs inside ./nn/logs/cae/c_ver
    train_cae(params=hyperparams)

    # Fine-tune the CAE, storing parameters inside ./nn/checkpoints/cae/c_ver/c_ver_ft, and logs
    # inside ./nn/logs/cae/c_ver/c_ver_ft
    if not (hyperparams['c_record'] and not hyperparams['c_record_ft']):
        fine_tune(params=hyperparams)

    # If you want to view loss curves, then run the following on the command line:
    # tensorboard --logdir="./nn/logs/cae/"
    # NOTE: You may need to alter the preceding directory as necessary.
    # This tensorboard can be view in a browser using the default URL: localhost:6006/
