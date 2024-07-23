import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()
DATA_PATH = os.getenv("PAUL_DATA_PATH")
DATA_NAME = os.getenv("PAUL_DATA_NAME")
CHECKPOINT_PATH = DATA_PATH

SEED = int(os.getenv("SEED", "1234"))

import time
import tensorflow as tf
import numpy as np
from pathlib import Path
from nn.models.cae_lambda import Autoencoder
from main_cae_data import load_dataset
from utils import hyperparams_setup

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Verify GPUs
def enable_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(len(gpus), "Physical GPUs,")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

enable_gpu()

tf.random.set_seed(SEED)
np.random.seed(SEED)
seed = SEED

def train(params, checkpoint_interval=1, gpus_to_use=None, max_to_keep=2):
    """ Train and validate the CAE model using the BPT dataset

    Args:
        params: Dict, The hyperparameter dictionary;
        checkpoint_interval: int, The step between saved checkpoints;
        gpus_to_use: list, The list of gpus (devices) to use for the training. Example: ["GPU:0", "GPU:1"].
        max_to_keep: int, the max number of checkpoints to keep;
    """
    
    strategy = tf.distribute.MirroredStrategy(gpus_to_use)

    print(f"{strategy.num_replicas_in_sync} GPUs in the strategy")

    global_batch_size = params['c_batch_size'] * strategy.num_replicas_in_sync

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    valid_dataset, _ = load_dataset(params=params, data_type='valid', path=fr"{DATA_PATH}/data/valid_clean/valid_specs.csv")
    valid_len = sum(1 for _ in valid_dataset.unbatch())

    train_dataset, _ = load_dataset(params=params, data_type='train', path=fr"{DATA_PATH}/data/train_clean/train_specs.csv")
    train_len = sum(1 for _ in train_dataset.unbatch())

    valid_batch_size = 2 ** int(np.log2(valid_len)) # to better leverage the GPU

    if valid_batch_size < global_batch_size:
        print(f"WARNING: The validation dataset is smaller than the batch size! ({valid_len} < {global_batch_size})")
        print("Setting the global batch size to the power of 2 closest to the validation dataset size")
        global_batch_size = valid_batch_size 
    
    valid_dataset = (valid_dataset
        .cache()
        .with_options(options)
        .unbatch()
        .shuffle(1000)
        .batch(global_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
    
    if train_len < global_batch_size:
        print(f"WARNING: The training dataset is smaller than the batch size! ({train_len} < {global_batch_size})")
        print("Setting the global batch size to the power of 2 closest to the training dataset size")
        global_batch_size = 2 ** int(np.log2(train_len))

    train_dataset = (train_dataset
        .cache()
        .with_options(options)
        .unbatch()
        .shuffle(1000)
        .batch(global_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dist_dataset = strategy.experimental_distribute_dataset(valid_dataset)

    with strategy.scope():
        mse_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        @tf.function
        def compute_loss(data, train_pred, model_losses):
            # per_example_loss will have the shape of the batch size as the last dimension will be reduced by the loss function.
            per_example_loss = mse_loss(data, train_pred)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) 
                # Now the loss is averaged over the batch size and divided by num_replicas_in_sync
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss

        # Adjust learning rate
        # initial_learning_rate = params['c_learning_rate']
        # scaled_learning_rate = initial_learning_rate * strategy.num_replicas_in_sync
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=True)
        model = Autoencoder(params)  # Train on the CAE, but...

        encoder = model.layers[0]  # ...save weights and checkpoints to the encoder...
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        encoder_mngr = tf.train.CheckpointManager(encoder_ckpt,
                                                    directory=f"{CHECKPOINT_PATH}/nn/checkpoints/cae/{params['c_ver']}/encoder",
                                                    max_to_keep=max_to_keep)

        decoder = model.layers[1]  # ...and the decoder, so that they can be run separately
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
        decoder_mngr = tf.train.CheckpointManager(decoder_ckpt,
                                                    directory=f"{CHECKPOINT_PATH}/nn/checkpoints/cae/{params['c_ver']}/decoder",
                                                    max_to_keep=max_to_keep)

        # Restore encoder and decoder weights and biases
        encoder_ckpt.restore(encoder_mngr.latest_checkpoint).expect_partial()
        decoder_ckpt.restore(decoder_mngr.latest_checkpoint).expect_partial()

        if encoder_mngr.latest_checkpoint and decoder_mngr.latest_checkpoint:
            print(f"Restored encoder from {encoder_mngr.latest_checkpoint}")
            print(f"Restored decoder from {decoder_mngr.latest_checkpoint}")
            # Extract the epoch number from the checkpoint filename
            init_epoch = int(encoder_mngr.latest_checkpoint.split('-')[-1]) + 1
        else:
            init_epoch = 1
            if params['c_record']:
                print('No checkpoint found! Initialising from scratch...')
            else:
                print('Attempting to train/evaluate new model without storing trained values!')
                print('Try setting "c_record True" inside the chosen config')
                print('Or make sure the version name ("c_ver") is spelled correctly\nExiting...')
                exit()

        if params['c_record']:
            # Create summary writers for losses
            train_log_dir = f"{CHECKPOINT_PATH}/nn/logs/cae/{params['c_ver']}/train"
            valid_log_dir = f"{CHECKPOINT_PATH}/nn/logs/cae/{params['c_ver']}/valid"
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            train_pred = model(data, training=True)
            loss = compute_loss(data, train_pred, model_losses=None)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    @tf.function
    def valid_step(data):
        valid_pred = model(data, training=False)
        loss = compute_loss(data, valid_pred, model_losses=None)
        return loss
    
    @tf.function
    def distributed_valid_step(dataset_inputs):
        per_replica_losses = strategy.run(valid_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)
    
    if os.path.exists(os.path.join(train_log_dir, 'train_losses.npy')):
        train_losses = list(np.load(os.path.join(train_log_dir, 'train_losses.npy')))
    else:
        train_losses = []

    if os.path.exists(os.path.join(valid_log_dir, 'valid_losses.npy')):
        valid_losses = list(np.load(os.path.join(valid_log_dir, 'valid_losses.npy')))
    else:
        valid_losses = []

    for epoch in range(init_epoch, params['c_epochs']):
        epoch_train_loss = 0
        epoch_valid_loss = 0

        num_train_batches = 0
        num_valid_batches = 0

        s_epoch = tf.timestamp()
        print(f"---=== Epoch {epoch}/{params['c_epochs']} ===---")

        s_train = tf.timestamp()
        for train_data in train_dist_dataset:
            epoch_train_loss += distributed_train_step(train_data)
            num_train_batches += 1
        epoch_train_loss /= num_train_batches

        train_losses.append(epoch_train_loss)
        d_train = tf.timestamp() - s_train

        s_val = tf.timestamp()
        for valid_data in valid_dist_dataset:
            epoch_valid_loss += distributed_valid_step(valid_data)
            num_valid_batches += 1
        epoch_valid_loss /= num_valid_batches

        valid_losses.append(epoch_valid_loss)
        d_val = tf.timestamp() - s_val

        if params['c_record']:
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_valid_loss, step=epoch)  # Save validation loss every epoch
                np.save(os.path.join(valid_log_dir, 'valid_losses.npy'), np.array(valid_losses))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_train_loss, step=epoch)  # Save training loss every epoch
                np.save(os.path.join(train_log_dir, 'train_losses.npy'), np.array(train_losses))
            if epoch % checkpoint_interval == 0:
                # also save the current epoch number
                encoder_ckpt.save(file_prefix=f"{encoder_mngr.directory}/ckpt-{epoch}")
                decoder_ckpt.save(file_prefix=f"{decoder_mngr.directory}/ckpt-{epoch}")

        if params['c_record']:
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_valid_loss, step=epoch)
                np.save(os.path.join(valid_log_dir, 'valid_losses.npy'), np.array(valid_losses))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_train_loss, step=epoch)
                np.save(os.path.join(train_log_dir, 'train_losses.npy'), np.array(train_losses))
            if epoch % checkpoint_interval == 0:
                # Save checkpoints with custom names
                encoder_save_path = encoder_mngr.save(checkpoint_number=epoch)
                decoder_save_path = decoder_mngr.save(checkpoint_number=epoch)
                print(f"Saved encoder checkpoint: {encoder_save_path}")
                print(f"Saved decoder checkpoint: {decoder_save_path}")

        f_epoch = tf.timestamp()
        d_epoch = f_epoch - s_epoch

        print(f"Time of Epoch, Train, Val: {d_epoch:.2f}, {d_train:.2f}, {d_val:.2f}")
        print(f"Train\Val Losses: {epoch_train_loss:.2e}, {epoch_valid_loss:.2e}")
        # model.summary()

    print('\nModel Trained!')

    return

"""
    Some notes:

    getting rid of .as_numpy_iterator() - reduced time by ~1s
    @tf.function decorator is slower by 1s, but has reduced warnings and Mirrored Strategy has a massive overhead without them
    .cache() didn't change the time, also isnt working on multi gpu as expected + I don't understand how it works
    with large batch sizes the time of epoch platoes to 11.50s, but we cannot increase further due to memory
    50k runs in 13.45s via mirrored strategy on a single gpu:
        10k in 14s
        60k in 13.5s
    to improve:
        -try mixed precision (not a priority)
        -try rate scheduling (not a priority)
"""

if __name__ == '__main__':
    # Load the chosen hyperparameters
    hyperparams = hyperparams_setup(cfg_path="./configs/version_paul.txt")

    # Train the CAE, storing parameters inside ./nn/checkpoints/cae/c_ver, and logs inside ./nn/logs/cae/c_ver
    train(params=hyperparams, checkpoint_interval=20, gpus_to_use = None, max_to_keep=1005)

    # If you want to view loss curves, then run the following on the command line:
    # tensorboard --logdir="./nn/logs/cae/"
    # NOTE: You may need to alter the preceding directory as necessary.
    # This tensorboard can be view in a browser using the default URL: localhost:6006/