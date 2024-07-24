import os
import sys
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv

from nn.models.cae_lambda import Autoencoder
from main_cae_data import load_dataset
from utils import hyperparams_setup, enable_gpu, setup_logger
import logging
from tqdm import tqdm

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv("PAUL_DATA_PATH")
DATA_NAME = os.getenv("PAUL_DATA_NAME")
CHECKPOINT_PATH = DATA_PATH
SEED = int(os.getenv("SEED", "1234"))

# Set random seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)

def train(params, checkpoint_interval=1, gpus_to_use=None, max_to_keep=2, verbose=False):
    """ Train and validate the CAE model using the BPT dataset

    Args:
        params: Dict, The hyperparameter dictionary;
        checkpoint_interval: int, The step between saved checkpoints;
        gpus_to_use: list, The list of gpus (devices) to use for the training. Example: ["GPU:0", "GPU:1"].
        max_to_keep: int, the max number of checkpoints to keep;
    """
    logger = setup_logger(verbose)

    # Enable mixed precision training
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    strategy = tf.distribute.MirroredStrategy(gpus_to_use)

    logger.info(f"{strategy.num_replicas_in_sync} GPUs in the strategy")

    global_batch_size = params['c_batch_size'] * strategy.num_replicas_in_sync

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    valid_dataset, _ = load_dataset(params=params, data_type='valid', path=fr"{DATA_PATH}/data/valid_clean/valid_specs.csv")
    valid_len = sum(1 for _ in valid_dataset.unbatch()) # Count the number of batches (2D scans) in the dataset

    train_dataset, _ = load_dataset(params=params, data_type='train', path=fr"{DATA_PATH}/data/train_clean/train_specs.csv")
    train_len = sum(1 for _ in train_dataset.unbatch())

    valid_batch_size = 2 ** int(np.log2(valid_len))

    if valid_batch_size < global_batch_size:
        logger.warning(f"The validation dataset is smaller than the batch size! ({valid_len} < {global_batch_size})")
        logger.warning("Setting the global batch size to the power of 2 closest to the validation dataset size")
        global_batch_size = valid_batch_size 
    
    valid_dataset = (valid_dataset
        .cache()
        .with_options(options)
        .unbatch()
        .shuffle(1000)
        .batch(global_batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE))
    
    if train_len < global_batch_size:
        logger.warning(f"The training dataset is smaller than the batch size! ({train_len} < {global_batch_size})")
        logger.warning("Setting the global batch size to the power of 2 closest to the training dataset size")
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
            per_example_loss = mse_loss(data, train_pred)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) 
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss

        optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=True)
        model = Autoencoder(params)

        encoder = model.layers[0]
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        encoder_mngr = tf.train.CheckpointManager(encoder_ckpt,
                                                    directory=f"{CHECKPOINT_PATH}/nn/checkpoints/cae/{params['c_ver']}/encoder",
                                                    max_to_keep=max_to_keep)

        decoder = model.layers[1]
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
        decoder_mngr = tf.train.CheckpointManager(decoder_ckpt,
                                                    directory=f"{CHECKPOINT_PATH}/nn/checkpoints/cae/{params['c_ver']}/decoder",
                                                    max_to_keep=max_to_keep)

        encoder_ckpt.restore(encoder_mngr.latest_checkpoint).expect_partial()
        decoder_ckpt.restore(decoder_mngr.latest_checkpoint).expect_partial()

        if encoder_mngr.latest_checkpoint and decoder_mngr.latest_checkpoint:
            logger.info(f"Restored encoder from {encoder_mngr.latest_checkpoint}")
            logger.info(f"Restored decoder from {decoder_mngr.latest_checkpoint}")
            init_epoch = int(encoder_mngr.latest_checkpoint.split('-')[-1]) + 1
        else:
            init_epoch = 1
            if params['c_record']:
                logger.info('No checkpoint found! Initialising from scratch...')
            else:
                logger.error('Attempting to train/evaluate new model without storing trained values!')
                logger.error('Try setting "c_record True" inside the chosen config')
                logger.error('Or make sure the version name ("c_ver") is spelled correctly\nExiting...')
                sys.exit()

        if params['c_record']:
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

    for epoch in tqdm(range(init_epoch, params['c_epochs']), desc="Epochs", disable=not verbose):
        epoch_train_loss = 0
        epoch_valid_loss = 0

        num_train_batches = 0
        num_valid_batches = 0

        s_epoch = tf.timestamp()

        logger.info(f"---=== Epoch {epoch}/{params['c_epochs']} ===---")

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
                tf.summary.scalar('loss', epoch_valid_loss, step=epoch)
                np.save(os.path.join(valid_log_dir, 'valid_losses.npy'), np.array(valid_losses))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_train_loss, step=epoch)
                np.save(os.path.join(train_log_dir, 'train_losses.npy'), np.array(train_losses))
            if epoch % checkpoint_interval == 0:
                encoder_save_path = encoder_mngr.save(checkpoint_number=epoch)
                decoder_save_path = decoder_mngr.save(checkpoint_number=epoch)
                logger.info(f"Saved encoder checkpoint: {encoder_save_path}")
                logger.info(f"Saved decoder checkpoint: {decoder_save_path}")

        f_epoch = tf.timestamp()
        d_epoch = f_epoch - s_epoch

        logger.info(f"Time of Epoch, Train, Val: {d_epoch:.2f}, {d_train:.2f}, {d_val:.2f}")
        logger.info(f"Train\Val Losses: {epoch_train_loss:.2e}, {epoch_valid_loss:.2e}")
    return

if __name__ == '__main__':
    enable_gpu()
    hyperparams = hyperparams_setup(cfg_path="./configs/version_paul.txt")
    train(params=hyperparams, checkpoint_interval=20, gpus_to_use = None, max_to_keep=5005, verbose=True)

    # If you want to view loss curves, then run the following on the command line:
    # tensorboard --logdir="./nn/logs/cae/"