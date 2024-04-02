import os
import tensorflow as tf
from pathlib import Path
from nn.models.cae_lambda import Autoencoder
from main_cae_data import load_dataset
from utils import hyperparams_setup
import time
import numpy as np
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

def set_seeds(seed=123):
    # Set the global random seed for TensorFlow
    tf.random.set_seed(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    return seed # Used as the operation-level seed for TensorFlow

seed = set_seeds()

def train(params, checkpoint_interval=1, gpus_to_use=None, max_to_keep=2):
    """ Train and validate the CAE model using the BPT dataset

    Args:
        params: Dict, The hyperparameter dictionary;
        checkpoint_interval: int, The step between saved checkpoints;
        gpus_to_use: list, The list of gpus (devices) to use for the training. Example: ['GPU:0', 'GPU:3'].
        max_to_keep: int, the max number of checkpoints to keep;
    """
    
    strategy = tf.distribute.MirroredStrategy(gpus_to_use)

    print(f"{strategy.num_replicas_in_sync} GPUs in the strategy")

    global_batch_size = params['c_batch_size'] * strategy.num_replicas_in_sync

    # Set the auto_shard_policy to DATA
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    valid_dataset, valid_len = load_dataset(params=params, data_type='valid', path='./data/valid_clean/valid_specs.csv')
    valid_dataset = valid_dataset.with_options(options).unbatch().shuffle(valid_len * 100, seed=seed).batch(global_batch_size)

    train_dataset, train_len = load_dataset(params=params, data_type='train', path='./data/train_clean/train_specs.csv')
    train_dataset = train_dataset.with_options(options).unbatch().shuffle(train_len * 100, seed=seed).batch(global_batch_size, drop_remainder=True)

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
        model = Autoencoder(params)  # Train on the CAE, but...

        encoder = model.layers[0]  # ...save weights and checkpoints to the encoder...
        encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
        encoder_mngr = tf.train.CheckpointManager(encoder_ckpt,
                                                    directory=f'./nn/checkpoints/cae/{params["c_ver"]}/encoder',
                                                    max_to_keep=max_to_keep)

        decoder = model.layers[1]  # ...and the decoder, so that they can be run separately
        decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
        decoder_mngr = tf.train.CheckpointManager(decoder_ckpt,
                                                    directory=f'./nn/checkpoints/cae/{params["c_ver"]}/decoder',
                                                    max_to_keep=max_to_keep)

        # Restore encoder and decoder weights and biases
        encoder_ckpt.restore(encoder_mngr.latest_checkpoint).expect_partial()
        decoder_ckpt.restore(decoder_mngr.latest_checkpoint).expect_partial()

        if encoder_mngr.latest_checkpoint and decoder_mngr.latest_checkpoint:
            print(f"Restored encoder from {encoder_mngr.latest_checkpoint}")
            print(f"Restored decoder from {decoder_mngr.latest_checkpoint}")
            init_epoch = int(encoder_ckpt.save_counter.read_value().numpy())
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

        s_epoch = tf.timestamp()
        current_epoch = epoch + 1
        print(f"---=== Epoch {current_epoch}/{params['c_epochs']} ===---")

        s_train = tf.timestamp()
        for train_data in train_dist_dataset:
            epoch_train_loss += distributed_train_step(train_data)
            # num_batches += 1

        # It is currently uknown whether or not this loss is batch-size independent.
        epoch_train_loss /= train_len
        train_losses.append(epoch_train_loss)
        d_train = tf.timestamp() - s_train

        # num_batches = 0
        s_val = tf.timestamp()
        for valid_data in valid_dist_dataset:
            epoch_valid_loss += distributed_valid_step(valid_data)
            # num_batches += 1
        epoch_valid_loss /= valid_len
        valid_losses.append(epoch_valid_loss)
        d_val = tf.timestamp() - s_val

        if params['c_record']:
            with valid_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_valid_loss, step=current_epoch)  # Save validation loss every epoch
                np.save(os.path.join(valid_log_dir, 'valid_losses.npy'), np.array(valid_losses))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_train_loss, step=current_epoch)  # Save training loss every epoch
                np.save(os.path.join(train_log_dir, 'train_losses.npy'), np.array(train_losses))
            if current_epoch % checkpoint_interval == 0:
                encoder_mngr.save()
                decoder_mngr.save()
                # print(f"Saved checkpoint at epoch {current_epoch}")

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
        -allow for multi-gpu (you may need to install NCCL)
        -try mixed precision (not a priority)
        -try rate scheduling (not a priority)
        -install linux on the local PC or make the old soft work on the windows
        -Write the Code to process Paul's dataset
"""

if __name__ == '__main__':
    # Load the chosen hyperparameters
    hyperparams = hyperparams_setup(cfg_path="./configs/version_lambda.txt")

    # Train the CAE, storing parameters inside ./nn/checkpoints/cae/c_ver, and logs inside ./nn/logs/cae/c_ver
    train(params=hyperparams, checkpoint_interval=10, gpus_to_use = ["GPU:0"], max_to_keep=500)

    # Fine-tune the CAE, storing parameters inside ./nn/checkpoints/cae/c_ver/c_ver_ft, and logs
    # inside ./nn/logs/cae/c_ver/c_ver_ft
    # if not (hyperparams['c_record'] and not hyperparams['c_record_ft']):
    #     fine_tune(params=hyperparams)

    # If you want to view loss curves, then run the following on the command line:
    # tensorboard --logdir="./nn/logs/cae/"
    # NOTE: You may need to alter the preceding directory as necessary.
    # This tensorboard can be view in a browser using the default URL: localhost:6006/

     # If you want to continue training from checkpoint 2500 for more epochs
    # continue_training = True

    # if continue_training:
    #     # Load checkpoint 2500
    #     encoder_ckpt.restore('./nn/checkpoints/cae/cae_v1/encoder/ckpt-2500').expect_partial()
    #     decoder_ckpt.restore('./nn/checkpoints/cae/cae_v1/decoder/ckpt-2500').expect_partial()

    #     # Modify hyperparameters if needed
    #     hyperparams['c_epochs'] = 2600  # Set to the desired total number of epochs

    #     # Train the CAE from checkpoint 2500 for additional epochs
    #     train_cae(params=hyperparams)

# The fine_tune is currently quite out-dated.
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
                                                max_to_keep=5) #was 2, not 5

    decoder = model.layers[1]  # ...and the decoder, so that they can be run separately
    decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)
    decoder_mngr = tf.train.CheckpointManager(decoder_ckpt,
                                                directory=f'./nn/checkpoints/cae/{params["c_ver"]}/decoder',
                                                max_to_keep=5) #was 2, not 5

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

    # Load in the validation dataset and its length
    valid_dataset, valid_len = load_dataset(params=params, data_type='valid')

    for epoch in range(init_epoch, params['c_epochs_ft']):
        epoch_train_loss = 0  # initialise epoch training loss
        epoch_valid_loss = 0  # initialise epoch training accuracy

        # Reload the training dataset (shuffles it)
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