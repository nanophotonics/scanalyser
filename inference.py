import os
import tensorflow as tf
from importlib import import_module
from utils import load_config
import numpy as np

def enable_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def load_model(ckpt_number=None, model_config="version_lambda.txt") -> tf.keras.Model:
    """ Load the model from the given checkpoint & the config.
    Args:
        ckpt_number: The checkpoint number to load the weights from.
        model_config: The config file to get the model name from.
    The model is loaded from the checkpoint with the provided ckpt_number.
    If the ckpt_number is not provided, the default checkpoint is used.
    If the ckpt_number is "latest", the latest checkpoint is used.
    """

    enable_gpu()

    DEFAULT_CHECKPOINT = 403
    ckpt_number = ckpt_number if ckpt_number is not None else DEFAULT_CHECKPOINT

    # Get the absolute path to the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Use os.path.join with current_dir to get absolute paths
    folder_path = os.path.join(current_dir, 'configs')
    params = load_config(os.path.join(folder_path, model_config))

    # Load the model with a name from the config file. (params["c_ver"])
    Autoencoder = getattr(import_module(f"nn.models.{params['c_ver']}"), 'Autoencoder')

    enc_dir = os.path.join(current_dir, f'nn/checkpoints/cae/{params["c_ver"]}/encoder')
    dec_dir = os.path.join(current_dir, f'nn/checkpoints/cae/{params["c_ver"]}/decoder')

    print(f"I: Encoder dir: {enc_dir}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
    model = Autoencoder(params)
    encoder = model.layers[0]
    decoder = model.layers[1]

    if ckpt_number == "latest":
        try:
            enc_path = tf.train.latest_checkpoint(f'{enc_dir}')
            dec_path = tf.train.latest_checkpoint(f'{dec_dir}')
        except:
            print("Failed to load the latest checkpoint. Loading the default checkpoint {DEFAULT_CHECKPOINT} instead.")
            enc_path = rf'{enc_dir}/ckpt-{DEFAULT_CHECKPOINT}'
            dec_path = rf'{dec_dir}/ckpt-{DEFAULT_CHECKPOINT}'
    else:
        try:
            enc_path = rf'{enc_dir}/ckpt-{ckpt_number}'
            dec_path = rf'{dec_dir}/ckpt-{ckpt_number}'
        except:
            print(f"Failed to load the checkpoint {ckpt_number}. Loading the latest checkpoint instead.")
            enc_path = tf.train.latest_checkpoint(f'{enc_dir}')
            dec_path = tf.train.latest_checkpoint(f'{dec_dir}')

    encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
    decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

    encoder_ckpt.restore(enc_path).expect_partial()
    decoder_ckpt.restore(dec_path).expect_partial()

    return encoder, decoder