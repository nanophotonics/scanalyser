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

def load_model(ckpt_number=None, config_to_load="version_lambda.txt"):
    enable_gpu()

    folder_path = './configs/'
    params = load_config(os.path.join(folder_path, config_to_load))

    # Load the model with a name from the config file. (params["c_ver"])
    Autoencoder = getattr(import_module(f"nn.models.{params['c_ver']}"), 'Autoencoder')

    enc_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder'
    dec_dir = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder'

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
    model = Autoencoder(params)
    encoder = model.layers[0]
    decoder = model.layers[1]

    # If the ckpt_number is not provided, use the latest checkpoint.
    if ckpt_number is None:
        enc_path = tf.train.latest_checkpoint(f'{enc_dir}')
        dec_path = tf.train.latest_checkpoint(f'{dec_dir}')
    else:
        enc_path = f'{enc_dir}/ckpt-{ckpt_number}'
        dec_path = f'{dec_dir}/ckpt-{ckpt_number}'

    encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
    decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

    encoder_ckpt.restore(enc_path).expect_partial()
    decoder_ckpt.restore(dec_path).expect_partial()

    return encoder, decoder

def forward_pass(input_tensor, config_to_load="version_lambda.txt", ckpt_number=None):
    encoder, decoder = load_model(ckpt_number, config_to_load)
    input_tensor = tf.expand_dims(input_tensor, axis=-1)
    output_tensor = decoder(encoder(input_tensor))
    output_tensor = np.squeeze(output_tensor)
    return output_tensor

# if __name__ == "__main__":
#     ckpt_number = 403 # this ckpt is used for inference
#     config_to_load = "version_lambda.txt"
#     encoder, decoder = load_model(ckpt_number, config_to_load)

#     # Assuming input_tensor is defined and properly formatted
#     output_tensor = forward_pass(input_tensor, encoder, decoder)