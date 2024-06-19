import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from analysis.code_tracks.detector import difference_scans
from nn.models.cae_lambda import Autoencoder
from main_cae_data import select_scan
from pathlib import Path
from utils import load_config

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # if you want to suppress unnecessary TF loading messages

# Constants.
scan_size = 1000
ckpt_number = 403 # this ckpt is used for inference

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

print(tf.config.list_physical_devices())
print(tf.__version__)

folder_path = './configs/'
config_to_load = "version_lambda.txt"
model_name = config_to_load
params = load_config(os.path.join(folder_path, config_to_load))

# p_scans contain picocavities, n_scans should not.
p_scans = select_scan(params=params, particles=params['particles'], picos=['True'], exps=[None])
n_scans = select_scan(params=params, particles=params['particles'], picos=['False'], exps=[None])

# Set up the model & restore the checkpoint.
enc_path = f'./nn/checkpoints/cae/{params["c_ver"]}/encoder/ckpt-{ckpt_number}'
dec_path = f'./nn/checkpoints/cae/{params["c_ver"]}/decoder/ckpt-{ckpt_number}'

optimizer = tf.keras.optimizers.Adam(learning_rate=params['c_learning_rate'], clipnorm=False)
model = Autoencoder(params)
encoder = model.layers[0]
decoder = model.layers[1]

encoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=encoder)
decoder_ckpt = tf.train.Checkpoint(optimizer=optimizer, model=decoder)

encoder_ckpt.restore(enc_path).expect_partial()
decoder_ckpt.restore(dec_path).expect_partial()

def obtain_difference_scans_v2(params, dataset, model_name, msg=None, 
                               path_difference="/home/ms3052/rds/hpc-work/Scanalyser",
                               ):
    """ Extracts Tracks from transient peak detections made on scans.

    Args:
        params: Dict, The hyperparameter dictionary.
        dataset: Sliced Tensorflow Array featuring the chosen scans.
        model_name: Used for displaying.
        msg: Custom subtitle to be displayed below the plots (default: None)
        path_difference: path for saving the folder with difference scans and histagrams, has no / at the and.
    """
    for data, label in dataset.as_numpy_iterator():
        # Make the label into a whole string with better formatting for display purposes
        label_str = label.astype('str')[1][-4:]
        print(f'>> Calculating difference spectra for Particle {label_str}')
        print(data.shape)
        print(data)

        # Generate the embeddings and reconstructions
        nbatches = data.shape[0] // scan_size
        embed = tf.Variable(tf.zeros((data.shape[0], params['c_embedding_dim'])))
        recon = tf.Variable(tf.zeros((data.shape[0], data.shape[1], 1)))

        for n in range(nbatches):
            # Generate the embedding and reconstruction tensors
            embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)].assign(encoder(data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)]))
            recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)].assign(decoder(embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)]))

        # Remove redundant dimensions and calculate the difference scan
        data = np.squeeze(data) #lol
        recon = np.squeeze(recon)

        # Plot histogram of values on the scan.
        plt.hist(recon.flatten(), bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.title(f'Histogram for {label_str}')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        output_folder = f"{path_difference}/hist_model_{model_name}"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        histogram_filename = os.path.join(output_folder, f'histogram_{label_str}.png')
        plt.savefig(histogram_filename)
        plt.close()

        # Additional recon processing/normalizing options, if you want to try out.
        # recon = np.clip(recon, 0, 1)
        # recon = normalize(recon, norm = 'max', axis = (0, 1))
        # recon_min = np.min(recon)
        # recon_max = np.max(recon)
        # recon = (recon - recon_min) / (recon_max - recon_min)

        # For debugging purposes.
        print(f"Recon shape final{recon.shape}")
        print(np.max(recon))
        print(np.min(recon))
        print(np.average(recon))

        difference = difference_scans(data, recon)

        # Save the visualization as an image
        fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin

        # Set the custom subtitle if provided, else generate default subtitle
        if msg is not None:
            subtitle = msg
        else:
            # Generate the default formatted subtitle text
            subtitle = (
                f"molecule: {params['molecule']}, model from: {model_name}"
            )
            
        # Add the subtitle below the subplots
        fig.text(0.5, 0, subtitle, fontsize=12, ha='center', va='bottom', multialignment='left')
        # Plot original data
        axes[0].imshow(data, cmap='inferno', aspect='auto')
        axes[0].set_title("Original Data")

        # Plot reconstruction
        axes[1].imshow(recon, cmap='inferno', aspect='auto')
        axes[1].set_title("Reconstruction")

        # Plot difference scan
        axes[2].imshow(difference, cmap='inferno', aspect='auto')
        axes[2].set_title("Difference Scan")

        # Adjust layout
        plt.tight_layout

        # Save the figure as a high-quality image (e.g., PNG)
        path = f'{path_difference}/difference_{params["molecule"]}_model_{model_name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/particle_{label_str}.png', dpi=300)  # Adjust dpi for desired quality

        plt.close(fig)  # Close the figure to free up resources

    return

def obtain_difference_scans(params, dataset, model_name, msg=None):
    """ Extracts Tracks from transient peak detections made on scans.
    

    Args:
        params: Dict, The hyperparameter dictionary.
        dataset: Sliced Tensorflow Array featuring the chosen scans (from the 'True' BPT dataset)
        model_name: Used for displaying.
        msg: Custom subtitle to be displayed below the plots (default: None)
    """
    for data, label in dataset.as_numpy_iterator():
        # Make the label into a whole string with better formatting for display purposes
        label_str = label.astype('str')[1][-4:]
        print(f'>> Calculating difference spectra for Particle {label_str}')

        # Generate the embeddings and reconstructions
        nbatches = data.shape[0] // params['c_batch_size']
        embed = tf.Variable(tf.zeros((data.shape[0], params['c_embedding_dim'])))
        recon = tf.Variable(tf.zeros((data.shape[0], data.shape[1], 1)))

        for n in range(nbatches):
            # Generate the embedding and reconstruction tensors
            embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)].assign(encoder(data[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)]))
            recon[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)].assign(decoder(embed[params['c_batch_size'] * n:params['c_batch_size'] * (n + 1)]))

        # Remove redundant dimensions and calculate the difference scan
        data = np.squeeze(data)
        recon = np.squeeze(recon)
        print(np.max(recon))
        print(np.min(recon))
        print(np.average(recon))

        difference = difference_scans(data, recon)

        # Save the visualization as an image
        fig, axes = plt.subplots(1, 3, figsize=(15, 7))
        plt.subplots_adjust(bottom=0.2)  # Increase the bottom margin
        # Set the custom subtitle if provided, else generate default subtitle
        if msg is not None:
            subtitle = msg
        else:
            # Generate the default formatted subtitle text
            subtitle = (
                f"molecule: {params['molecule']}, model from: {model_name}"
            )
            
        # Add the subtitle below the subplots
        fig.text(0.5, 0, subtitle, fontsize=12, ha='center', va='bottom', multialignment='left')
        # Plot original data
        axes[0].imshow(data, cmap='inferno', aspect='auto')
        axes[0].set_title("Original Data")

        # Plot reconstruction
        axes[1].imshow(recon, cmap='inferno', aspect='auto')
        axes[1].set_title("Reconstruction")

        # Plot difference scan
        axes[2].imshow(difference, cmap='inferno', aspect='auto')
        axes[2].set_title("Difference Scan")

        # Adjust layout
        plt.tight_layout

        # Save the figure as a high-quality image (e.g., PNG)
        path = f'/home/ms3052/rds/hpc-work/Scanalyser/difference_{params["molecule"]}_model_{model_name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/particle_{label_str}.png', dpi=300)  # Adjust dpi for desired quality

        plt.close(fig)  # Close the figure to free up resources

    return

"""
The code below was used for trying out new ideas and debugging.
"""
# Set the auto_shard_policy to DATA
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

# test_dataset, test_len = load_dataset(params=params, data_type='test', path='./data/test/test_specs.csv')
# test_dataset = test_dataset.with_options(options).unbatch()

# print(test_dataset, test_len)

# model(data, training=False)

# p_scans = p_scans.with_options(options).unbatch()

# print(p_scans)

# recon = model(test_dataset, training=False)

# print(recon)

# obtain_difference_scans(params=params, dataset=p_scans, model_name="version_000")
obtain_difference_scans_v2(params=params, dataset=p_scans, model_name="lambda_pico403")
