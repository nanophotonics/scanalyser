# Add the path to the Scanalyser package to the system path.
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file & add the path to the Scanalyser package to the system path.
load_dotenv()
PATH_TO_SCANALYSER = os.getenv("PATH_TO_SCANALYSER")
sys.path.append(PATH_TO_SCANALYSER)

from inference import load_model
from main_cae_data import select_scan
from utils import load_config
from analysis.code_tracks.detector import difference_scans

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def obtain_difference_scans(params, dataset, model_name, msg=None, 
                               path_difference="",
                               ):
    
    for data, label in dataset.as_numpy_iterator():
        # Make the label into a whole string with better formatting for display purposes
        label_str = label.astype('str')[1][-4:]
        print(f'>> Calculating difference spectra for Particle {label_str}')
        print(data.shape)

        recon = decoder(encoder(data))

        # Remove redundant dimensions and calculate the difference scan
        recon = np.squeeze(recon)
        data = np.squeeze(data)

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
        # path = f'{path_difference}/difference_{params["molecule"]}_model_{model_name}/'
        path = fr'{path_difference}/test_{model_name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/particle_{label_str}.png', dpi=300)  # Adjust dpi for desired quality

        plt.close(fig)

    return

if __name__ == "__main__":

    ckpt_number = 403 # this ckpt is used for inference

    # p_scans contain picocavities, n_scans should not.
    data_cfg_path = os.path.join(PATH_TO_SCANALYSER, "configs", "version_lambda.txt")
    params = load_config(data_cfg_path)
    p_scans = select_scan(params=params, particles=params['particles'], picos=['True'], exps=[None])
    # n_scans = select_scan(params=params, particles=params['particles'], picos=['False'], exps=[None])

    # Load the model. NOTE: You don't have to provide the ckpt_number if you want to use the default checkpoint.
    encoder, decoder = load_model()
    obtain_difference_scans(params=params, dataset=p_scans, model_name="sat29_inf_v4", path_difference="/home/ms3052/rds/hpc-work/Scanalyser")
