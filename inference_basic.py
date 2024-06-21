import os
import matplotlib.pyplot as plt
import numpy as np
from main_cae_data import select_scan
from pathlib import Path
from utils import load_config
from inference import forward_pass
from analysis.code_tracks.detector import difference_scans


# Constants.
scan_size = 1000
ckpt_number = 403 # this ckpt is used for inference

# p_scans contain picocavities, n_scans should not.
params = load_config("version_lambda.txt")
p_scans = select_scan(params=params, particles=params['particles'], picos=['True'], exps=[None])
n_scans = select_scan(params=params, particles=params['particles'], picos=['False'], exps=[None])


def obtain_difference_scans(params, dataset, model_name, msg=None, 
                               path_difference="/home/ms3052/rds/hpc-work/Scanalyser",
                               ):
    
    for data, label in dataset.as_numpy_iterator():
        # Make the label into a whole string with better formatting for display purposes
        label_str = label.astype('str')[1][-4:]
        print(f'>> Calculating difference spectra for Particle {label_str}')
        print(data.shape)

        recon = forward_pass(data, ckpt_number=ckpt_number)

        # Remove redundant dimensions and calculate the difference scan
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
        path = f'{path_difference}/difference_{params["molecule"]}_model_{model_name}/'
        Path(path).mkdir(parents=True, exist_ok=True)
        fig.savefig(f'{path}/particle_{label_str}.png', dpi=300)  # Adjust dpi for desired quality

        plt.close(fig)

        return

if __name__ == "__main__":
    obtain_difference_scans(params=params, dataset=p_scans, model_name="lambda_fr21")
