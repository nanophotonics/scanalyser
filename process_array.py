import numpy as np
import os
from main_cae_data import load_dataset
from utils import hyperparams_setup
from pathlib import Path

params = hyperparams_setup(cfg_path="./configs/version_6.txt")

train_log_dir = f'./nn/logs/cae/{params["c_ver"]}/train'
valid_log_dir = f'./nn/logs/cae/{params["c_ver"]}/valid'

train_losses = list(np.load(os.path.join(train_log_dir, 'train_losses.npy')))
valid_losses = list(np.load(os.path.join(valid_log_dir, 'valid_losses.npy')))

# Format the values in scientific notation with 3 digits
train_losses = np.array([f'{val:.3e}' for val in train_losses])
valid_losses = np.array([f'{val:.3e}' for val in valid_losses])

# Save the formatted values to a .npy file
np.save('formatted_values_t.npy', np.array(train_losses))
np.save('formatted_values_v.npy', np.array(valid_losses))