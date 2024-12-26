# -*- coding: utf-8 -*-

# Install required libraries
# !pip install -q mne mne_features coffeine

# Import necessary modules
import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import coffeine

# Define constants and paths
RAW_DATA_FOLDER = "/content/drive/MyDrive/all_group/"
METADATA_FILE = os.path.join(RAW_DATA_FOLDER, "filtered_subjects_with_age.tsv")
N_ROWS = 100  # Number of rows to load from metadata

# Load metadata
metadata = pd.read_csv(METADATA_FILE, sep="\t", nrows=N_ROWS)
metadata.columns = [col.strip() for col in metadata.columns]  # Clean column names
file_paths = [os.path.join(RAW_DATA_FOLDER, f"{participant_id}_sflip_parc-raw.fif") for participant_id in metadata["participant_id"]]

# Load raw data and pick 'misc' channel
data_raw = [mne.io.read_raw_fif(path, preload=True, verbose=False).pick('misc') for path in tqdm(file_paths, desc="Loading Data")]
for raw in data_raw:
    raw.set_channel_types({ch_name: 'eeg' for ch_name in raw.info['ch_names'] if 'misc' in ch_name}) 
channel = data_raw[0].info['ch_names']
for r in data_raw:
    r.pick(channel)

mne.set_log_level("ERROR")
epoch_duration = 50
start_time = 0
stop_time = 574
events = mne.make_fixed_length_events(
    data_raw[0], start=start_time, duration=epoch_duration - 1 / data_raw[0].info['sfreq'], overlap=0., stop=stop_time
)
data_epoch = [mne.Epochs(raw, events=events, event_id=1, tmin=0, tmax=epoch_duration, proj=True, baseline=None, preload=True) for raw in data_raw]

# Filter valid epochs
valid_data_epoch = [epoch for epoch in data_epoch if epoch is not None and len(epoch) > 0]
invalid_indices = [i for i, epoch in enumerate(data_epoch) if epoch is None or len(epoch) == 0]
print(f"Removed epochs at indices: {invalid_indices}")
print(f"Remaining valid epochs: {len(valid_data_epoch)}")


# Define frequency bands for coffeine
FREQUENCY_BANDS = {
    "low": (0.1, 1),
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 15),
    "beta_low": (15, 26),
    "beta_mid": (26, 35),
    "beta_high": (35, 49)
}

def extract_fb_covs(epochs, n_jobs=4):
    features, meta_info = coffeine.compute_features(
        epochs, features=('covs',), n_fft=1024, n_overlap=512,
        fs=epochs.info['sfreq'], fmax=49, frequency_bands=FREQUENCY_BANDS, n_jobs=n_jobs
    )
    features['meta_info'] = meta_info
    return features

# Compute frequency band covariance features
data_train = [
    extract_fb_covs(epoch.set_channel_types({ch_name: 'eeg' for ch_name in channel}), n_jobs=4) for epoch in tqdm(valid_data_epoch, desc="Computing Frequency Band Covariances")
]
covs = np.array([sub['covs'] for sub in data_train])
X = pd.DataFrame({band: list(covs[:, ii]) for ii, band in enumerate(FREQUENCY_BANDS)})
y = [age for i, age in enumerate(metadata["age"]) if i not in invalid_indices]

bins = np.arange(0, 100, 10) 
y_binned = np.digitize(y, bins=bins, right=False)
# Prepare training and validation datasets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y_binned
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_valid)}")

# Train Ridge regression model
filter_bank_transformer = coffeine.make_filter_bank_transformer(
    names=list(FREQUENCY_BANDS), method='riemann',
    projection_params=dict(scale='auto', n_compo=len(channel) - 1)
)
model = make_pipeline(
    filter_bank_transformer, StandardScaler(),
    RidgeCV(alphas=np.logspace(-5, 10, 100))
)
model.fit(X_train, y_train)

# Evaluate the model
train_predictions = model.predict(X_train)
valid_predictions = model.predict(X_valid)
print(f"Train R2: {r2_score(y_train, train_predictions)}")
print(f"Train MAE: {mean_absolute_error(y_train, train_predictions)}")
print(f"Validation R2: {r2_score(y_valid, valid_predictions)}")
print(f"Validation MAE: {mean_absolute_error(y_valid, valid_predictions)}")

# Plot results
plt.figure(figsize=(12, 6))

# Train set
plt.subplot(1, 2, 1)
plt.scatter(y_train, train_predictions, alpha=0.7, label="Predictions")
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', label="Ideal")
plt.title("Train Set: Ground Truth vs Predictions")
plt.xlabel("Ground Truth (y_train)")
plt.ylabel("Predictions")
plt.legend()
plt.grid(True)

# Validation set
plt.subplot(1, 2, 2)
plt.scatter(y_valid, valid_predictions, alpha=0.7, label="Predictions")
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], 'r--', label="Ideal")
plt.title("Validation Set: Ground Truth vs Predictions")
plt.xlabel("Ground Truth (y_valid)")
plt.ylabel("Predictions")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
