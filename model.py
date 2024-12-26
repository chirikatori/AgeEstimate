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
from mne_features.feature_extraction import extract_features
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

# Create 2-second epochs
events = mne.make_fixed_length_events(
    data_raw[0], start=2, duration=2 - 1 / data_raw[0].info['sfreq'], overlap=0., stop=100
)
data_epoch = [mne.Epochs(raw, events=events, event_id=1, tmin=0, tmax=2, proj=False, baseline=None, preload=True) for raw in data_raw]

# Filter valid epochs
valid_data_epoch = [epoch for epoch in data_epoch if epoch is not None and len(epoch) > 0]
invalid_indices = [i for i, epoch in enumerate(data_epoch) if epoch is None or len(epoch) == 0]
print(f"Removed epochs at indices: {invalid_indices}")
print(f"Remaining valid epochs: {len(valid_data_epoch)}")

# Feature extraction parameters
# HC_SELECTED_FUNCS = [
#     'std', 'kurtosis', 'skewness', 'quantile', 'ptp_amp', 'mean', 'pow_freq_bands',
#     'spect_entropy', 'app_entropy', 'samp_entropy', 'svd_entropy', 'hurst_exp',
#     'hjorth_complexity', 'hjorth_mobility', 'line_length', 'wavelet_coef_energy',
#     'higuchi_fd', 'zero_crossings', 'svd_fisher_info'
# ]
# HC_FUNC_PARAMS = {
#     'quantile__q': [0.1, 0.25, 0.75, 0.9],
#     'pow_freq_bands__freq_bands': [0, 2, 4, 8, 13, 18, 24, 30, 49],
#     'pow_freq_bands__ratios': 'all',
#     'pow_freq_bands__ratios_triu': True,
#     'pow_freq_bands__log': True,
#     'pow_freq_bands__normalize': None
# }

# # Extract features from epochs
# final_features = [
#     extract_features(
#         epoch.get_data(), epoch.info['sfreq'], HC_SELECTED_FUNCS,
#         funcs_params=HC_FUNC_PARAMS, n_jobs=-1, ch_names=epoch.ch_names
#     ) for epoch in tqdm(valid_data_epoch, desc="Extracting Features")
# ]

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

# Prepare training and validation datasets
train_size = int(len(X) * 0.75)
X_train, X_valid = X[:train_size], X[train_size:]
y_train = [age for i, age in enumerate(metadata["age"]) if i not in invalid_indices]
y_train = y_train[:len(X)]
y_train, y_valid = y_train[:train_size], y_train[train_size:]

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
plt.savefig("ground_truth_vs_predictions.png", dpi=300, bbox_inches='tight')
plt.show()

