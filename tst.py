import os
import mne
import pandas as pd
from config import kumars_path

def load_eeg_csv_to_raw(file_path, sfreq=250):
    data = pd.read_csv(file_path).values.T  # transpose for MNE (channels, samples)
    ch_names = [f"Ch{i+1}" for i in range(data.shape[0])]
    ch_types = ['eeg'] * data.shape

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw

DATASET_DIR = kumars_path 
import os

print(f"Listing files inside: {DATASET_DIR}")
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        print(os.path.join(root, file))

