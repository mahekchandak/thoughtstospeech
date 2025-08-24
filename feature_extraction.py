import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch

from config import kumars_path

def load_subject(subject_file, n_trials=5, trial_len=5.0):
    raw = mne.io.read_raw_fif(subject_file, preload=True)
    fs = raw.info['sfreq']
    n_samples = int(trial_len * fs)

    trials = []
    for i in range(n_trials):
        start = i * n_samples
        stop = start + n_samples
        if stop > raw.n_times:  # stop if recording too short
            break
        data, times = raw[:, start:stop]
        trials.append((data, times))
    return raw, trials

def eda_report(subject_id, n_trials=5, trial_len=5.0):
    # locate subject file
    cleaned_dir = os.path.join(kumars_path, "cleaned_fif")
    fif_files = sorted([f for f in os.listdir(cleaned_dir) if f.endswith(".fif")])
    if subject_id >= len(fif_files):
        raise IndexError("Subject ID out of range")

    subject_file = os.path.join(cleaned_dir, fif_files[subject_id])
    raw, trials = load_subject(subject_file, n_trials=n_trials, trial_len=trial_len)

    fs = raw.info['sfreq']
    ch_names = raw.ch_names
    bads = raw.info.get("bads", [])

    # --- Plot raw traces (5 channels, 2 trials) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    n_show_chs = min(5, len(ch_names))

    for i in range(min(2, len(trials))):  # only first 2 trials
        data, times = trials[i]
        for j in range(n_show_chs):
            axes[i].plot(times, data[j]*1e6 + j*50, label=ch_names[j])  # µV + offset
        axes[i].set_title(f"Trial {i+1} (5 channels)")
        axes[i].set_ylabel("µV (offset)")
        if i == 1:
            axes[i].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f"eda_subject{subject_id}_raw.png")
    plt.close(fig)

    # --- Plot PSD (Welch) of 1 channel from first trial ---
    data, _ = trials[0]
    ch_idx = 0  # first channel
    f, pxx = welch(data[ch_idx], fs=fs, nperseg=fs*2)
    plt.figure(figsize=(8,4))
    plt.semilogy(f, pxx)
    plt.title(f"PSD (Welch) - {ch_names[ch_idx]} - Trial 1")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (V^2/Hz)")
    plt.tight_layout()
    plt.savefig(f"eda_subject{subject_id}_psd.png")
    plt.close()

    # --- Save metadata JSON ---
    meta = {
        "fs": fs,
        "n_channels": len(ch_names),
        "n_trials": len(trials),
        "trial_length_s": trial_len,
        "channels_missing": bads
    }
    with open(f"eda_subject{subject_id}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"EDA report saved for subject {subject_id}")
    return meta

if __name__ == "__main__":
    meta = eda_report(subject_id=0, n_trials=5, trial_len=5.0)
    print(json.dumps(meta, indent=2))
