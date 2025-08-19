import os
import numpy as np
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

from config import kumars_path

folders_to_use = ['Char', 'Digit']
non_eeg_chs = [
    'COUNTER','INTERPOLATED','RAW_CQ','GYROX','GYROY','MARKER','SYNC',
    'TIME_STAMP_s','TIME_STAMP_ms','CQ_AF3','CQ_F7','CQ_F3','CQ_FC5','CQ_T7',
    'CQ_P7','CQ_O1','CQ_O2','CQ_P8','CQ_T8','CQ_FC6','CQ_F4','CQ_F8','CQ_AF4',
    'CQ_CMS','CQ_DRL'
]

def find_edf_files(base_path, include_folders):
    out = []
    for f in include_folders:
        p = os.path.join(base_path, f)
        if not os.path.isdir(p):
            continue
        for x in os.listdir(p):
            if x.lower().endswith('.edf'):
                out.append(os.path.join(p, x))
    return sorted(out)

def load_raw(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    drop = [ch for ch in non_eeg_chs if ch in raw.ch_names]
    if drop:
        raw.drop_channels(drop)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    return raw

def basic_filters(raw):
    raw.filter(1., 40., fir_design='firwin')
    raw.notch_filter(freqs=50, fir_design='firwin')
    return raw

def reref(raw):
    raw.set_eeg_reference('average', projection=True)
    raw.apply_proj()
    return raw

def detect_bad_channels(raw, ptp_uv_thresh=500., flat_uv_thresh=1.0):
    data = raw.get_data(picks='eeg') * 1e6
    chs = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True)).ch_names
    bads = []
    for i, ch in enumerate(chs):
        ptp = np.ptp(data[i])
        std = data[i].std()
        if ptp > ptp_uv_thresh or std < flat_uv_thresh:
            bads.append(ch)
    return bads

def interpolate_bads(raw, bads):
    if bads:
        raw.info['bads'] = list(set(raw.info.get('bads', []) + bads))
        raw.interpolate_bads(reset_bads=True)
    return raw

def run_ica_with_retry(raw, n_components=None, seed=97):
    try:
        ica = ICA(n_components=n_components, random_state=seed, max_iter='auto')
        ica.fit(raw)
    except Exception:
        ica = ICA(n_components=n_components, random_state=seed, max_iter=1000)
        ica.fit(raw)
    return ica

def plot_before_after(raw_before, raw_after, n_show_channels=10, dur=10.0):
    raw_before.plot(n_channels=n_show_channels, duration=dur, title='Raw (before cleaning)')
    raw_before.compute_psd(fmin=1, fmax=40, average='mean').plot().suptitle('PSD (before)')

    raw_after.plot(n_channels=n_show_channels, duration=dur, title='Raw (after cleaning)')
    raw_after.compute_psd(fmin=1, fmax=40, average='mean').plot().suptitle('PSD (after)')

    chs = raw_after.ch_names[:3]
    data_b, t = raw_before[chs, :int(raw_before.info['sfreq']*dur)]
    data_a, _ = raw_after[chs, :int(raw_after.info['sfreq']*dur)]

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, ch in enumerate(chs):
        ax[i].plot(t, data_b[i]*1e6, alpha=0.6, label=f'{ch} before')
        ax[i].plot(t, data_a[i]*1e6, alpha=0.9, label=f'{ch} after')
        ax[i].set_ylabel('µV')
        ax[i].legend(loc='upper right')
    ax[-1].set_xlabel('time (s)')
    fig.suptitle('Before vs After (overlay, 3 channels)')
    plt.tight_layout()
    plt.show()

def clean_one_file(file_path, show_plots=True):
    raw = load_raw(file_path)

    raw_before = raw.copy()
    raw_before = reref(raw_before)
    raw_before = basic_filters(raw_before)
    bads = detect_bad_channels(raw_before)
    raw_before = interpolate_bads(raw_before, bads)

    raw_after = raw_before.copy()
    ica = run_ica_with_retry(raw_after, n_components=None)

    # Visualize ICA components for manual artifact identification
    ica.plot_components(ch_type='eeg')

    # Prompt user to set which components to exclude after inspection
    print("Please enter ICA component indices to exclude (comma separated), e.g. 0,1,2 or leave blank:")
    user_input = input()
    if user_input.strip():
        ica.exclude = [int(x) for x in user_input.split(',')]
    else:
        ica.exclude = []

    ica.apply(raw_after)

    if show_plots:
        plot_before_after(raw_before, raw_after)

    return raw_after

def save_fif(raw, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    raw.save(out_path, overwrite=True)

if __name__ == '__main__':
    edfs = find_edf_files(kumars_path, folders_to_use)
    if not edfs:
        raise FileNotFoundError('No .edf files found in the selected folders')

    print(f'Found {len(edfs)} EDF files')
    first = edfs[0]
    print(f'Previewing and plotting: {os.path.basename(first)}')
    cleaned = clean_one_file(first, show_plots=True)

    out_dir = os.path.join(kumars_path, 'cleaned_fif')
    out_file = os.path.splitext(os.path.basename(first))[0] + '_cleaned.fif'
    save_fif(cleaned, os.path.join(out_dir, out_file))0
    print('Saved first cleaned file')

    do_batch = False  # Set True to process all silently
    if do_batch:
        for p in edfs[1:]:
            print(f'Cleaning: {os.path.basename(p)}')
            c = clean_one_file(p, show_plots=False)
            of = os.path.splitext(os.path.basename(p))[0] + '_cleaned.fif'
            save_fif(c, os.path.join(out_dir, of))
        print('Batch done')
0