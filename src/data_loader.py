"""
Unified HAR Data Loader — Cross-domain evaluation framework.
Supports: UCI HAR, Shoaib, MotionSense, HHAR
All datasets resampled to 20Hz, windowed at 6s (120 samples), 6 channels (acc+gyro).
Uses total_acc (raw acceleration with gravity) for UCI HAR to ensure cross-domain
signal consistency with Shoaib, MotionSense, and HHAR.
"""
import os
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

# ============================================================================
# PREPROCESSING UTILITIES
# ============================================================================

def resample_signal(data, original_rate, target_rate):
    """Resample multi-channel signal. data: (T, C) -> (T', C)."""
    if original_rate == target_rate:
        return data
    if data.ndim == 1:
        data = data[:, np.newaxis]
    n = data.shape[0]
    new_n = int(np.round(n * target_rate / original_rate))
    if new_n < 1:
        return data[:1]
    return scipy_signal.resample(data, new_n, axis=0)

def create_windows_from_segments(data, label, window_size=120, overlap=0.5):
    """Sliding window over a single-label segment. Returns (windows, labels)."""
    step = int(window_size * (1 - overlap))
    if step < 1:
        step = 1
    n = data.shape[0]
    if n < window_size:
        return np.empty((0, window_size, data.shape[1])), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, n - window_size + 1, step)
    windows = np.stack([data[s:s+window_size] for s in starts])
    labels = np.full(len(starts), label, dtype=np.int64)
    return windows, labels

# ============================================================================
# MOTIONSENSE LOADER
# ============================================================================

def load_motionsense(data_path, target_rate=20, window_size=120, overlap=0.5):
    """Load MotionSense dataset. Returns X:(N,120,6), y:(N,)."""
    base = os.path.join(data_path, 'MotionSense', 'A_DeviceMotion_data')
    TRIAL_FOLDERS = {
        'dws': ['dws_1','dws_2','dws_11'], 'ups': ['ups_3','ups_4','ups_12'],
        'wlk': ['wlk_7','wlk_8','wlk_15'], 'jog': ['jog_9','jog_16'],
        'sit': ['sit_5','sit_13'], 'std': ['std_6','std_14'],
    }
    LABEL_MAP = {'wlk': 0, 'sit': 1, 'std': 1, 'ups': 2, 'dws': 3}
    ACC = ['userAcceleration.x','userAcceleration.y','userAcceleration.z']
    GYRO = ['rotationRate.x','rotationRate.y','rotationRate.z']

    all_w, all_l = [], []
    for act, folders in TRIAL_FOLDERS.items():
        if act not in LABEL_MAP:
            continue
        label = LABEL_MAP[act]
        for folder in folders:
            folder_path = os.path.join(base, folder)
            if not os.path.isdir(folder_path):
                continue
            for f in sorted(os.listdir(folder_path)):
                if not f.startswith('sub_') or not f.endswith('.csv'):
                    continue
                df = pd.read_csv(os.path.join(folder_path, f))
                try:
                    data = np.hstack([df[ACC].values, df[GYRO].values])
                except KeyError:
                    continue
                valid = ~np.any(np.isnan(data), axis=1)
                data = data[valid]
                if len(data) < window_size:
                    continue
                data = resample_signal(data, 50, target_rate)
                w, l = create_windows_from_segments(data, label, window_size, overlap)
                if len(w) > 0:
                    all_w.append(w); all_l.append(l)
    return np.concatenate(all_w).astype(np.float32), np.concatenate(all_l).astype(np.int64)

# ============================================================================
# SHOAIB LOADER
# ============================================================================

def load_shoaib(data_path, target_rate=20, window_size=120, overlap=0.5,
                position='Right_pocket'):
    """Load Shoaib dataset (specified body position). Returns X:(N,120,6), y:(N,)."""
    base = os.path.join(data_path, 'Shoaib')
    OFFSETS = {'Left_pocket': 0, 'Right_pocket': 14, 'Wrist': 28, 'Upper_arm': 42, 'Belt': 56}
    LABEL_MAP = {'walking': 0, 'sitting': 1, 'standing': 1,
                 'upstairs': 2, 'downstairs': 3}

    offset = OFFSETS[position]
    acc_cols = [offset+1, offset+2, offset+3]
    gyro_cols = [offset+7, offset+8, offset+9]
    data_cols = acc_cols + gyro_cols

    all_w, all_l = [], []
    for pid in range(1, 11):
        fpath = os.path.join(base, f'Participant_{pid}.csv')
        if not os.path.isfile(fpath):
            continue
        df = pd.read_csv(fpath, header=None, skiprows=2, low_memory=False)
        activity_col = df.iloc[:, -1]
        try:
            sensor = df.iloc[:, data_cols].values.astype(np.float64)
        except (ValueError, IndexError):
            continue

        cur_act, seg_start = None, 0
        for i in range(len(activity_col) + 1):
            act = str(activity_col.iloc[i]).strip().lower() if i < len(activity_col) else None
            if act != cur_act:
                if cur_act is not None and cur_act in LABEL_MAP:
                    seg = sensor[seg_start:i]
                    valid = ~np.any(np.isnan(seg), axis=1)
                    seg = seg[valid]
                    if len(seg) >= window_size:
                        seg = resample_signal(seg, 50, target_rate)
                        w, l = create_windows_from_segments(seg, LABEL_MAP[cur_act], window_size, overlap)
                        if len(w) > 0:
                            all_w.append(w); all_l.append(l)
                cur_act, seg_start = act, i

    return np.concatenate(all_w).astype(np.float32), np.concatenate(all_l).astype(np.int64)

# ============================================================================
# HHAR LOADER
# ============================================================================

def load_hhar(data_path, target_rate=20, window_size=120, overlap=0.5, devices='all'):
    """Load HHAR dataset. Merges acc+gyro by timestamp. Returns X:(N,120,6), y:(N,)."""
    base = os.path.join(data_path, 'hhar')
    LABEL_MAP = {'walk': 0, 'sit': 1, 'stand': 1, 'stairsup': 2, 'stairsdown': 3}

    file_pairs = [
        (os.path.join(base, 'Phones_accelerometer.csv', 'Phones_accelerometer.csv'),
         os.path.join(base, 'Phones_gyroscope.csv', 'Phones_gyroscope.csv')),
    ]
    if devices == 'all':
        file_pairs.append(
            (os.path.join(base, 'Watch_accelerometer.csv', 'Watch_accelerometer.csv'),
             os.path.join(base, 'Watch_gyroscope.csv', 'Watch_gyroscope.csv')),
        )

    all_w, all_l = [], []
    for acc_file, gyro_file in file_pairs:
        _process_hhar_pair(acc_file, gyro_file, LABEL_MAP, target_rate,
                           window_size, overlap, all_w, all_l)

    return np.concatenate(all_w).astype(np.float32), np.concatenate(all_l).astype(np.int64)

def _process_hhar_pair(acc_file, gyro_file, label_map, target_rate,
                       window_size, overlap, all_w, all_l):
    """Load and merge one acc+gyro file pair for HHAR."""
    acc_df = pd.read_csv(acc_file, usecols=['Creation_Time','x','y','z','User','Device','gt'])
    acc_df = acc_df[acc_df['gt'].isin(label_map.keys())].copy()
    acc_df.rename(columns={'x':'acc_x','y':'acc_y','z':'acc_z'}, inplace=True)

    gyro_df = pd.read_csv(gyro_file, usecols=['Creation_Time','x','y','z','User','Device','gt'])
    gyro_df = gyro_df[gyro_df['gt'].isin(label_map.keys())].copy()
    gyro_df.rename(columns={'x':'gyro_x','y':'gyro_y','z':'gyro_z'}, inplace=True)

    for (user, device), acc_grp in acc_df.groupby(['User','Device']):
        gyro_grp = gyro_df[(gyro_df['User']==user) & (gyro_df['Device']==device)]
        if len(gyro_grp) == 0:
            continue

        acc_grp = acc_grp.sort_values('Creation_Time').reset_index(drop=True)
        gyro_grp = gyro_grp.sort_values('Creation_Time').reset_index(drop=True)

        merged = pd.merge_asof(
            acc_grp[['Creation_Time','acc_x','acc_y','acc_z','gt']],
            gyro_grp[['Creation_Time','gyro_x','gyro_y','gyro_z']],
            on='Creation_Time', direction='nearest', tolerance=50_000_000)
        merged = merged.dropna()
        if len(merged) < window_size:
            continue

        activities = merged['gt'].values
        sensor = merged[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']].values
        timestamps = merged['Creation_Time'].values

        cur_act, seg_start = None, 0
        for i in range(len(activities) + 1):
            act = activities[i] if i < len(activities) else None
            if act != cur_act:
                if cur_act is not None and cur_act in label_map:
                    seg = sensor[seg_start:i].copy()
                    valid = ~np.any(np.isnan(seg), axis=1)
                    seg = seg[valid]
                    if len(seg) >= window_size:
                        ts = timestamps[seg_start:i]
                        ts_valid = ts[valid[:len(ts)]] if len(valid) >= len(ts) else ts
                        if len(ts_valid) > 1:
                            dt = np.median(np.diff(ts_valid))
                            est_rate = np.clip(1e9 / dt, 10, 500) if dt > 0 else 100
                        else:
                            est_rate = 100
                        seg = resample_signal(seg, est_rate, target_rate)
                        w, l = create_windows_from_segments(
                            seg, label_map[cur_act], window_size, overlap)
                        if len(w) > 0:
                            all_w.append(w); all_l.append(l)
                cur_act, seg_start = act, i

# ============================================================================
# UCI HAR LOADER
# ============================================================================

def load_uci(data_path, target_rate=20, window_size=120, overlap=0.5):
    """Load UCI HAR. Reconstructs continuous signal from overlapping windows.
    Uses total_acc (raw acc with gravity) instead of body_acc (gravity-subtracted)
    to match the signal type of Shoaib, MotionSense, and HHAR.
    Returns X:(N,120,6), y:(N,)."""
    base = os.path.join(data_path, 'UCI_HAR_Dataset', 'UCI HAR Dataset')
    LABEL_MAP = {1: 0, 4: 1, 5: 1, 2: 2, 3: 3}
    EXCLUDE = {6}

    all_w, all_l = [], []
    for split in ['train', 'test']:
        channels, labels, subjects = _load_uci_split(base, split)
        for subj in np.unique(subjects):
            mask = subjects == subj
            subj_labels = labels[mask]

            ch_signals = {}
            for ch_name, ch_data in channels.items():
                ch_signals[ch_name] = _reconstruct_continuous(ch_data[mask])

            min_len = min(len(v) for v in ch_signals.values())
            continuous = np.column_stack([
                ch_signals['total_acc_x'][:min_len], ch_signals['total_acc_y'][:min_len],
                ch_signals['total_acc_z'][:min_len], ch_signals['body_gyro_x'][:min_len],
                ch_signals['body_gyro_y'][:min_len], ch_signals['body_gyro_z'][:min_len],
            ])

            cont_labels = _reconstruct_labels(subj_labels, len(continuous))

            mapped = np.full_like(cont_labels, -1)
            for orig, unified in LABEL_MAP.items():
                mapped[cont_labels == orig] = unified
            for ex in EXCLUDE:
                mapped[cont_labels == ex] = -1

            cur_lbl, seg_start = -1, 0
            for i in range(len(mapped) + 1):
                lbl = mapped[i] if i < len(mapped) else -1
                if lbl != cur_lbl:
                    if cur_lbl >= 0:
                        seg = continuous[seg_start:i]
                        if len(seg) >= 20:
                            seg = resample_signal(seg, 50, target_rate)
                            w, l = create_windows_from_segments(
                                seg, cur_lbl, window_size, overlap)
                            if len(w) > 0:
                                all_w.append(w); all_l.append(l)
                    cur_lbl, seg_start = lbl, i

    return np.concatenate(all_w).astype(np.float32), np.concatenate(all_l).astype(np.int64)

def _load_uci_split(base, split):
    """Load one split of UCI HAR raw inertial signals.
    Uses total_acc (raw acc including gravity) for cross-domain compatibility."""
    split_dir = os.path.join(base, split)
    inertial = os.path.join(split_dir, 'Inertial Signals')
    ch_names = ['total_acc_x','total_acc_y','total_acc_z',
                'body_gyro_x','body_gyro_y','body_gyro_z']
    channels = {}
    for ch in ch_names:
        channels[ch] = np.loadtxt(os.path.join(inertial, f'{ch}_{split}.txt'))
    labels = np.loadtxt(os.path.join(split_dir, f'y_{split}.txt'), dtype=int)
    subjects = np.loadtxt(os.path.join(split_dir, f'subject_{split}.txt'), dtype=int)
    return channels, labels, subjects

def _reconstruct_continuous(windows):
    """Overlap-add reconstruction from 128-sample windows with 50% overlap."""
    n_win, win_len = windows.shape
    step = win_len // 2
    total = step * (n_win - 1) + win_len
    signal = np.zeros(total, dtype=np.float64)
    counts = np.zeros(total, dtype=np.float64)
    for i, w in enumerate(windows):
        s = i * step
        signal[s:s+win_len] += w
        counts[s:s+win_len] += 1.0
    return signal / np.maximum(counts, 1.0)

def _reconstruct_labels(window_labels, total_samples):
    """Reconstruct per-sample labels from window labels (128-sample, 64-step)."""
    labels = np.zeros(total_samples, dtype=int)
    for i, lbl in enumerate(window_labels):
        s = i * 64
        e = min(s + 128, total_samples)
        labels[s:e] = lbl
    return labels

# ============================================================================
# UNIFIED GET_DATASET INTERFACE
# ============================================================================

LOADERS = {
    'uci': load_uci,
    'shoaib': load_shoaib,
    'motionsense': load_motionsense,
    'hhar': load_hhar,
}

def get_dataset(name, data_path, target_rate=20, window_size=120, overlap=0.5, **kwargs):
    """Load any HAR dataset by name."""
    name = name.lower()
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADERS.keys())}")
    loader_fn = LOADERS[name]
    import inspect
    sig = inspect.signature(loader_fn)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    X, y = loader_fn(data_path, target_rate, window_size, overlap, **valid_kwargs)
    print(f"  [{name}] Loaded {len(X)} windows | shape={X.shape} | "
          f"labels={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y