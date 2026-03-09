"""
Shared CWRU data loading and feature extraction for apple-to-apple comparison
between ML (ID-crwu-bearings-benchmarks-shap-explainability) and DL (ID-crwu-bearings-wdcnn-shap-signal-explainability) notebooks.
"""
import os
import numpy as np
import scipy.io
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

CWRU_MAT_FILES = {
    "Normal_1": "Time_Normal_1_098.mat",
    "Ball_007_1": "B007_1_123.mat",
    "Ball_014_1": "B014_1_190.mat",
    "Ball_021_1": "B021_1_227.mat",
    "IR_007_1": "IR007_1_110.mat",
    "IR_014_1": "IR014_1_175.mat",
    "IR_021_1": "IR021_1_214.mat",
    "OR_007_6_1": "OR007_6_1_136.mat",
    "OR_014_6_1": "OR014_6_1_202.mat",
    "OR_021_6_1": "OR021_6_1_239.mat",
}
CLASS_NAMES = list(CWRU_MAT_FILES.keys())

FEATURE_COLS = ["max", "min", "mean", "sd", "rms", "skewness", "kurtosis", "crest", "form"]


def load_cwru(raw_dir, segment_length=2048, samples_per_class=230):
    """
    Load CWRU .mat files, deterministic non-overlapping segments.
    Returns X (N, 2048), y (N,), class_names.
    No normalization; apply after split to avoid data leakage.
    """
    X_list, y_list = [], []
    for fault_name in CLASS_NAMES:
        fname = CWRU_MAT_FILES[fault_name]
        path = os.path.join(raw_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing: {path}")
        data = scipy.io.loadmat(path)
        key = "DE_time" if "DE_time" in data else [k for k in data if not k.startswith("__")][0]
        sig = data[key].flatten().astype(np.float64)
        n_avail = len(sig) // segment_length
        n_take = min(n_avail, samples_per_class)
        for i in range(n_take):
            start = i * segment_length
            seg = sig[start : start + segment_length]
            X_list.append(seg.astype(np.float32))
            y_list.append(fault_name)
        for j in range(samples_per_class - n_take):
            start = (j % max(1, n_take)) * segment_length
            X_list.append(sig[start : start + segment_length].astype(np.float32))
            y_list.append(fault_name)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y, CLASS_NAMES


def compute_time_features(segments):
    """
    Compute 9 time-domain features from 2048-point segments.
    segments: (N, 2048) array
    Returns: (N, 9) array with columns [max, min, mean, sd, rms, skewness, kurtosis, crest, form]
    """
    eps = 1e-8
    max_val = np.max(segments, axis=1)
    min_val = np.min(segments, axis=1)
    mean_val = np.mean(segments, axis=1)
    sd_val = np.std(segments, axis=1)
    rms_val = np.sqrt(np.mean(segments**2, axis=1))
    skew_val = skew(segments, axis=1)
    kurt_val = kurtosis(segments, axis=1)
    crest_val = np.max(np.abs(segments), axis=1) / (rms_val + eps)
    form_val = rms_val / (np.mean(np.abs(segments), axis=1) + eps)
    return np.column_stack([max_val, min_val, mean_val, sd_val, rms_val, skew_val, kurt_val, crest_val, form_val])


def prepare_canonical_split(
    raw_dir="raw",
    segment_length=2048,
    samples_per_class=230,
    test_size=750,
    seed=1234,
    split_mode="temporal",
):
    """
    Load raw CWRU data, compute ML features, perform split, and scale raw data for DL.

    split_mode:
        - "temporal": split by position within each recording (first N train, next val, last M test).
          Avoids temporal/record-level leakage from same-recording segments in train and test.
        - "random": stratified random split (legacy behavior).

    Returns dict for use by both ML and DL notebooks.
    """
    X_raw, y, _ = load_cwru(raw_dir, segment_length, samples_per_class)
    X_ml = compute_time_features(X_raw)
    n = len(y)
    n_classes = len(CLASS_NAMES)
    le = LabelEncoder()
    le.fit(y)

    if split_mode == "temporal":
        # Per-class block indices: first train, then val, then test (no same-recording overlap)
        test_per_class = test_size // n_classes  # 75
        train_val_per_class = samples_per_class - test_per_class  # 155
        train_per_class = int(train_val_per_class * 0.8)  # 124
        val_per_class = train_val_per_class - train_per_class  # 31

        train_idx, val_idx, test_idx = [], [], []
        for c in range(n_classes):
            start = c * samples_per_class
            train_idx.extend(range(start, start + train_per_class))
            val_idx.extend(range(start + train_per_class, start + train_val_per_class))
            test_idx.extend(range(start + train_val_per_class, start + samples_per_class))

        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)
    else:
        # Random stratified split (legacy)
        indices = np.arange(n)
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=y, random_state=seed
        )
        y_train_val = y[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            stratify=y_train_val,
            random_state=seed,
        )
        train_idx = np.array(train_idx)
        val_idx = np.array(val_idx)
        test_idx = np.array(test_idx)

    X_raw_train = X_raw[train_idx]
    X_raw_val = X_raw[val_idx]
    X_raw_test = X_raw[test_idx]

    scaler = StandardScaler()
    X_raw_train = scaler.fit_transform(X_raw_train).astype(np.float32)
    X_raw_val = scaler.transform(X_raw_val).astype(np.float32)
    X_raw_test = scaler.transform(X_raw_test).astype(np.float32)

    y_train = le.transform(y[train_idx])
    y_val = le.transform(y[val_idx])
    y_test = le.transform(y[test_idx])

    return {
        "X_ml": X_ml,
        "y": y,
        "X_raw": X_raw,
        "X_raw_train": X_raw_train,
        "X_raw_val": X_raw_val,
        "X_raw_test": X_raw_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "class_names": CLASS_NAMES,
        "label_encoder": le,
        "scaler": scaler,
    }
