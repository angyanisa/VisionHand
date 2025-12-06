#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

WINDOW_SIZE = 100
WINDOW_INCREMENT = 50
NUM_CHANNELS = 8

def sliding_windows(data, window_size, increment):
    for start in range(0, len(data) - window_size + 1, increment):
        yield data[start:start + window_size]

def extract_features(window):
    features = []
    for ch in range(NUM_CHANNELS):
        signal = window[:, ch]
        mean = np.mean(signal)
        std = np.std(signal)
        rms = np.sqrt(np.mean(signal**2))
        wl = np.sum(np.abs(np.diff(signal)))
        zc = np.sum((signal[:-1] * signal[1:] < 0).astype(int))
        ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
        features.extend([mean, std, rms, wl, zc, ssc])
    return features

def process_csv(file_path):
    df = pd.read_csv(file_path)
    if 'label' not in df.columns:
        raise ValueError("CSV file must contain 'label' column")

    windows = []
    labels = []

    for label in df['label'].unique():
        df_label = df[df['label'] == label].reset_index(drop=True)
        emg_data = df_label.iloc[:, :NUM_CHANNELS].to_numpy()

        for window in sliding_windows(emg_data, WINDOW_SIZE, WINDOW_INCREMENT):
            windows.append(extract_features(window))
            labels.append(label)

    print(f"Extracted {len(windows)} feature windows from {file_path}")
    return np.array(windows), np.array(labels)

def process_folder(data_dir):
    X_all, y_all = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                X, y = process_csv(os.path.join(data_dir, file))
                if X.size > 0 and y.size > 0:
                    X_all.append(X)
                    y_all.append(y)
                else:
                    print(f"Skipping {file}: No valid windows.")
            except ValueError as e:
                print(f"Skipping {file}: {e}")
    if not X_all:
        raise ValueError("No valid data found in folder.")
    return np.vstack(X_all), np.concatenate(y_all)



if __name__ == "__main__":
    X, y = process_folder("data")
    print(f"Total windows: {X.shape[0]}, Feature vector size: {X.shape[1]}")
