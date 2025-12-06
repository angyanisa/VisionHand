#!/usr/bin/env python3

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
from EMG_feature_extractor import process_folder

MODEL_PATH = "emg_model_lda.pkl"

def train_model():
    X, y = process_folder("data")
    print(f"Training model on {X.shape[0]} samples with {X.shape[1]} features")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    joblib.dump(lda, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()