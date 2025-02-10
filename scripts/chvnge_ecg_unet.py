#!/usr/bin/env python

# # Imports

# Standard libraries
import os

# Numerical and data processing libraries
import numpy as np
import pandas as pd
import pickle
from scipy import signal

# TensorFlow and Keras
from tensorflow import keras

# Custom libraries

from libs.paths import data_folder, results_folder, models_folder
from libs import feature_extraction_lib as ftelib
from libs.feature_extraction_lib_extension import process_ecg_features
from libs import preprocessing_lib as pplib

from libs import unet_model as unet

# # Input files

from models import config

BATCH_SIZE = config.BATCH_SIZE
patch_size = config.patch_size
nch = config.nch
stride = config.stride

# Load data
data_file_path = data_folder / "chvnge_df.pkl"
chvnge_df = pd.read_pickle(data_file_path)

# Create a new DataFrame by dropping the 'ECG Signal' column
ecg_df = chvnge_df.drop(columns=['PCG Signal'])


## Feature Extraction

FS = 500  # 500 sps original frequency

# Notch filter. Remove 50 Hz band
B, A = signal.iirnotch(50, Q=30, fs=FS)

features_df = process_ecg_features(ecg_df,B,A,FS)

feature_data = features_df[['Patient ID', 'Hilbert', 'Shannon', 'Homomorphic', 'Hamming']].to_numpy()

print(features_df.head())

# Create patches and structures for NN training
patched_features = ftelib.process_dataset_no_labels(feature_data, patch_size, stride)

# # Upload Model

# U-NET architecture

# Define the model architecture
model = unet.arch_unet(nch, patch_size=patch_size)

# Compile the model (this is necessary even if you are not training)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy',
              metrics=['CategoricalAccuracy', 'Precision', 'Recall'])


# Define checkpoint path
checkpoint_path = models_folder / "ecg_unet_weights" / "checkpoint.h5"

# Load weights if the file exists
if os.path.exists(checkpoint_path):
    try:
        model.load_weights(checkpoint_path)
        print("Weights loaded successfully!")
    except Exception as e:
        print("Error loading weights:", e)
else:
    print("Checkpoint file does not exist.")

# # Predictions
# Inference pipeline
ecg_pred = model.predict(patched_features)

# Reconstruct from patches

# Get original lengths from validation data
original_lengths = [len(seq) for seq in feature_data[:, 1]]
reconstructed_labels = ftelib.reconstruct_original_data(
    ecg_pred, original_lengths, patch_size, stride)

# Save results
results_file_path = results_folder / "ecg_unet_predictions.pkl"

with open(results_file_path, 'wb') as f:
    pickle.dump(reconstructed_labels, f)

print(f"Results saved to: {results_file_path}")