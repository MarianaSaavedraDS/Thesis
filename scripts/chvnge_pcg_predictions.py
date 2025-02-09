#!/usr/bin/env python

# # Imports

# Standard libraries
import os
import sys

# Numerical and data processing libraries
import numpy as np
import pandas as pd
import pickle

# TensorFlow and Keras
from tensorflow import keras

# Custom libraries

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libs.paths import data_folder, results_folder, models_folder
from libs import feature_extraction_lib as ftelib
from libs.feature_extraction_lib_extension import process_pcg_signals_from_pkl

from libs import unet_model as unet

# # Input files

from libs import config

BATCH_SIZE = config.BATCH_SIZE
patch_size = config.patch_size
nch = config.nch
stride = config.stride

# Load data
data_file_path = data_folder / "chvnge_df.pkl"
chvnge_df = pd.read_pickle(data_file_path)

# Create a new DataFrame by dropping the 'ECG Signal' column
pcg_df = chvnge_df.drop(columns=['ECG Signal'])

print(pcg_df.head())

## Feature Extraction

features_df = process_pcg_signals_from_pkl(pcg_df)

features_df['Homomorphic'] = features_df['Features'].apply(lambda x: x[:, 0])
features_df['CWT_Morl'] = features_df['Features'].apply(lambda x: x[:, 1])
features_df['CWT_Mexh'] = features_df['Features'].apply(lambda x: x[:, 2])
features_df['Hilbert_Env'] = features_df['Features'].apply(lambda x: x[:, 3])
features_df = features_df.drop(columns=['Features'])

# Convert the loaded DataFrames to numpy arrays
feature_data = features_df[['ID', 'Homomorphic', 'CWT_Morl',
                   'CWT_Mexh', 'Hilbert_Env']].to_numpy()

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
checkpoint_path = models_folder / "pcg_unet_weights" / "checkpoint_wv.h5"

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
pcg_pred = model.predict(patched_features)

# Reconstruct from patches

# Get original lengths from validation data
original_lengths = [len(seq) for seq in feature_data[:, 1]]
reconstructed_labels = ftelib.reconstruct_original_data(
    pcg_pred, original_lengths, patch_size, stride)

# Save results
results_file_path = results_folder / "pcg_unet_predictions.pkl"

with open(results_file_path, 'wb') as f:
    pickle.dump(reconstructed_labels, f)

print(f"Results saved to: {results_file_path}")