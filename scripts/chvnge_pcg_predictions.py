# # Imports

# Standard libraries
import os
import sys

# Numerical and data processing libraries
import numpy as np
import pandas as pd
import pickle

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Custom libraries

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from libs.paths import data_folder, results_folder, models_folder
from libs import preprocessing_lib as pplib
from libs import feature_extraction_lib as ftelib

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

def process_signals_from_pkl(df):
    """
    Process signals from a pickle file containing a DataFrame with processed signals.

    Parameters:
        file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: DataFrame with extracted features for each processed signal.
    """

    processed_features = []

    for _, row in df.iterrows():
        try:
            data = row['PCG Signal']

            # Schmidt despiking
            despiked_signal = pplib.schmidt_spike_removal(data, 1000)

            # Butterworth bandpass filtering
            filtered_pcg = pplib.butterworth_filter(
                despiked_signal, 'bandpass', 4, 1000, [15, 450])

            # Feature extraction
            # Homomorphic Envelope
            homomorphic = ftelib.homomorphic_envelope(filtered_pcg, 1000, 50)

            # CWT Scalogram Envelope
            cwt_morl = ftelib.c_wavelet_envelope(
                filtered_pcg, 1000, 50, interest_frequencies=[40, 200]
            )

            cwt_mexh = ftelib.c_wavelet_envelope(
                filtered_pcg, 1000, 50, wv_family='mexh',
                interest_frequencies=[40, 200]
            )

            # Hilbert Envelope
            hilbert_env = ftelib.hilbert_envelope(filtered_pcg, 1000, 50)

            # Organize and stack features
            features = np.column_stack((
                homomorphic, cwt_morl, cwt_mexh, hilbert_env
            ))

            # Append features to the list
            processed_features.append({
                'ID': row['ID'],
                'Auscultation Point': row['Auscultation Point'],
                'Features': features
            })
        except Exception as e:
            print(f"Error proccessing Patient ID {row['ID']}: {e}")
            continue  # Skip the file
    # Create a new DataFrame with features
    features_df = pd.DataFrame(processed_features)
    return features_df

features_df = process_signals_from_pkl(pcg_df)

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
def unet_pcg(nch, patch_size, dropout=0.05):
    inputs = tf.keras.layers.Input(shape=(patch_size, nch))
    conv1 = tf.keras.layers.Conv1D(
        8, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv1D(
        8, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    pool1 = tf.keras.layers.Dropout(dropout)(pool1)

    conv2 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    pool2 = tf.keras.layers.Dropout(dropout)(pool2)

    conv3 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv3)
    pool3 = tf.keras.layers.Dropout(dropout)(pool3)

    conv4 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv4)
    pool4 = tf.keras.layers.Dropout(dropout)(pool4)

    conv5 = tf.keras.layers.Conv1D(
        128, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv1D(
        128, 3, activation='relu', padding='same')(conv5)

    up6_prep = tf.keras.layers.UpSampling1D(size=2)(conv5)

    up6 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(64, 2, padding='same')(up6_prep), conv4], axis=2)
    up6 = tf.keras.layers.Dropout(dropout)(up6)
    conv6 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv1D(
        64, 3, activation='relu', padding='same')(conv6)

    up7_prep = tf.keras.layers.UpSampling1D(size=2)(conv6)

    up7 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(64, 2, padding='same')(up7_prep), conv3], axis=2)
    up7 = tf.keras.layers.Dropout(dropout)(up7)
    conv7 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv1D(
        32, 3, activation='relu', padding='same')(conv7)

    up8_prep = tf.keras.layers.UpSampling1D(size=2)(conv7)

    up8 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(32, 2, padding='same')(up8_prep), conv2], axis=2)
    up8 = tf.keras.layers.Dropout(dropout)(up8)
    conv8 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv1D(
        16, 3, activation='relu', padding='same')(conv8)

    up9_prep = tf.keras.layers.UpSampling1D(size=2)(conv8)

    up9 = tf.keras.layers.concatenate(
        [tf.keras.layers.Conv1D(8, 2, padding='same')(up9_prep), conv1], axis=2)
    up9 = tf.keras.layers.Dropout(dropout)(up9)
    conv9 = tf.keras.layers.Conv1D(
        8, 3, activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv1D(
        8, 3, activation='tanh', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv1D(4, 1, activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    return model

# Define the model architecture
model = unet_pcg(nch, patch_size=patch_size)

# Compile the model (this is necessary even if you are not training)
model.compile(optimizer=Adam(learning_rate=1e-4), 
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