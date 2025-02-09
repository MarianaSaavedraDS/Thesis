"""
ECG and PCG feature extraction library extension.

@file: feature_extraction_extension.py

@coding: utf_8

@description: This module extends the feature_extraction_lib with custom 
functions necessary for ECG and PCG signals, without modifying the original 
library. It includes additional signal processing and feature extraction 
functions specific to the project.

@functions:
- process_pcg_signals_from_pkl: Processes PCG signals from a pickle file, 
  performs Schmidt despiking, Butterworth filtering, and extracts various 
  features (homomorphic envelope, CWT, Hilbert envelope). The function 
  returns a DataFrame of extracted features for each signal.
  
@see: feature_extraction_lib.py for the original signal processing and 
      feature extraction functions.

@version: 0.1
@createdBy: Mariana Lourenço
@creationDate: 2025-02-09
"""
import libs.feature_extraction_lib as ftelib
from libs import preprocessing_lib as pplib
import numpy as np
import pandas as pd

def process_pcg_signals_from_pkl(df):
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
