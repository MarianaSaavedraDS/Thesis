"""
ECG and PCG feature extraction library extension.

@file: feature_extraction_extension.py

@coding: utf_8

@description: This module extends the feature_extraction_lib with custom 
functions necessary for the CHVNGE dataset's signals, without modifying the original 
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
from scipy import signal

def process_pcg_features(df):
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
            
            # Frequency downsampling
            data = pplib.downsample(data, 3000, 1000)

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

def process_ecg_features(dataset,B,A,FS):

    # Initialize lists to store data
    hilbert_list, shannon_list, homomorphic_list, hamming_list = [], [], [], []
    labels_list, ids_list = [], []

    for i, row in dataset.iterrows():
        try:
            # Signal processing
            ecg_raw = row.get('ECG Signal')
            patient_id = row.get('ID')

            # Bandpass
            ecg_bandpass = pplib.butterworth_filter(
                ecg_raw, 'bandpass', order=6, fs=FS, fc=[0.5, 100])

            # Notch. Remove 50 Hz
            ecg_notch = signal.filtfilt(B, A, ecg_bandpass)
            # Detrend
            ecg_notch -= np.median(ecg_notch)

            hilbert_env = ftelib.hilbert_envelope(
                ecg_notch, fs_inicial=FS, fs_final=50)
            shannon_env = ftelib.shannon_envelopenergy(
                ecg_notch, fs_inicial=FS, fs_final=50)
            homomorphic_env = ftelib.homomorphic_envelope(
                ecg_notch, median_window=21, fs_inicial=FS, fs_final=50)
            hamming_env = ftelib.hamming_smooth_envelope(
                ecg_notch, window_size=21, fs_inicial=FS, fs_final=50)



            ids_list.append(patient_id)
            hilbert_list.append(hilbert_env)
            shannon_list.append(shannon_env)
            homomorphic_list.append(homomorphic_env)
            hamming_list.append(hamming_env)


        except Exception as e:
            print(f"Error processing patient ID {dataset.ID[i]}: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame({
        'Patient ID': ids_list,
        'Hilbert': hilbert_list,
        'Shannon': shannon_list,
        'Homomorphic': homomorphic_list,
        'Hamming': hamming_list,
    })
    return df
