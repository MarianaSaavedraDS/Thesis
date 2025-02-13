# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:22:58 2025

@author: danie
"""

import pandas as pd
import pickle

import libs.preprocessing_lib as pplib
from libs.paths import data_folder


def process_dataframe(df):
    """
    Processes the input dataframe and returns two separate dataframes for ECG and PCG signals.

    Parameters:
    df (pd.DataFrame): Original dataframe containing 'ID', 'Auscultation Point', 'ECG Signal', and 'PCG Signal'.
    downsample (function): Private function to downsample the PCG signal.

    Returns:
    df_ecg (pd.DataFrame): Processed ECG dataframe with z-score standardization.
    df_pcg (pd.DataFrame): Processed PCG dataframe with downsampling and z-score standardization.
    """

    # Separate ECG and PCG data
    df_ecg = df[['ECG Signal']]
    df_pcg = df[['PCG Signal']]

    # Remove empty lists
    df_ecg = df_ecg[df_ecg['ECG Signal'].apply(lambda x: len(x) > 0)]
    df_pcg = df_pcg[df_pcg['PCG Signal'].apply(lambda x: len(x) > 0)]

    # Apply downsampling only for IDs 1 to 108 (index 0 to 404)
    def downsample_pcg(signal, idx):
        if idx <= 404:  # IDs 1 to 108 correspond to index range 0 to 404
            return pplib.downsample(signal, orig_freq=8000, target_freq=3000)
        return signal  # Keep the rest unchanged

    df_pcg['PCG Signal'] = [downsample_pcg(sig, i)
                        for i, sig in enumerate(df_pcg['PCG Signal'])]

    # Apply z-score standardization
    df_ecg['ECG Signal'] = df_ecg['ECG Signal'].apply(
        lambda x: pplib.z_score_standardization(x) if len(x) > 1 else x)
    df_pcg['PCG Signal'] = df_pcg['PCG Signal'].apply(
        lambda x: pplib.z_score_standardization(x) if len(x) > 1 else x)
    
    df_merge = pd.concat([df['ID'],df['Auscultation Point'],df_ecg,df_pcg], axis=1)

    return df_merge

# Load data
data_file_path = data_folder / "chvnge_df.pkl"
chvnge_df = pd.read_pickle(data_file_path)

std_chvnge_df = process_dataframe(chvnge_df)

print(std_chvnge_df.head())

# Save results
data_file_path = data_folder / "std_chvnge_df.pkl"

with open(data_file_path, 'wb') as f:
    pickle.dump(std_chvnge_df, f)

print(f"Results saved to: {data_file_path}")