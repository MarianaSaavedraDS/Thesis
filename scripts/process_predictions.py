#!/usr/bin/env python

# Imports
import copy
import numpy as np
import pandas as pd
import pickle

from libs.paths import results_folder, data_folder
from libs import feature_extraction_lib as ftelib
from libs import preprocessing_lib as pplib
from libs import cti_interval_lib as ctilib
from libs.label_mappings import get_label_meaning

# Input files
signal_x = 'ECG'  # First signal
signal_y = 'PCG'  # Second signal (can be the same as signal_x)
label_x = 2       # Label from first signal
label_y = 2       # Label from second signal

label_string, name_x, name_y = get_label_meaning(signal_x, signal_y, label_x, label_y)
print(label_string)

# Load data
data_file_path = data_folder / "std_chvnge_df.pkl"
chvnge_df = pd.read_pickle(data_file_path)

# Load predictions dynamically
signals = {signal_x: label_x, signal_y: label_y}
predictions = {}

for signal in signals.keys():
    results_file_path = results_folder / f"{signal}_unet_predictions.pkl"
    with open(results_file_path, 'rb') as f:
        predictions[signal] = pickle.load(f)

print(f"Loaded predictions successfully: {type(predictions)}")

for sig, pred_list in predictions.items():
    print(f"Signal: {sig}")
    print(f"Type of pred_list: {type(pred_list)}")
    if isinstance(pred_list, list):
        print(f"First element type: {type(pred_list[0])}")
        print(f"Shape of first element: {getattr(pred_list[0], 'shape', 'N/A')}")

# Moving Average
window_size = 5

# Apply Moving Average to each column separately
smoothed_predictions = {
    sig: np.array([
        np.apply_along_axis(lambda col: pplib.moving_average(col, window_size), axis=0, arr=pred)
        for pred in pred_list
    ], dtype=object)
    for sig, pred_list in predictions.items()
}

print(f"Smoothed predictions size: {len(smoothed_predictions[signal_x])}, and shape of elements: {getattr(smoothed_predictions[signal_x][0], 'shape', 'N/A')}")

sequenced_predictions = {
    signal_x: np.array([ftelib.reverse_one_hot_encoding(pred) for pred in smoothed_predictions[signal_x]], dtype=object),
    signal_y: np.array([ftelib.max_temporal_modelling(ftelib.reverse_one_hot_encoding(pred)) for pred in smoothed_predictions[signal_y]], dtype=object)
}

print(f"Temporal predictions sizes: {len(sequenced_predictions[signal_x])}, and shape of elements:{getattr(sequenced_predictions[signal_x][0], 'shape', 'N/A')}")

# Save processing steps:
# Combine both signals into a single dictionary
all_processed_predictions = {
    signal_x: {
        "smoothed": smoothed_predictions[signal_x],
        "sequence": sequenced_predictions[signal_x]
    },
    signal_y: {
        "smoothed": smoothed_predictions[signal_y],
        "sequence": sequenced_predictions[signal_y]
    }
}

# Save the combined dictionary
combined_results_file_path = results_folder / "combined_processed_predictions.pkl"
with open(combined_results_file_path, 'wb') as f:
    pickle.dump(all_processed_predictions, f)

print(f"Results saved to: {combined_results_file_path}")
