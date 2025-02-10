#!/usr/bin/env python

# Imports
import copy
import numpy as np
import pandas as pd
import pickle

from libs.paths import results_folder, data_folder
from libs import feature_extraction_lib as ftelib
from libs import cti_interval_lib as ctilib
from libs.label_mappings import get_label_meaning

# Input files
signal_x = 'ecg'  # First signal
signal_y = 'pcg'  # Second signal (can be the same as signal_x)
label_x = 2       # Label from first signal
label_y = 2       # Label from second signal

label_string = get_label_meaning(signal_x, signal_y, label_x, label_y)
print(label_string)

# Load data
data_file_path = data_folder / "chvnge_df.pkl"
chvnge_df = pd.read_pickle(data_file_path)

# Load predictions dynamically
signals = {signal_x: label_x, signal_y: label_y}
predictions = {}

for signal in signals.keys():
    results_file_path = results_folder / f"{signal}_unet_predictions.pkl"
    with open(results_file_path, 'rb') as f:
        predictions[signal] = pickle.load(f)

print(f"Loaded predictions successfully: {type(predictions)}")

# Process predictions
processed_predictions = {
    sig: np.array([ftelib.max_temporal_modelling(ftelib.reverse_one_hot_encoding(pred)) for pred in pred_list], dtype=object)
    for sig, pred_list in predictions.items()
}

print(f"Processed predictions sizes: {len(processed_predictions[signal_x])}, {len(processed_predictions[signal_y])}")

# Extract events for both labels
all_events = []
for i in range(len(processed_predictions[signal_x])):  # Assuming same length
    seq_x = ctilib.extract_label_events(processed_predictions[signal_x][i], target_labels=[label_x])
    seq_y = ctilib.extract_label_events(processed_predictions[signal_y][i], target_labels=[label_y])
    
    merged_seq = {**seq_x, **seq_y}  # Merge events from both signals
    all_events.append(merged_seq)
    
print(f"chvnge_df length: {len(chvnge_df)}, all_events length: {len(all_events)}")


# Compute intervals using different time points
interval_functions = {
    "Intervals_start_times": ctilib.compute_intervals_using_start_times,
    "Intervals_end_times": ctilib.compute_intervals_using_end_times,
    "Intervals_mid_times": ctilib.compute_intervals_using_midpoints,
}

intervals = {name: func(all_events, label_start=label_x, label_end=label_y) for name, func in interval_functions.items()}

# Apply outlier filtering
filtered_intervals = {f"Filtered_{name}": ctilib.filter_outliers(vals) for name, vals in intervals.items()}

# Compute averages
avg_intervals = {f"Avg_{name}": ctilib.compute_avg_intervals(vals) for name, vals in filtered_intervals.items()}

# Compute the length of each signal
signal_lengths = chvnge_df['ECG Signal'].apply(len)  # Get length of each signal

# Get indices of the two shortest signals
smallest_indices = signal_lengths.nsmallest(2).index.tolist()

print(f"Indices of the two smallest signals: {smallest_indices}")

# Drop them from chvnge_df
chvnge_df = chvnge_df.drop(index=smallest_indices).reset_index(drop=True)

print(f"Updated chvnge_df length: {len(chvnge_df)}")

print(f"Intervals length: {[len(v) for v in intervals.values()]}")
print(f"Filtered Intervals length: {[len(v) for v in filtered_intervals.values()]}")
print(f"Avg Intervals length: {[len(v) for v in avg_intervals.values()]}")


interval_df = pd.DataFrame({
    **{f'{label}': [events.get(str(label), []) for events in all_events] for label in [label_x, label_y]},
    **intervals,
    **filtered_intervals,
    **avg_intervals,
    'ID': chvnge_df['ID'].astype('Int64'),
    'Auscultation Point': chvnge_df['Auscultation Point'].astype(str),
})

# Save the DataFrame as CSV
csv_file_path = results_folder / f"{label_string}_estimates.csv"
interval_df.to_csv(csv_file_path, index=False)

print(f"CSV file with intervals was saved in: {csv_file_path}")
