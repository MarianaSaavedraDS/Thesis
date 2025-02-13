#!/usr/bin/env python

# Imports
import pandas as pd
import pickle

from libs.paths import results_folder, data_folder
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

# Drop empty columns
chvnge_df = chvnge_df.drop(index=[491, 503])
chvnge_df = chvnge_df.reset_index(drop=True)

# Load predictions dynamically
signals = {signal_x: label_x, signal_y: label_y}

# Save processing steps:

# Load predictions dynamically

# Load the data from pickle
with open(results_folder / "combined_processed_predictions.pkl", "rb") as f:
    all_data = pickle.load(f)

# Initialize an empty dictionary to hold the processed predictions
processed_predictions = {}

# Extract the max temporal predictions for each signal and store them in processed_predictions
for signal in signals:
    processed_predictions[signal] = all_data[signal]["sequence"]

# Convert the processed_predictions into a Pandas DataFrame
# We assume each signal's predictions are arrays of the same length
processed_predictions_df = pd.DataFrame({signal: pd.Series(processed_predictions[signal]) for signal in signals})

# You can now work with the DataFrame
print(processed_predictions_df.head())

all_events = []
for i in range(len(processed_predictions[signal_x])):  # Assuming same length
    # Get the event names for both signals
    label_string, name_x, name_y = get_label_meaning(signal_x, signal_y, label_x, label_y)

    # Extract events for the first signal (with label_x and name_x)
    seq_x = ctilib.extract_label_events(processed_predictions[signal_x][i], target_labels=[label_x], event_name=name_x)

    # Extract events for the second signal (with label_y and name_y)
    seq_y = ctilib.extract_label_events(processed_predictions[signal_y][i], target_labels=[label_y], event_name=name_y)

    # Merge events from both signals, keeping the event names as keys
    merged_seq = {**seq_x, **seq_y}
    
    all_events.append(merged_seq)

# Print all events for further verification
print(f"Merged Sequence {i}: {merged_seq}")

# Compute intervals using different time points
interval_functions = {
    "Intervals_start_times": ctilib.compute_intervals_using_start_times,
    "Intervals_end_times": ctilib.compute_intervals_using_end_times,
    "Intervals_mid_times": ctilib.compute_intervals_using_midpoints,
}

intervals = {name: func(all_events, label_start=name_x, label_end=name_y) for name, func in interval_functions.items()}

# # Print the computed intervals for each method
# for interval_name, interval_values in intervals.items():
#     print(f"{interval_name}:")
#     for i, sequence_intervals in enumerate(interval_values):
#         print(f"  Sequence {i + 1}: {sequence_intervals}")
        
# Apply outlier filtering

# Define the physiological range for QS2 interval (in ms)
min_qs2_interval = 150  # Minimum QS2 interval in ms
max_qs2_interval = 600  # Maximum QS2 interval in ms

# Apply filtering for each interval type
filtered_intervals = {
    f"Filtered_{name}": ctilib.filter_intervals_within_range(interval_values, min_qs2_interval, max_qs2_interval)
    for name, interval_values in intervals.items()
}

# # Print the filtered intervals
# for interval_name, filtered_values in filtered_intervals.items():
#     print(f"{interval_name} (Filtered):")
#     for i, sequence_intervals in enumerate(filtered_values):
#         print(f"  Sequence {i + 1}: {sequence_intervals}")

# Compute averages
avg_intervals = {f"Avg_{name}": ctilib.compute_avg_intervals(vals) for name, vals in filtered_intervals.items()}
# Check the contents of avg_intervals


print(f"Intervals length: {[len(v) for v in intervals.values()]}")
print(f"Filtered Intervals length: {[len(v) for v in filtered_intervals.values()]}")
print(f"Avg Intervals length: {[len(v) for v in avg_intervals.values()]}")

import pandas as pd

interval_df = pd.DataFrame({
    'ID': chvnge_df['ID'].astype('Int64'),
    'Auscultation Point': chvnge_df['Auscultation Point'].astype(str),
    **{f'{name}': [events.get(str(name), []) for events in all_events] for name in [ name_x, name_y]},
    **intervals,
    **filtered_intervals,
    **avg_intervals,
})

print(interval_df.columns)

# Save the combined dictionary
estimates_file_path = results_folder / f"{label_string}_estimates.csv"
with open(estimates_file_path, 'wb') as f:
    pickle.dump(interval_df, f)

print(f"Results saved to: {estimates_file_path}")