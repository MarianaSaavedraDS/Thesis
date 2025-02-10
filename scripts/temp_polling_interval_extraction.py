#!/usr/bin/env python

# # Imports

# Standard libraries
import os
import sys
import copy

# Numerical and data processing libraries
import numpy as np
import pandas as pd
import pickle

# Custom libraries

from libs.paths import results_folder, data_folder
from libs import feature_extraction_lib as ftelib
from libs import cti_interval_lib as ctilib

# # Input files

signal = "pcg"

# Define the labels (x and y) that you want to extract
label_x = 0  # Replace with your chosen label for x
label_y = 2  # Replace with your chosen label for y

# Load data
data_file_path = data_folder / "chvnge_df.pkl"
chvnge_df = pd.read_pickle(data_file_path)

# Define the path to the pickle file
results_file_path = results_folder / f"{signal}_unet_predictions.pkl"

# Load the pickle file
with open(results_file_path, 'rb') as f:
    predictions = pickle.load(f)

# Print or inspect the loaded data
print(f"Loaded predictions succefully: {type(predictions)}")

# Automatic CTI extraction: Max-temporal-polling

# Perform Max temporal Poilling on predictions

pred_labels = [ftelib.reverse_one_hot_encoding(pred) for pred in predictions]

prediction_labels = copy.deepcopy(pred_labels)

predictions = np.array([ftelib.max_temporal_modelling(prediction) for prediction in prediction_labels], dtype=object)

# Example: Apply extraction function to each predicted sequence

target_labels = [label_x, label_y]  # Extract (0) and (2) events

# Example: Apply extraction function to each predicted sequence for S1 and S2 (labels 0 and 2)
all_events = []
for seq in predictions:
    seq_events = ctilib.extract_label_events(seq, target_labels=[0, 2])  # Extract for S1 and S2
    all_events.append(seq_events)


### Start

# Example: Compute intervals between S1 (0) and S2 (2)
intervals_start_times = ctilib.compute_intervals_using_start_times(all_events, label_start=0, label_end=2)

# Example: Compute intervals between S1 (0) and S2 (2) using end times
intervals_end_times = ctilib.compute_intervals_using_end_times(all_events, label_start=0, label_end=2)

# Example: Compute intervals between S1 (0) and S2 (2) using midpoints
intervals_mid_times = ctilib.compute_intervals_using_midpoints(all_events, label_start=0, label_end=2)

# Apply outlier filtering to intervals based on start times
filtered_intervals_start = ctilib.filter_outliers(intervals_start_times)

# Apply outlier filtering to intervals based on end times
filtered_intervals_end = ctilib.filter_outliers(intervals_end_times)

# Apply outlier filtering to intervals based on midpoints
filtered_intervals_mid = ctilib.filter_outliers(intervals_mid_times)

# # Iterate over all subjects and print their filtered S1-S2 intervals
# for idx, subject_intervals in enumerate(filtered_intervals_start):
#     print(f"Subject {idx + 1} S1-S2 intervals:", subject_intervals)

# Compute the average S1-S2 intervals for each subject based on start times
avg_intervals_start = ctilib.compute_avg_intervals(filtered_intervals_start)

# Compute the average S1-S2 intervals for each subject based on end times
avg_intervals_end = ctilib.compute_avg_intervals(filtered_intervals_end)

# Compute the average S1-S2 intervals for each subject based on midpoints
avg_intervals_mid = ctilib.compute_avg_intervals(filtered_intervals_mid)

# Create the DataFrame with actual label names
# Assuming all_events is a list of dictionaries with label event times, and you have the target_labels

chvnge_df = chvnge_df[~chvnge_df['ID'].isin([130, 135])]

# Create the DataFrame with extracted event times
interval_df = pd.DataFrame({
    f'{label}': [events.get(str(label), []) for events in all_events]  # Dynamically handle any label
    for label in target_labels  # Iterate over each label in target_labels
})

# Add interval columns to the DataFrame (these can also be generalized if needed)
interval_df['Intervals_start_times'] = intervals_start_times
interval_df['Intervals_mid_times'] = intervals_mid_times
interval_df['Intervals_end_times'] = intervals_end_times
interval_df['Filtered_intervals_start'] = filtered_intervals_start
interval_df['Filtered_intervals_mid'] = filtered_intervals_mid
interval_df['Filtered_intervals_end'] = filtered_intervals_end
interval_df['Avg_intervals_start'] = avg_intervals_start
interval_df['Avg_intervals_mid'] = avg_intervals_mid
interval_df['Avg_intervals_end'] = avg_intervals_end
interval_df['ID'] = chvnge_df['ID'].astype('Int64')

print(interval_df['ID'])

# Define the file path within the results folder, using variables in the file name
csv_file_path = results_folder / f"{signal}_{label_x}_{label_y}_intervals.csv"

# Save the DataFrame as a CSV file
interval_df.to_csv(csv_file_path, index=False)

print(f"CVS File with intervals was saved in: {csv_file_path}")