#!/usr/bin/env python

# # Imports

# Standard libraries
import os
import sys

# Numerical and data processing libraries
import numpy as np
import pandas as pd
import pickle

# Custom libraries

from libs.paths import data_folder, results_folder
from libs.label_mappings import get_label_meaning

from libs import unet_model as unet

# # Input files

signal = 'pcg'
label_x = 0  # Replace with your chosen label for x
label_y = 2  # Replace with your chosen label for y

label_string = get_label_meaning(signal, label_x, label_y)

print(label_string)  # Output: 'S1S2' for PCG, 'baseline segmento QRS' for ECG

# Load Estimates
data_file_path = results_folder / f"{signal}_{label_string}_estimates.csv"
est_intervals_df = pd.read_csv(data_file_path)  # Use read_csv instead of read_pickle for CSV files

print(est_intervals_df.columns)

# # Load Annotations

# Load Excel file from the correct path
data_file_path = data_folder / "Frederikke_Annotations.xlsx"
df_annotations = pd.read_excel(data_file_path)  # Load annotations from the file path

# Ensure column names match
df_annotations.rename(columns={"Patientid": "ID", "Valve": "Auscultation Point"}, inplace=True)

# Convert ID in both dataframes to the same format (remove 'id' prefix if needed)
df_annotations["ID"] = df_annotations["ID"].astype("Int64")

# Drop rows with 'ID' equal to 130 or 135 if they exist
df_annotations = df_annotations[~df_annotations['ID'].isin([130, 135])]

# Define the base columns to keep
base_columns = ['ID', 'Auscultation Point', 'Status_of_EF', 'HR (Heart rate)']

# Ensure label_string exists in the DataFrame before selecting it
if label_string in df_annotations.columns:
    selected_columns = base_columns + [label_string]  # Add label_string column
else:
    raise ValueError(f"Column '{label_string}' not found in df_annotations.")

# Create the new filtered DataFrame
base_annotations_df = df_annotations[selected_columns]

merged_df = est_intervals_df.merge(base_annotations_df, on=['ID', 'Auscultation Point'], how='inner')

print(merged_df.columns)

# Define the file path within the results folder, using variables in the file name
csv_file_path = results_folder / f"{label_string}_estimates_and_annotations.csv"

# Save the DataFrame as a CSV file
merged_df.to_csv(csv_file_path, index=False)
print(merged_df.head())

print(f"CVS file with estimates and annotations of the interval {label_string} was saved in: {csv_file_path}")