"""
Label Mapping Library.

@file: label_mappings.py

@coding: utf_8

@description: This module contains mappings for label values in ECG and PCG 
models, allowing easy conversion from numerical labels to meaningful 
descriptions. It also provides a function to retrieve a concatenated label 
meaning for interval-based analysis.

@functions:
- get_label_meaning: Given a signal type ('pcg' or 'ecg') and two label indices, 
  this function returns the concatenated meaning string (e.g., 'S1S2' for PCG or 
  'baseline segmento QRS' for ECG).

@constants:
- LABEL_MAP: A dictionary mapping numerical labels to their corresponding 
  descriptions for both PCG and ECG signals.

@version: 0.1
@createdBy: Mariana Lourenço
@creationDate: 2025-02-09
"""


# Mapping of model labels to their meanings

LABEL_MAP = {
    'PCG': {
        0: 'S1',
        1: 'diastol',
        2: 'S2',
        3: 'sistol'
    },
    'ECG': {
        0: 'baseline',
        1: 'P',
        2: 'Q',
        3: 'T'
    }
}

def get_label_meaning(signal_x, signal_y, label_x, label_y):
    """
    Given two signal types ('pcg' or 'ecg') and their corresponding label indices,
    return the concatenated meaning string (e.g., 'S1S2' if signal_x='pcg' and signal_y='pcg').
    """
    if signal_x not in LABEL_MAP or signal_y not in LABEL_MAP:
        raise ValueError(f"Invalid signal types: {signal_x}, {signal_y}. Choose from 'pcg' or 'ecg'.")

    # Get label meanings for both signals
    meaning_x = LABEL_MAP[signal_x].get(label_x, f"Unknown({label_x})")
    meaning_y = LABEL_MAP[signal_y].get(label_y, f"Unknown({label_y})")

    return f"{meaning_x}{meaning_y}", f"{meaning_x}",f"{meaning_y}"