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
    'pcg': {
        0: 'S1',
        1: 'diastole',
        2: 'S2',
        3: 'sístole'
    },
    'ecg': {
        0: 'baseline',
        1: 'P',
        2: 'Q',
        3: 'T'
    }
}

def get_label_meaning(signal, label_x, label_y):
    """
    Given a signal type ('pcg' or 'ecg') and two label indices,
    return the concatenated meaning string (e.g., 'S1S2').
    """
    if signal not in LABEL_MAP:
        raise ValueError(f"Invalid signal type: {signal}. Choose 'pcg' or 'ecg'.")
    
    mapping = LABEL_MAP[signal]
    
    meaning_x = mapping.get(label_x, f"Unknown({label_x})")
    meaning_y = mapping.get(label_y, f"Unknown({label_y})")
    
    return f"{meaning_x}{meaning_y}"