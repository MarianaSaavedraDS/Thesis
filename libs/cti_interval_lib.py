"""
CTI Interval Extraction Library.

@file: cti_interval_lib.py

@coding: utf_8

@description: This module contains functions for extracting and processing 
intervals from label sequences in cardiovascular signal data. These 
functions can be used in analyzing and comparing time intervals in ECG and 
PCG signal processing.

@functions:
- extract_label_events: Extracts the start and end times of specific labels 
  (e.g., S1, S2) from a sequence of labeled events. The function returns 
  a dictionary with the start and end times for each label.
  
- compute_intervals_using_start_times: Computes the intervals between the 
  start times of two specified labels (e.g., S1 and S2) across multiple 
  signal sequences.

- compute_intervals_using_end_times: Computes the intervals between the 
  end time of one label and the start time of another label for all signal 
  sequences.

- compute_intervals_using_midpoints: Computes intervals based on the midpoints 
  of two labels, offering an alternative way to analyze the event timing.

- filter_outliers: Filters out intervals that are considered too short or too 
  long based on defined minimum and maximum interval thresholds.

- compute_avg_intervals: Computes the average interval for each signal 
  after outlier removal.

@version: 0.1
@createdBy: Mariana Lourenço
@creationDate: 2025-02-09
"""

import numpy as np

## Specific funtions for max temporal polling sequences

def extract_label_events(sequence, target_labels, event_name):
    """
    Extracts the start and end indices of specific labels from a labeled sequence.

    Parameters:
    sequence (list or np.array): The sequence of labels (e.g., 0=S1, 1=Systole, 2=S2, 3=Diastole)
    target_labels (list): A list of labels to extract start/end indices (e.g., [0, 2] for S1 and S2)
    event_name (str): The name of the event to be used as a key in the returned dictionary

    Returns:
    dict: A dictionary with event names as keys and start/end indices as values.
    """
    events = {event_name: []}  # Initialize dictionary with event name as key

    prev_label = None
    start_idx = None

    for i, label in enumerate(sequence):
        if label in target_labels:  # Look for the target labels
            if prev_label != label:  # New event detected
                start_idx = i
        if prev_label in target_labels and label != prev_label:  # Transition out of a target label
            end_idx = i - 1
            events[event_name].append(
                (start_idx, end_idx)  # Store the start and end indices
            )
        prev_label = label

    # Handle last segment if sequence ends with a target label
    if prev_label in target_labels and start_idx is not None:
        events[event_name].append(
            (start_idx, len(sequence) - 1)
        )

    return events

def compute_intervals_using_start_times(all_events, label_start, label_end, sample_rate=50):
    """
    Compute intervals between the start times of two labels for all sequences.

    Parameters:
    - all_events (list of dict): List of extracted events for each sequence.
                                  Each dict contains event timings for different labels.
    - label_start (str): The name of the starting event (e.g., 'Q').
    - label_end (str): The name of the ending event (e.g., 'S2').
    - sample_rate (int): The sample rate in Hz (default: 50 Hz).

    Returns:
    - intervals (list of lists): Intervals between the start times of the two labels for each sequence.
    """
    time_step_ms = (1 / sample_rate) * 1000  # Convert time step from seconds to milliseconds
    intervals = []

    for events in all_events:
        # Get the start times for the specified events
        start_times_q = [start for start, _ in events[label_start]]  # Start times of 'Q'
        start_times_s2 = [start for start, _ in events[label_end]]  # Start times of 'S2'
        
        label_intervals = []

        # Iterate through each start time of 'Q' and find the closest 'S2' start time
        end_idx = 0  # Pointer for 'S2' start times

        for start_time_q in start_times_q:
            # Find the first 'S2' event that occurs after the 'Q' event
            while end_idx < len(start_times_s2) and start_times_s2[end_idx] < start_time_q:
                end_idx += 1

            if end_idx < len(start_times_s2):
                # Calculate interval (in ms) between the 'Q' event and the next 'S2' event
                interval = (start_times_s2[end_idx] - start_time_q) * time_step_ms
                label_intervals.append(interval)

        intervals.append(label_intervals)  # Store intervals for this sequence

    return intervals

def compute_intervals_using_end_times(all_events, label_start, label_end, sample_rate=50):
    """
    Compute intervals between the end times of two labels for all sequences.

    Parameters:
    - all_events (list of dict): List of extracted events for each sequence.
                                  Each dict contains event timings for different labels.
    - label_start (str): The name of the starting event (e.g., 'Q').
    - label_end (str): The name of the ending event (e.g., 'S2').
    - sample_rate (int): The sample rate in Hz (default: 50 Hz).

    Returns:
    - intervals (list of lists): Intervals between the end times of the two labels for each sequence.
    """
    time_step_ms = (1 / sample_rate) * 1000  # Convert time step from seconds to milliseconds
    intervals = []

    for events in all_events:
        # Get the end times for the specified events
        end_times_q = [end for _, end in events[label_start]]  # End times of 'Q'
        end_times_s2 = [end for _, end in events[label_end]]  # End times of 'S2'
        
        label_intervals = []

        # Iterate through each end time of 'Q' and find the closest 'S2' end time
        end_idx = 0  # Pointer for 'S2' end times

        for end_time_q in end_times_q:
            # Find the first 'S2' event that occurs after the 'Q' event end time
            while end_idx < len(end_times_s2) and end_times_s2[end_idx] < end_time_q:
                end_idx += 1

            if end_idx < len(end_times_s2):
                # Calculate interval (in ms) between the 'Q' event end time and the next 'S2' event end time
                interval = (end_times_s2[end_idx] - end_time_q) * time_step_ms
                label_intervals.append(interval)

        intervals.append(label_intervals)  # Store intervals for this sequence

    return intervals

def compute_intervals_using_midpoints(all_events, label_start, label_end, sample_rate=50):
    """
    Compute intervals between the midpoints of two labels for all sequences.

    Parameters:
    - all_events (list of dict): List of extracted events for each sequence.
                                  Each dict contains event timings for different labels.
    - label_start (str): The name of the starting event (e.g., 'Q').
    - label_end (str): The name of the ending event (e.g., 'S2').
    - sample_rate (int): The sample rate in Hz (default: 50 Hz).

    Returns:
    - intervals (list of lists): Intervals between the midpoints of the two labels for each sequence.
    """
    time_step_ms = (1 / sample_rate) * 1000  # Convert time step from seconds to milliseconds
    intervals = []

    for events in all_events:
        # Calculate the midpoints for the specified events
        midpoints_q = [(start + end) / 2 for start, end in events[label_start]]  # Midpoints of 'Q'
        midpoints_s2 = [(start + end) / 2 for start, end in events[label_end]]  # Midpoints of 'S2'
        
        label_intervals = []

        # Iterate through each midpoint of 'Q' and find the closest 'S2' midpoint
        end_idx = 0  # Pointer for 'S2' midpoints

        for midpoint_q in midpoints_q:
            # Find the first 'S2' event that occurs after the 'Q' event midpoint
            while end_idx < len(midpoints_s2) and midpoints_s2[end_idx] < midpoint_q:
                end_idx += 1

            if end_idx < len(midpoints_s2):
                # Calculate interval (in ms) between the 'Q' event midpoint and the next 'S2' event midpoint
                interval = (midpoints_s2[end_idx] - midpoint_q) * time_step_ms
                label_intervals.append(interval)

        intervals.append(label_intervals)  # Store intervals for this sequence

    return intervals


# General fucntions

def filter_intervals_within_range(intervals, min_interval, max_interval):
    """
    Filters out intervals that are outside the physiological range.

    Parameters:
    - intervals (list of lists): List of intervals for each sequence.
    - min_interval (int): Minimum valid interval (in ms).
    - max_interval (int): Maximum valid interval (in ms).

    Returns:
    - filtered_intervals (list of lists): Intervals that are within the valid range.
    """
    filtered_intervals = []
    
    for sequence_intervals in intervals:
        # Filter out intervals outside the physiological range
        valid_intervals = [interval for interval in sequence_intervals if min_interval <= interval <= max_interval]
        filtered_intervals.append(valid_intervals)

    return filtered_intervals


def compute_avg_intervals(filtered_intervals):
    """
    Compute the average CTI interval for each subject.

    Parameters:
    - filtered_intervals (list of lists): The filtered CTI intervals for each signal.

    Returns:
    - avg_intervals (list): List of average CTI intervals for each subject.
    """
    avg_intervals = [
        np.mean(subject_intervals) if len(subject_intervals) > 0 else np.nan
        for subject_intervals in filtered_intervals
    ]
    return avg_intervals
