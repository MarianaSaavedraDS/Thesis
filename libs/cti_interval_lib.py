import numpy as np

def extract_label_events(sequence, target_labels, sample_rate=50):
    """
    Extracts the start and end times of specific labels from a labeled sequence.

    Parameters:
    sequence (list or np.array): The sequence of labels (e.g., 0=S1, 1=Systole, 2=S2, 3=Diastole)
    target_labels (list): A list of labels to extract start/end times (e.g., [0, 2] for S1 and S2)
    sample_rate (int): Sampling frequency in Hz (default: 50 Hz)

    Returns:
    dict: A dictionary with start/end times for each target label.
    """
    time_step = 1 / sample_rate  # Duration of each sample (20 ms for 50 Hz)
    events = {str(label): [] for label in target_labels}  # Initialize an empty list for each target label

    prev_label = None
    start_idx = None

    for i, label in enumerate(sequence):
        if label in target_labels:  # Look for the target labels
            if prev_label != label:  # New event detected
                start_idx = i
        if prev_label in target_labels and label != prev_label:  # Transition out of a target label
            end_idx = i - 1
            events[str(prev_label)].append(
                (start_idx * time_step, end_idx * time_step)  # Convert to seconds
            )
        prev_label = label

    # Handle last segment if sequence ends with a target label
    if prev_label in target_labels and start_idx is not None:
        events[str(prev_label)].append(
            (start_idx * time_step, len(sequence) * time_step)
        )

    return events

def compute_intervals_using_start_times(all_events, label_start, label_end):
    """
    Compute intervals between the start times of two labels for all sequences.

    Parameters:
    - all_events (list of dict): List of extracted events for each sequence.
                                  Each dict contains event timings for different labels.
    - label_start (int): The label for the starting event (e.g., 0 for S1).
    - label_end (int): The label for the ending event (e.g., 2 for S2).

    Returns:
    - intervals (list of lists): Intervals between the start times of the two labels for each sequence.
    """
    intervals = []

    for events in all_events:
        # Corrected: Access '0' and '2' directly as keys
        start_times = [start for start, _ in events[str(label_start)]]  # Get start times of label_start
        end_times = [start for start, _ in events[str(label_end)]]  # Get start times of label_end
        label_intervals = []

        end_idx = 0  # Pointer for end_times (label_end)

        for start_time in start_times:
            # Find the next end time (label_end) that comes after the current start time (label_start)
            while end_idx < len(end_times) and end_times[end_idx] < start_time:
                end_idx += 1

            if end_idx < len(end_times):
                interval = end_times[end_idx] - start_time  # Compute interval between start times
                label_intervals.append(interval)

        intervals.append(label_intervals)  # Store intervals for this sequence

    return intervals

def compute_intervals_using_end_times(all_events, label_start, label_end):
    """
    Compute intervals between the end time of one label and the start time of another label.

    Parameters:
    - all_events (list of dict): List of extracted events for each sequence.
                                  Each dict contains event timings for different labels.
    - label_start (int): The label for the starting event (e.g., 0 for S1).
    - label_end (int): The label for the ending event (e.g., 2 for S2).

    Returns:
    - intervals (list of lists): Intervals between the end time of label_start and the start time of label_end for each sequence.
    """
    intervals = []

    for events in all_events:
        # Corrected: Access '0' and '2' directly as keys
        start_end_times = [end for _, end in events[str(label_start)]]  # Get end times of label_start
        end_start_times = [start for start, _ in events[str(label_end)]]  # Get start times of label_end
        label_intervals = []

        end_idx = 0  # Pointer for end_start_times (label_end)

        for start_end_time in start_end_times:
            # Find the next end time (label_end) that comes after the current start_end time (label_start)
            while end_idx < len(end_start_times) and end_start_times[end_idx] < start_end_time:
                end_idx += 1

            if end_idx < len(end_start_times):
                interval = end_start_times[end_idx] - start_end_time  # Compute interval between end and start times
                label_intervals.append(interval)

        intervals.append(label_intervals)  # Store intervals for this sequence

    return intervals

def compute_intervals_using_midpoints(all_events, label_start, label_end):
    """
    Compute intervals between the midpoints of two labels for all sequences.

    Parameters:
    - all_events (list of dict): List of extracted events for each sequence.
                                  Each dict contains event timings for different labels.
    - label_start (int): The label for the starting event (e.g., 0 for S1).
    - label_end (int): The label for the ending event (e.g., 2 for S2).

    Returns:
    - intervals (list of lists): Intervals between the midpoints of label_start and label_end for each sequence.
    """
    intervals = []

    for events in all_events:
        # Corrected: Access '0' and '2' directly as keys
        start_midpoints = [(start + end) / 2 for start, end in events[str(label_start)]]  # Compute midpoints for label_start
        end_midpoints = [(start + end) / 2 for start, end in events[str(label_end)]]  # Compute midpoints for label_end
        label_intervals = []

        end_idx = 0  # Pointer for end_midpoints (label_end)

        for start_mid in start_midpoints:
            # Find the next end midpoint (label_end) that comes after the current start midpoint (label_start)
            while end_idx < len(end_midpoints) and end_midpoints[end_idx] < start_mid:
                end_idx += 1

            if end_idx < len(end_midpoints):
                interval = end_midpoints[end_idx] - start_mid  # Compute interval between midpoints
                label_intervals.append(interval)

        intervals.append(label_intervals)  # Store intervals for this sequence

    return intervals

def filter_outliers(cti_intervals, min_interval=0.15, max_interval=0.6):
    """
    Filter out intervals that are too long or too short based on a given threshold.

    Parameters:
    - cti_intervals (list of lists): The S1-S2 intervals for each signal.
    - min_interval (float): Minimum valid interval in seconds (default: 100 ms).
    - max_interval (float): Maximum valid interval in seconds (default: 300 ms).

    Returns:
    - filtered_intervals (list of lists): The filtered CTI intervals for each signal.
    """
    filtered_intervals = []

    for intervals in cti_intervals:
        # Filter intervals that are outside the specified range
        filtered_intervals.append([interval for interval in intervals if min_interval <= interval <= max_interval])

    return filtered_intervals

def compute_avg_intervals(filtered_intervals):
    """
    Compute the average CTI interval for each subject.

    Parameters:
    - filtered_intervals (list of lists): The filtered CTI intervals for each signal.

    Returns:
    - avg_intervals (list): List of average CTI intervals for each subject.
    """
    avg_intervals = [np.mean(subject_intervals) if len(subject_intervals) > 0 else None
                     for subject_intervals in filtered_intervals]
    return avg_intervals