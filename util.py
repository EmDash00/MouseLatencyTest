import numpy as np


def find_closest(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0:
        return np.full(y.shape, np.nan)

    sorted_x = np.sort(x)

    indices = np.searchsorted(sorted_x, y)

    left_idx = np.clip(indices - 1, 0, len(sorted_x) - 1)
    right_idx = np.clip(indices, 0, len(sorted_x) - 1)

    left_vals = sorted_x[left_idx]
    right_vals = sorted_x[right_idx]

    left_diff = np.abs(left_vals - y)
    right_diff = np.abs(right_vals - y)

    closest_indices_in_sorted = np.where(left_diff < right_diff, left_idx, right_idx)
    original_indices = np.argsort(x)

    closest_indices = original_indices[closest_indices_in_sorted]
    return closest_indices
