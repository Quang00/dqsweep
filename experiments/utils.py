import os
import numpy as np


# =============================================================================
# Helper Functions
# =============================================================================
def truncate_param(name: str, n: int = 3) -> str:
    """Truncates a parameter name to improve readability in plots.

    Args:
        name (str): Parameter name to truncate.
        n (int, optional): Number of tokens to keep. Defaults to 3.

    Returns:
        str: Truncated parameter name.
    """
    return " ".join(name.split("_")[:n])


def create_unique_dir(directory: str) -> str:
    """Creates a unique directory if one with the same name exists.

    Args:
        directory (str): Target directory path.

    Returns:
        str: Unique directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory
    counter = 1
    new_dir = f"{directory}_{counter}"
    while os.path.exists(new_dir):
        counter += 1
        new_dir = f"{directory}_{counter}"
    os.makedirs(new_dir)
    return new_dir


def parse_range(range_str: str) -> np.ndarray:
    """Parses a range string and returns a numpy array of values.

    Args:
        range_str (str): Range in format "start,end,points".

    Returns:
        np.ndarray: Array of evenly spaced values.
    """
    try:
        start, end, points = map(float, range_str.split(","))
        return np.linspace(start, end, int(points))
    except ValueError:
        raise ValueError("Invalid range format. Use 'start,end,points'.") from None


def extract_params_from_dm(dm):
    """
    Extract phase (phi) and amplitude (theta) from a density matrix.

    Args:
        dm (ndarray): 2x2 density matrix.

    Returns:
        tuple: (phi, theta) extracted values.
    """
    a = np.clip(np.sqrt(np.real(dm[0, 0])), 0.0, 1.0)
    theta = 2 * np.arccos(a)
    phi = (-np.angle(dm[0, 1]) if abs(dm[0, 1]) > 1e-12 else 0.0) % (2 * np.pi)
    return phi, theta
