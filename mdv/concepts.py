import numpy as np


def is_floating(x) -> bool:
    """
    Determine if the given value is a floating-point number.

    This function checks whether the input is of type float or its numpy equivalents. 
    It returns True if the input is a floating-point number, otherwise False.

    Args:
        x: The value to be checked.

    Returns:
        bool: True if x is a floating-point number, False otherwise.
    """
    return np.issubdtype(type(x), np.floating) or isinstance(x, float)


def is_array(x) -> bool:
    """
    Determine if the provided value is a NumPy array.

    This function checks whether the input is an instance of a NumPy ndarray. 
    It returns True if the input is a NumPy array, otherwise it returns False.

    Args:
        x: The value to be checked.

    Returns:
        bool: True if x is a NumPy array, False otherwise.
    """
    return isinstance(x, np.ndarray)
