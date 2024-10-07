import logging


logger = logging.getLogger(__name__)


def is_populated(self) -> bool:
    """
    Check if all attributes of the demonstration are populated.

    This method verifies that the time, position, velocity, and acceleration attributes are not None. 
    It returns True if all attributes are populated, otherwise it returns False.

    Returns:
        bool: True if all attributes are populated, False otherwise.
    """

    return all((
        self.t is not None,
        self.p is not None,
        self.v is not None,
        self.a is not None,
    ))


def ensure_is_populated(self):
    """
    Ensure that all attributes of the demonstration are populated.

    This method checks if the time, position, velocity, and acceleration attributes are populated. 
    If any of these attributes are None, it raises a ValueError to indicate that the demonstration is incomplete.

    Raises:
        ValueError: If any fields of the Demonstration are not populated.
    """

    if not self.is_populated():
        raise ValueError("Not all fields of the Demonstration are populated.")


def have_valid_shapes(self) -> bool:
    """
    Check if the shapes of the demonstration attributes are valid.

    This method ensures that the demonstration is populated and then verifies that the number of time samples 
    matches the dimensions of the position, velocity, and acceleration arrays. 

    Returns:
        bool: True if the shapes of the attributes are valid, False otherwise.

    Raises:
        ValueError: If the demonstration is not populated.
    """

    self.ensure_is_populated()
    p_shape_id = len(self.p.shape) - 1
    v_shape_id = len(self.v.shape) - 1
    a_shape_id = len(self.a.shape) - 1
    return all((
        self.t.shape[0] == self.p.shape[p_shape_id],
        self.t.shape[0] == self.v.shape[v_shape_id],
        self.t.shape[0] == self.a.shape[a_shape_id],
    ))


def ensure_have_valid_shapes(self):
    """
    Verify that the shapes of the demonstration attributes are valid.

    This method checks if the shapes of the time, position, velocity, and acceleration attributes are valid. 
    If the shapes are not valid, it prints the shapes of each attribute and raises a ValueError.

    Raises:
        ValueError: If the shapes of the Demonstration fields are not valid.
    """

    if not self.have_valid_shapes():
        logger.warn("Shapes of the Demonstration fields:")
        logger.warn(f"t: {self.t.shape}")
        logger.warn(f"p: {self.p.shape}")
        logger.warn(f"v: {self.v.shape}")
        logger.warn(f"a: {self.a.shape}")
        raise ValueError("The shapes of the Demonstration fields are not valid")
