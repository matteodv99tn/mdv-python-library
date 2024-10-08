import numpy as np
import matplotlib.pyplot as plt

from typing import Optional


class Demonstration:
    """
    A class that stores a demonstration data.

    This class stores time, position, velocity, and acceleration data, and provides methods to plot this data 
    and set it to follow a cosine function. It allows for comparison between two demonstrations.

    Attributes:
        t (np.ndarray): Time data, as an array of size (, ns).
        p (np.ndarray): Position data, as an array of size (np, ns).
        v (np.ndarray): Velocity data, as an array of size (nv, ns).
        a (np.ndarray): Acceleration data, as an array of size (na, ns).
    """

    def __init__(
        self,
        t: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        a: Optional[np.ndarray] = None
    ):
        self.t = t
        self.p = p
        self.v = v
        self.a = a

    # Import trivial functionalities that might offuscate the main logic of the class.
    from .demonstration_impl import (
        is_populated,
        ensure_is_populated,
        have_valid_shapes,
        ensure_have_valid_shapes,
    )

    def is_scalar(self) -> bool:
        """
        Check if the demonstration is scalar.

        This method checks if the position, velocity, and acceleration attributes are scalar values. 
        It returns True if all attributes are scalar, otherwise it returns False.

        Returns:
            bool: True if the demonstration is scalar, False otherwise.
        """
        self.ensure_have_valid_shapes()
        return all((
            len(self.p.shape) == 1,
            len(self.v.shape) == 1,
            len(self.a.shape) == 1,
        ))

    def pos_dimension(self) -> int:
        """
        Get the dimension of the position data.

        This method returns the number of entries in each position data sample.

        Returns:
            int: The dimension of the position data.
        """
        self.ensure_have_valid_shapes()
        return 1 if self.is_scalar() else self.p.shape[0]

    def vel_dimension(self) -> int:
        """
        Get the dimension of the velocity data.

        This method returns the number of entries in each velocity data sample.

        Returns:
            int: The dimension of the velocity data.
        """
        self.ensure_have_valid_shapes()
        return 1 if self.is_scalar() else self.v.shape[0]

    def plot(self, title: str = ""):
        """
        Plot the position, velocity, and acceleration of the demonstration over time.

        This method generates subplots for the position, velocity, and acceleration attributes, ensuring that 
        the shapes of these attributes are valid before plotting. It allows for an optional title to be added to 
        the plots for better context.

        Args:
            title (str, optional): The title to be prefixed to each subplot title. Defaults to an empty string.

        Returns:
            tuple: A tuple containing the figure and axes objects of the plot.

        Raises:
            ValueError: If the shapes of the demonstration fields are not valid.

        Note:
            This method does not call plt.show() to display the plot in a new window.
        """
        self.ensure_have_valid_shapes()

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        for i in range(self.p.shape[0]):
            axs[0].plot(self.t, self.p[i, :], label=f"p{i}")
        for i in range(self.v.shape[0]):
            axs[1].plot(self.t, self.v[i, :], label=f"v{i}")
            axs[2].plot(self.t, self.a[i, :], label=f"a{i}")

        if self.p.shape[0] > 1:
            axs[0].legend()
        if self.v.shape[0] > 1:
            axs[1].legend()
            axs[2].legend()

        prefix = f"{title} - " if title else ""
        axs[0].set_title(f"{prefix}Position")
        axs[1].set_title(f"{prefix}Velocity")
        axs[2].set_title(f"{prefix}Acceleration")
        axs[0].set_xlabel('Time [s]')

        return fig, axs

    def plot_compare(self, other: 'Demonstration'):
        self.ensure_have_valid_shapes()
        other.ensure_have_valid_shapes()

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axs[0].plot(self.t, self.p, label='Current')
        axs[0].plot(other.t, other.p, "--", label='Reference')
        axs[1].plot(self.t, self.v, label='Current')
        axs[1].plot(other.t, other.v, label='Reference')
        axs[2].plot(self.t, self.a, label='Current')
        axs[2].plot(other.t, other.a, label='Reference')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[0].set_title('Position')
        axs[1].set_title('Velocity')
        axs[2].set_title('Acceleration')
        plt.xlabel('Time [s]')

        return fig, axs

    def set_cosine(
        self,
        T: float = 10.0,
        n_samples: int = 100,
        omega: Optional[float | np.ndarray] = None
    ) -> 'Demonstration':
        """
        Set the demonstration based on a cosine function.

        This method generates time, position, velocity, and acceleration data based on a cosine function 
        with a specified period and number of samples. 
        If no angular frequency is provided, it is calculated based on the period T in order to follow
        cover 1/2 of such period in the given time.

        Args:
            T (float, optional): The period of the cosine function. Defaults to 10.0.
            n_samples (int, optional): The number of samples to generate. Defaults to 100.
            omega (Optional[float | np.ndarray], optional): The angular frequency or frequencies. 
                If None, it is calculated from T. Defaults to None.

        Returns:
            Demonstration: The updated demonstration object with the generated time, position, velocity, 
            and acceleration data.
        """

        if omega is None:
            omega = 0.5 * 2 * np.pi / T
        omega = np.atleast_1d(omega)
        self.t = np.linspace(0, T, n_samples)
        self.p = np.zeros((len(omega), n_samples))
        self.v = np.zeros((len(omega), n_samples))
        self.a = np.zeros((len(omega), n_samples))
        for i in range(len(omega)):
            self.p[i] = np.cos(omega[i] * self.t)
            self.v[i] = -omega[i] * np.sin(omega[i] * self.t)
            self.a[i] = -omega[i]**2 * np.cos(omega[i] * self.t)
        if self.p.shape[0] == 1: self.p = self.p.squeeze()
        if self.v.shape[0] == 1: self.v = self.v.squeeze()
        if self.a.shape[0] == 1: self.a = self.a.squeeze()
        return self


if __name__ == "__main__":
    # Create a cosine based demonstration with default parameters
    dem1 = Demonstration().set_cosine()
    assert dem1.is_scalar()

    # Create a cosine based demonstration with custom angular frequencies
    # and demonstration rate, and plot it
    dem2 = Demonstration().set_cosine(
        omega=np.array([1.0 / np.pi, 2.0 / np.pi, 3.0 / np.pi]), n_samples=200, T=15.0
    )
    print(dem2.pos_dimension())
    assert dem2.pos_dimension() == 3
    dem2.plot()
    plt.show()  # the plot does not automatically call plt.show()

    # Create demonstration with random data
    t = np.linspace(0, 10, 100)
    p = np.random.rand(3, 100)
    v = np.random.rand(3, 100)
    a = np.random.rand(3, 100)
    dem3 = Demonstration(t, p, v, a)
    assert dem3.pos_dimension() == 3

    # Update one field (e.g., the acceleration) with non-matching shaped data
    dem3.a = np.random.rand(3, 50)
    try:
        dem3.ensure_have_valid_shapes()
    except ValueError as e:
        print("Correctly caught the exception")
