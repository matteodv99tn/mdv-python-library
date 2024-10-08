import numpy as np
import matplotlib.pyplot as plt

from .demonstration import Demonstration
from typing import Optional
from . import logger


def eval_gaussian_basis(
    x: float | np.ndarray, c: float | np.ndarray, h: float | np.ndarray
) -> float | np.ndarray:
    from ..concepts import is_floating, is_array

    def eval_single_basis(x: float, c: float, h: float) -> float:
        return np.exp(-h * (x - c)**2)

    def eval_whole_basis(x: float, c: np.ndarray, h: np.ndarray) -> np.ndarray:
        row = np.array([np.exp(-h[i] * (x - c[i])**2) for i in range(len(c))])
        return row / np.sum(row)

    assert (type(c) == type(h))
    if (is_floating(x) and is_floating(c)):
        return eval_single_basis(x, c, h)
    if (is_floating(x) and is_array(c)):
        return eval_whole_basis(x, c, h)
    if (is_array(x) and is_array(c)):
        Phi = np.zeros((len(x), len(c)))
        for i in range(len(x)):
            Phi[i, :] = eval_whole_basis(x[i], c, h)
        return Phi

    raise ValueError('Invalid input types')


class Dmp:
    """
    A class representing a Dynamic Movement Primitive (DMP) for trajectory generation.

    This class encapsulates the parameters and methods necessary to define and learn a DMP, 
    allowing for the integration of learned trajectories and the plotting of basis functions. 
    It provides functionality to compute desired forcing terms, learn weights from demonstrations, 
    and integrate the DMP over time.

    Attributes:
        alpha (float): The alpha parameter controlling the convergence rate.
        beta (float): The beta parameter controlling the attractor strength.
        gamma (float): The gamma parameter controlling the phase variable decay.
        n_basis (int): The number of basis functions used in the DMP.
        tau (Optional[float]): The time constant for the DMP.
        g (Optional[float | np.ndarray]): The goal configuration of the DMP.
        y0 (Optional[float | np.ndarray]): The starting configuration of the DMP.
        c (np.ndarray): The weight centers for the basis functions.
        h (np.ndarray): The widths of the basis functions.
        w (Optional[np.ndarray]): The learned weights for the DMP.
        y (Optional[float | np.ndarray]): The current position of the DMP.
    """


    def __init__(self, alpha: float, beta: float, gamma: float, n_basis: int = 15):
        """
        Initialize a Dynamic Movement Primitive (DMP) with specified parameters.

        This constructor sets up the DMP with the given alpha, beta, gamma, and number of basis functions. 
        It also initializes weight centers and checks that the provided parameters are valid.

        Args:
            alpha (float): The alpha parameter controlling the convergence rate.
            beta (float): The beta parameter controlling the attractor strength.
            gamma (float): The gamma parameter controlling the phase variable decay.
            n_basis (int, optional): The number of basis functions used in the DMP. Defaults to 15.

        Raises:
            ValueError: If alpha, beta, or gamma are not positive, or if n_basis is not a positive integer.
        """

        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.n_basis: int = n_basis

        self.tau: Optional[float] = None
        self.g: Optional[float | np.ndarray] = None
        self.y0: Optional[float | np.ndarray] = None

        # Weight centers
        self.c = np.array([
            np.exp(-self.gamma * i / (self.n_basis + 1)) for i in range(self.n_basis)
        ])
        self.h = np.zeros(n_basis)
        self.w = np.zeros(n_basis)
        for i in range(n_basis - 1):
            self.h[i] = 1.2 / (self.c[i + 1] - self.c[i])**2
        self.h[-1] = self.h[-2]

        self.w: Optional[np.ndarray] = None
        self.g: Optional[float | np.ndarray] = None
        self.y: Optional[float | np.ndarray] = None
        self.tau: Optional[float] = None

        # Parameter check
        if self.alpha <= 0: raise ValueError("alpha must be positive")
        if self.beta <= 0: raise ValueError("beta must be positive")
        if self.gamma <= 0: raise ValueError("gamma must be positive")
        if self.n_basis <= 0 and not isinstance(self.n_basis, int):
            raise ValueError("n_basis must be a positive integer")

    def construct_s(
        self, t: float | np.ndarray, tau: float = 1.0
    ) -> float | np.ndarray:
        """
        Calculate the phase variable based on time and the specified tau.

        This method computes the phase variable, which is used in the DMP, by applying an exponential decay 
        function to the input time values. The decay is controlled by the gamma parameter and the tau value.

        Args:
            t (float | np.ndarray): The time or array of time values for which to compute the phase variable.
            tau (float, optional): The time constant for the decay. Defaults to 1.0.

        Returns:
            float | np.ndarray: The computed phase variable as a float or an array of floats.
        """
        return np.exp(-self.gamma * t / tau)

    def compute_desired_forcing_term(self, dem: Demonstration) -> np.ndarray:
        """
        Calculate the desired forcing term based on the provided demonstration.

        This method computes the desired forcing term for the DMP using the acceleration, position, and velocity 
        data from the demonstration. It ensures that the demonstration is populated before performing the calculation.

        Args:
            dem (Demonstration): The demonstration object containing the trajectory data.

        Returns:
            np.ndarray: The computed desired forcing term as a numpy array.

        Raises:
            ValueError: If the demonstration is not populated.
        """
        dem.ensure_is_populated()
        g = dem.p[-1]
        tau = dem.tau()
        return tau**2 * dem.a - self.alpha * (self.beta * (g - dem.p) - dem.v)

    def learn_weights(self, dem: Demonstration) -> np.ndarray:
        """
        Learn the weights of the Dynamic Movement Primitive (DMP) from a demonstration.

        This method computes the weights for the DMP based on the provided demonstration data. 
        It ensures that the demonstration is populated and contains a sufficient number of samples before 
        calculating the desired forcing term and learning the weights.

        Args:
            dem (Demonstration): The demonstration object containing the trajectory data.

        Returns:
            np.ndarray: The learned weights for the DMP as a numpy array.

        Raises:
            ValueError: If the demonstration is not populated or if the number of samples is too low to learn the DMP.
        """

        dem.ensure_is_populated()

        # Check that demonstration contains a reasonable amount of samples
        if self.n_basis > round(1.5 * dem.samples_count()):
            raise ValueError("Number of samples is too low to learn the DMP")

        fd = self.compute_desired_forcing_term(dem)
        s = self.construct_s(dem.t, dem.tau())
        logger.info("Learning weights of a demonstration")
        logger.debug(f" - number of samples: {len(dem.t)}")
        logger.debug(f" - number of basis functions: {self.n_basis}")
        logger.debug(f" - dimensionsionality of each sample: {dem.pos_dimension()}")

        self.y0 = dem.p[0] if dem.is_scalar() else dem.p[:, 0]
        self.g = dem.p[-1] if dem.is_scalar() else dem.p[:, -1]
        self.tau = dem.tau()

        Phi = eval_gaussian_basis(s, self.c, self.h)
        f = fd.T / (s * (self.g - self.y0))
        self.w = np.linalg.lstsq(Phi, f)[0]
        return self.w

    def is_learned(self) -> bool:
        return all((
            self.w is not None,
            self.y0 is not None,
            self.g is not None,
        ))

    def integrate(
        self,
        dt: float,
        T: float,
        tau: Optional[float] = None,
        y0: Optional[float | np.ndarray] = None,
        g: Optional[float | np.ndarray] = None,
    ) -> Demonstration:
        """
        Integrate the Dynamic Movement Primitive (DMP) over a specified time duration.

        This method computes the trajectory of the DMP by integrating its dynamics over time, 
        using the learned weights and initial conditions. It returns a Demonstration object containing 
        the time, position, velocity, and acceleration data.

        Args:
            dt (float): The time step for the integration.
            T (float): The total time duration for the integration.
            tau (Optional[float], optional): The time constant for the DMP. Defaults to the DMP's tau, or T.
            y0 (Optional[float | np.ndarray], optional): The initial position. Defaults to the DMP's y0.
            g (Optional[float | np.ndarray], optional): The goal position. Defaults to the DMP's g.

        Returns:
            Demonstration: A Demonstration object containing the integrated trajectory data.

        Raises:
            ValueError: If the DMP has not been learned before integration.
        """
        if not self.is_learned():
            raise ValueError("DMP must be learned before integration")
        t = np.arange(0, T, dt)
        s = self.construct_s(t, T)
        tau = tau or self.tau or T
        g = g or self.g
        y0 = y0 or self.y0

        y = np.zeros_like(t)
        z = np.zeros_like(t)
        a = np.zeros_like(t)
        y[0] = y0
        z[0] = 0.0

        for i in range(len(t) - 1):
            Phi = eval_gaussian_basis(s[i], self.c, self.h)
            f = Phi @ self.w * (g-y0) * s[i]
            dz_dt = self.alpha * (self.beta * (g - y[i]) - z[i]) + f
            a[i] = dz_dt / self.tau
            z[i + 1] = z[i] + dt * dz_dt / self.tau
            y[i + 1] = y[i] + dt * z[i] / self.tau
        a[-1] = a[-2]

        return Demonstration(t, y, z / self.tau, a / self.tau)

    def plot_basis(self, T: Optional[float] = None, tau: Optional[float] = None):
        """
        Plot the basis functions and the phase variable of the Dynamic Movement Primitive (DMP).

        This method generates plots to visualize the relationship between the basis functions and the phase variable 
        over time. It allows for an optional total time duration and time constant to be specified for the plots.

        Args:
            T (Optional[float], optional): The total time duration for the plots. Defaults to the DMP's tau if not provided.
            tau (Optional[float], optional): The time constant for the DMP. Defaults to the DMP's tau if not provided.

        Returns:
            tuple: A tuple containing the figure and axes objects of the generated plots.
        """
        tau = tau or self.tau
        t = np.linspace(0, T or tau, 300)
        s = self.construct_s(t, tau)
        Phi = eval_gaussian_basis(s, self.c, self.h)
        fig, axs = plt.subplots(3, 1, figsize=(8, 6))

        axs[0].set_title("Basis VS Phase variable")
        axs[1].set_title("Basis VS Time")
        axs[2].set_title("Phase variable VS Time")

        for i in range(self.n_basis):
            axs[0].plot(s, Phi[:, i], label=f'Basis {i}')
            axs[1].plot(t, Phi[:, i], label=f'Basis {i}')

        axs[2].plot(t, s, label='Phase variable')
        axs[0].set_xlabel('coordinate system')
        axs[1].set_xlabel('time [s]')
        axs[2].set_xlabel('time [s]')
        axs[2].set_ylabel('Phase variable')
        axs[2].set_ylim(-0.1, 1.1)

        return fig, axs

    def describe_properties(self, prfx: str = "") -> str:
        y0_desc = str(self.y0) if (self.y0 is not None) else "not set"
        g_desc = str(self.g) if (self.g is not None) else "not set"
        lines = (
            f"{prfx}alpha: {self.alpha}",
            f"{prfx}beta: {self.beta}",
            f"{prfx}gamma: {self.gamma}",
            f"{prfx}number of basis: {self.n_basis}",
            f"{prfx}starting configuration: {y0_desc}",
            f"{prfx}goal configuration: {g_desc}",
        )
        return "\n".join(lines)


if __name__ == "__main__":

    import logging
    logger.setLevel(logging.WARNING)

    print("Loading reference data")
    dem = Demonstration().set_cosine()

    print("Creating a DMP object")
    alpha = 48.0
    beta = alpha / 4.0
    gamma = 2.0
    nb = 15  # Number of basis functions
    dmp = Dmp(alpha, beta, gamma, nb)
    print("dmp object properties:")
    print(dmp.describe_properties(" - "))  # Create bullet-list like description
    assert not dmp.is_learned()

    print("Learning weights")
    w = dmp.learn_weights(dem)
    assert len(w) == nb
    assert dmp.is_learned()
    # Now shall print also starting and goal configuration
    print(dmp.describe_properties(" - "))

    print("Integrating the DMP")
    exec_traj = dmp.integrate(0.01, dem.tau())

    print("Integration completed")
    # Plot both reference trajectory and one obtained by integration
    dem.plot("Reference")
    exec_traj.plot("DMP integration")

    dmp.plot_basis()
    plt.show()
