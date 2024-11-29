import sys
import math
import logging
import numpy as np
import casadi as ca

from typing import Optional
from .. import default_logger_formatter
from . import Dmp, Demonstration


class DmpOptimisationProblemBase:
    """
    Base class for defining a DMP optimization problem. 
    
    This class sets up the necessary parameters and logging for optimizing Dynamic Movement Primitives (DMPs).

    This class of problems is particularly targetted to point-to-point (discrete) DMPs;
    to learn more about the fundamental equation of DMPs, refer to the documentation of the :class:`mdv.dmp.Dmp` class.

    Note:
        In the following code, the following shorthand notations are used:
        - `s` is the canonical system variable
        - `psi` is the vector of basis functions
        - `f` is the forcing function of the DMP

    Note:
        The shorthand symbols are:
        - `nb` is the number of basis functions
        - `np` the number of dimensions of the position state
        - `nk` (or `N`) the number of time steps for numerical integration (for the optimization problem)

    Attributes:
        logger (Logger): Logger for tracking optimization process.
        alpha (float): DMP alpha parameter.
        beta (float): DMP beta parameter.
        gamma (float): DMP gamma parameter.
        c (np.ndarray): Centers of the basis functions.
        h (np.ndarray): Widths of the basis functions.
        tau (float): Time scaling factor for the DMP.
        nb (int): Number of basis functions.
        w (np.ndarray): Weights of the basis functions.
        nlp_prob: Non-linear programming problem instance.

    Args:
        dmp (Dmp): An instance of the Dmp class containing DMP parameters.

    Examples:
        >>> dmp = Dmp(...)
        >>> problem = DmpOptimisationProblemBase(dmp)
    """

    def __init__(self, dmp: Dmp):
        # Set up logger
        self.logger = logging.getLogger("DmpOptimisationProblem")
        self.logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(default_logger_formatter)
        self.logger.addHandler(stdout_handler)

        # Retrieve parameters from the DMP object
        self.alpha: float = dmp.alpha
        self.beta: float = dmp.beta
        self.gamma: float = dmp.gamma
        self.c: np.ndarray = dmp.c
        self.h: np.ndarray = dmp.h
        self.tau: float = dmp.tau
        self.nb: int = dmp.n_basis

        self.w: np.ndarray = dmp.w
        self.nlp_prob = None

    def construct_forcing_function(self) -> ca.Function:
        """
        Construct the forcing function of the DMP

        The forcing function is defined as:

        .. math:: 

           f(x) = \\frac{\sum_{j=1}^{nb} w_j  \psi_j(x)}{\sum_{j=1}^{nb} \psi_j(x)}
        """
        s = ca.MX.sym('s')

        psi = ca.MX.zeros(self.nb)
        for j in range(self.nb):
            psi[j] = ca.exp(-self.h[j] * (s - self.c[j])**2)

        w = ca.MX.sym('w', self.nb)
        f = ca.dot(w, psi) / ca.sum1(psi)
        return ca.Function('forcing_function', [s, w], [f])

    def _prepare_nlp_props(
        self,
        wmin: Optional[np.ndarray] = None,
        wmax: Optional[np.ndarray] = None,
        wguess: Optional[np.ndarray] = None,
        options: Optional[dict] = None
    ):
        """
        Prepare the properties for the non-linear programming (NLP) problem, by constructing an appropriate dictionary. 

        This method sets the bounds and initial guesses for the weights used in the optimization process, 
        ensuring they align with the number of basis functions.

        Args:
            wmin (Optional[np.ndarray]): 
                Minimum bounds for the weights. If provided, its length must match the number of basis functions.
            wmax (Optional[np.ndarray]): i
                Maximum bounds for the weights. If provided, its length must match the number of basis functions.
            wguess (Optional[np.ndarray]): Initial guess for the weights.
            options (dict): Additional options for configuring NLP properties.

        Raises:
            ValueError: If the dimensions of wmin or wmax do not match the number of basis functions.

        Returns:
            dict: A dictionary containing the prepared NLP properties, including bounds and initial guesses.

        Examples:
            >>> nlp_props = self._prepare_nlp_props(wmin=np.array([-1, -1]), wmax=np.array([1, 1]))
        """
        options = options or {}

        if wmin is not None and wmin.shape[-1] != self.nb:
            raise ValueError(
                f"The length of wmin ({wmin.shape[-1]}) must be equal "
                "to the number of basis functions ({self.nb})"
            )
        if wmax is not None and wmax.shape[-1] != self.nb:
            raise ValueError(
                f"The length of wmax ({wmax.shape[-1]}) must be equal "
                "to the number of basis functions ({self.nb})"
            )

        # Construct NLP bounds properties
        self.nlp_props = {
            "tau0": options.get("tau0", self.tau or 1.0),
            "tau_min": options.get("tau_min", self.tau or 0.1),
            "w0": options.get("w0", wguess or self.w),
            "wmin": options.get("wmin", wmin or self.nb * [-ca.inf]),
            "wmax": options.get("wmax", wmax or self.nb * [ca.inf]),
            "vmax": options.get("vmax", 1.0),
            "vf_max": options.get("vf_max", 0.1),
            "ef_max": options.get("ef_max", 0.1)
        }
        return self.nlp_props

    def _print_nlp_properties(self, d: dict):
        """
        Print the properties of the non-linear programming (NLP) problem in a formatted table. 
        
        This method provides a clear overview of the current properties and their values, making it easier to understand the optimization setup.

        Args:
            d (dict): 
                Dictionary containing NLP properties to print, where keys are property names and values are their corresponding values.
                If not provided, the default NLP properties are used.

        Returns:
            None: This method does not return any value; it directly prints the formatted table to the console.

        Examples:
            >>> self._print_nlp_properties(nlp_props)
        """
        from tabulate import tabulate
        from ..concepts import is_floating

        d = d or self.nlp_props
        entries = []
        for k, v in d.items():
            if is_floating(v):
                entries.append((k, f"{v:.4f}"))
            else:
                line = ", ".join(f"{e:.1f}" for e in v)
                entries.append((k, line))

        print(tabulate(entries, headers=["Property", "Value"], tablefmt="rst"))


class ScalarDmpOptimProblem(DmpOptimisationProblemBase):

    def __init__(self, dmp: Dmp, w: Optional[np.ndarray] = None):
        if not dmp.is_scalar():
            raise ValueError("The DMP must be scalar for this optimisation problem")
        super().__init__(dmp)

    def write_nlp_problem(
        self,
        N: int,
        nlpsol_opts: Optional[dict] = None,
    ) -> ca.Function:
        """
        Construct and configure a non-linear programming (NLP) problem for optimizing Dynamic Movement Primitives (DMPs). 
        This method sets up the necessary variables, constraints, and parametric initial conditions 
        for the optimization process based on the specified number of time steps.

        Args:
            N (int): The number of time steps for the numerical integration.
            nlpsol_opts (Optional[dict]): Additional options for the NLP solver configuration.

        Returns:
            ca.Function: The configured NLP problem ready for solving.

        Examples:
            >>> opti_prob = ScalarDmpOptimProblem(dmp)
            >>> nlp_problem = opti_problem.write_nlp_problem(N=100)

            To provide solver options, you can pass them as a dictionary to the `nlpsol_opts` argument, e.g.

            .. code-block:: python

               jit_options = {"flags": ["-O3"], "verbose": False, "compiler": "gcc"}
               options = {
                   "jit": False,
                   "compiler": "shell",
                   "jit_options": jit_options,
                   "verbose": False
               }
               opti_problem.write_nlp_problem(200, options)



        
        """

        self.nk = N
        nlp_g = []  # constraints
        nlp_p = []  # parameters

        # Configure parameters
        y0 = ca.MX.sym('y0')
        g = ca.MX.sym('g')
        nlp_p += [y0, g]

        # Configure optimsation variables
        tau = ca.MX.sym('tau')
        w = ca.MX.sym('w', self.nb)
        Zs = ca.MX.sym('zs', N + 1)
        Ys = ca.MX.sym('ys', N + 1)
        nlp_x = ca.vertcat(tau, w, Zs, Ys)

        # Extra variables
        dt = float(1 / N)
        psi = ca.MX.zeros(self.nb)
        zcurr = ca.MX.sym('zcurr')
        ycurr = ca.MX.sym('ycurr')
        gg = ca.MX.sym('gg')
        ff = ca.MX.sym('forcing_term')

        # DMP Dynamics
        z_dot = self.alpha * (self.beta * (gg-ycurr) - zcurr) + ff
        y_dot = zcurr
        z_ode = ca.Function('z_ode', [zcurr, ycurr, ff, gg], [z_dot])
        y_ode = ca.Function('y_ode', [zcurr], [y_dot])

        # Initial Condition
        nlp_g += [Zs[0], Ys[0] - y0]

        # Dynamic constraints
        for i in range(N):
            # Using MX expressions, it is not possible to rely on a function to compute
            # the forcing term, since we can't insert the a-priori knowledge of the
            # coordinate system variable. So we compute it each time.
            s = math.exp(-self.gamma * i * dt)
            for j in range(self.nb):
                psi[j] = ca.exp(-self.h[j] * (s - self.c[j])**2)
            f = ca.dot(w, psi) / ca.sum1(psi) * (g-y0) * s

            zk_next = Zs[i] + dt * z_ode(Zs[i], Ys[i], f, g)
            yk_next = Ys[i] + dt * y_ode(Zs[i])
            nlp_g += [Zs[i + 1] / tau, Ys[i + 1] - yk_next, Zs[i + 1] - zk_next]

        # Final condition
        nlp_g += [Zs[N] / tau, Ys[N] - g]

        self.prob_config = {
            'f': tau, 'x': nlp_x, 'g': ca.vertcat(*nlp_g), 'p': ca.vertcat(*nlp_p)
        }
        opts = nlpsol_opts or {}
        self.nlp_prob = ca.nlpsol('solver', 'ipopt', self.prob_config, opts)
        return self.nlp_prob

    def _nlp_parameters(
        self,
        y0: float,
        g: float,
        wmin: Optional[np.ndarray] = None,
        wmax: Optional[np.ndarray] = None,
        wguess: Optional[np.ndarray] = None,
        options: Optional[dict] = None
    ):
        """
        Prepare the parameters and bounds for the non-linear programming (NLP) problem. 
        
        This method sets up the initial conditions, variable bounds, and other properties 
        necessary for solving the optimization problem based on the specified initial and goal states.

        Args:
            y0 (float): The initial state of the system.
            g (float): The goal state of the system.
            wmin (Optional[np.ndarray]): 
                Minimum bounds for the weights. 
                If provided, its length must match the number of basis functions.
            wmax (Optional[np.ndarray]): 
                Maximum bounds for the weights. 
                If provided, its length must match the number of basis functions.
            wguess (Optional[np.ndarray]): Initial guess for the weights.
            options (Optional[dict]): Additional options for configuring NLP properties.

        Raises:
            RuntimeError: If the NLP problem has not been defined prior to calling this method.
            ValueError: If the lengths of wmin or wmax do not match the number of basis functions.

        Returns:
            tuple: A tuple containing the parameters, initial guess, lower bounds, 
                upper bounds, lower constraints, upper constraints, and NLP properties.

        Examples:
            >>> params, x0, lbx, ubx, lbg, ubg, nlp_props = self._nlp_parameters(y0=0.0, g=1.0)
        """

        if self.nlp_prob is None:
            raise RuntimeError(
                "The NLP problem has not been written yet. "
                "Call write_nlp_problem() first"
            )

        options = options or {}
        self._prepare_nlp_props(wmin, wmax, wguess, options)

        # Set NLP parameters
        params = [y0, g]

        if wmin is not None and len(wmin) != self.nb:
            raise ValueError(
                f"The length of wmin ({len(wmin)}) must be equal "
                "to the number of basis functions ({self.nb})"
            )
        if wmax is not None and len(wmax) != self.nb:
            raise ValueError(
                f"The length of wmax ({len(wmax)}) must be equal "
                "to the number of basis functions ({self.nb})"
            )

        # Construct NLP bounds properties
        nlp_props = self._prepare_nlp_props(wmin, wmax, wguess)
        nlp_props["y0"] = y0
        nlp_props["g"] = g

        # Guess and bounds on the optimisation variable
        x0 = [nlp_props["tau0"], *nlp_props["w0"]]
        lbx = [nlp_props["tau_min"], *nlp_props["wmin"]]
        ubx = [ca.inf, *nlp_props["wmax"]]
        x0 += 2 * (self.nk + 1) * [0.0]
        lbx += 2 * (self.nk + 1) * [-ca.inf]
        ubx += 2 * (self.nk + 1) * [ca.inf]

        # Bounds on the constraints
        lbg = [0.0, 0.0] + self.nk * [-nlp_props["vmax"], 0.0, 0.0]
        ubg = [0.0, 0.0] + self.nk * [nlp_props["vmax"], 0.0, 0.0]
        lbg += [-nlp_props["vf_max"], -nlp_props["ef_max"]]
        ubg += [nlp_props["vf_max"], nlp_props["ef_max"]]

        return params, x0, lbx, ubx, lbg, ubg, nlp_props

    def solve(
        self, y0: float, g: float, opts: Optional[dict] = None, verbose: bool = True
    ):
        """
        Solve the non-linear programming (NLP) problem for optimizing Dynamic Movement Primitives (DMPs). 

        This method prepares the necessary parameters and constraints, 
        then invokes the NLP solver to find the optimal solution based on the initial and goal states.

        Args:
            y0 (float): The initial state of the system.
            g (float): The goal state of the system.
            opts (dict, optional): 
                Additional options for the NLP solver. Defaults to an empty dictionary.
            verbose (bool, optional): 
                If True, prints the NLP properties before solving. Defaults to True.

        Raises:
            RuntimeError: If the NLP problem has not been defined prior to calling this method.

        Returns:
            dict: The solution returned by the NLP solver, containing the optimized variables.

        Examples:
            >>> solution = self.solve(y0=0.0, g=1.0)
        """
        opts = opts or {}
        if self.nlp_prob is None:
            raise RuntimeError(
                "The NLP problem has not been written yet. "
                "Call write_nlp_problem() first"
            )
        p, x0, lbx, ubx, lbg, ubg, nlp_props = self._nlp_parameters(y0, g)
        if verbose:
            self._print_nlp_properties(nlp_props)

        return self.nlp_prob(
            p=p,
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dem = Demonstration().set_cosine()
    dmp = Dmp(48.0, 12.0, 2.0, 15)
    dmp.learn_weights(dem)

    opti_prob = ScalarDmpOptimProblem(dmp)
    opti_prob.write_nlp_problem(200)
    dmp.y0 = 1.0
    dmp.g = -3.0
    sol = opti_prob.solve(dmp.y0, dmp.g)

    tau = sol['x'][0]
    w = sol['x'].toarray()[1:16].flatten()

    dmp.w = w
    dmp.tau = float(tau)
    print("Optimal tau:", dmp.tau)
    opt = dmp.integrate(0.01, dmp.tau)
    opt.plot()
    plt.show()
