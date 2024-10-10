import sys
import logging
import numpy as np
import casadi as ca

from typing import Optional
from .. import default_logger_formatter
from . import Dmp, Demonstration


class DmpOptimisationProblemBase:
    # As a rule of thumb
    #  - nk number of time steps for numerical integration
    #  - np number of dimensions of the position state
    #  - nb number of basis functions

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

    def construct_forcing_function(self) -> ca.Function:
        """
        Construct the forcing function of the DMP

        The forcing function is defined as:

        f(x) = (sum_{j=1}^{nb} w_j * psi_j(x)) / (sum_{j=1}^{nb} psi_j(x))
        """
        s = ca.SX.sym('s')

        psi = ca.SX.zeros(self.nb)
        for j in range(self.nb):
            psi[j] = ca.exp(-self.h[j] * (s - self.c[j])**2)

        w = ca.SX.sym('w', self.nb)
        f = ca.dot(w, psi) / ca.sum1(psi)
        return ca.Function('forcing_function', [s, w], [f])


class ScalarDmpOptimProblem(DmpOptimisationProblemBase):

    def __init__(self, dmp: Dmp, w: Optional[np.ndarray] = None):
        if not dmp.is_scalar():
            raise ValueError("The DMP must be scalar for this optimisation problem")
        super().__init__(dmp)
        self.w: np.ndarray = w or dmp.w

    def write_nlp_problem(
        self,
        N: int,
        vmax: float = 1.0,
        vdelta: float = 0.3,
        gdelta: float = 0.2,
        wmin: Optional[np.ndarray] = None,
        wmax: Optional[np.ndarray] = None
    ):
        # N: number of discrete time steps

        self.nlp_x = list()  # optimisation variables
        self.nlp_x0 = list()  # initial guess
        self.nlp_lbx = list()  # lower bounds
        self.nlp_ubx = list()  # upper bounds
        self.nlp_g = list()  # constraints
        self.nlp_lbg = list()
        self.nlp_ubg = list()
        self.nlp_params = list()  # parameters

        y0 = ca.SX.sym('y0')
        g = ca.SX.sym('g')
        self.nlp_params += [y0, g]

        tau = ca.SX.sym('tau')
        w = ca.SX.sym('w', self.nb)

        f = self.construct_forcing_function()

        xi = ca.SX.sym('xi')
        s = ca.exp(-self.gamma * xi)
        zcurr = ca.SX.sym('zcurr')
        ycurr = ca.SX.sym('ycurr')
        yy0 = ca.SX.sym('yy0')
        gg = ca.SX.sym('gg')
        z_dot = self.alpha * (self.beta * (gg-ycurr) - zcurr) + f(s, w) * (gg-yy0) * s
        y_dot = zcurr

        z_ode = ca.Function('z_ode', [zcurr, ycurr, xi, w, yy0, gg], [z_dot])
        y_ode = ca.Function('y_ode', [zcurr], [y_dot])

        self.nlp_x += [tau]
        self.nlp_x += [w[i] for i in range(self.nb)]
        self.nlp_x0 += [self.tau, *self.w]
        self.nlp_lbx += [0.1]
        self.nlp_ubx += [ca.inf]
        self.nlp_lbx += wmin if wmin is not None else self.nb * [-ca.inf]
        self.nlp_ubx += wmax if wmax is not None else self.nb * [ca.inf]

        zk = ca.SX.sym('z0')
        yk = ca.SX.sym('y0')
        self.nlp_x += [zk, yk]
        self.nlp_x0 += [0.0, y0]
        self.nlp_lbx += [-ca.inf, -ca.inf]
        self.nlp_ubx += [ca.inf, ca.inf]
        self.nlp_g += [zk, yk - y0]
        self.nlp_lbg += [0.0, 0.0]
        self.nlp_ubg += [0.0, 0.0]

        dt = float(1 / N)
        for i in range(N):
            zk_next = zk + dt * z_ode(zk, yk, i * dt, w, y0, g)
            yk_next = yk + dt * y_ode(zk)
            zk = ca.SX.sym(f'z_{i+1}')
            yk = ca.SX.sym(f'y_{i+1}')
            self.nlp_x += [zk, yk]
            self.nlp_x0 += [0.0, y0]
            self.nlp_lbx += [-ca.inf, -ca.inf]
            self.nlp_ubx += [ca.inf, ca.inf]

            self.nlp_g += [zk / tau, yk - yk_next, zk - zk_next]
            self.nlp_lbg += [-vmax, 0.0, 0.0]
            self.nlp_ubg += [vmax, 0.0, 0.0]

        self.nlp_g += [zk / tau, yk - g]
        self.nlp_lbg += [-vdelta, -gdelta]
        self.nlp_ubg += [vdelta, gdelta]

        prob_config = {
            'f': tau,
            'x': ca.vertcat(*self.nlp_x),
            'g': ca.vertcat(*self.nlp_g),
            'p': ca.vertcat(*self.nlp_params)
        }
        jit_options = {"flags": ["-O3"], "verbose": False, "compiler": "gcc"}
        options = {
            "jit": False,
            "compiler": "shell",
            "jit_options": jit_options,
            "verbose": False
        }
        self.nlp_prob = ca.nlpsol('solver', 'ipopt', prob_config, options)
        return self.nlp_prob

    def solve(self, y0: float, g: float):
        print("Solving the optimisation problem")
        pars = [y0, g]
        sol = self.nlp_prob(
            p=pars,
            x0=self.nlp_x0,
            lbx=self.nlp_lbx,
            ubx=self.nlp_ubx,
            lbg=self.nlp_lbg,
            ubg=self.nlp_ubg,
        )

        return sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dem = Demonstration().set_cosine()
    dmp = Dmp(48.0, 12.0, 2.0, 15)
    dmp.learn_weights(dem)

    # dmp.integrate(0.01, dmp.tau).plot()
    # plt.show()

    opti_prob = ScalarDmpOptimProblem(dmp)
    opti_prob.write_nlp_problem(200)
    dmp.y0 = 1
    dmp.g = -1
    sol = opti_prob.solve(1, -1)

    tau = sol['x'][0]
    w = sol['x'].toarray()[1:16].flatten()

    dmp.w = w
    dmp.tau = float(tau)
    print("Optimal tau:", dmp.tau)
    opt = dmp.integrate(0.01, dmp.tau)
    opt.plot()
    plt.show()
