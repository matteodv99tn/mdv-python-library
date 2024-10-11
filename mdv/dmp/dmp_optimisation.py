import sys
import math
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

        self.w: np.ndarray = dmp.w
        self.nlp_prob = None

    def construct_forcing_function(self) -> ca.Function:
        """
        Construct the forcing function of the DMP

        The forcing function is defined as:

        f(x) = (sum_{j=1}^{nb} w_j * psi_j(x)) / (sum_{j=1}^{nb} psi_j(x))
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
        options: dict = {}
    ):
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
        nlp_props = {
            "tau0": options.get("tau0", self.tau or 1.0),
            "tau_min": options.get("tau_min", self.tau or 0.1),
            "w0": options.get("w0", wguess or self.w),
            "wmin": options.get("wmin", wmin or self.nb * [-ca.inf]),
            "wmax": options.get("wmax", wmax or self.nb * [ca.inf]),
            "vmax": options.get("vmax", 1.0),
            "vf_max": options.get("vf_max", 0.1),
            "ef_max": options.get("ef_max", 0.1)
        }
        return nlp_props

    def _print_nlp_properties(self, d: dict):
        from tabulate import tabulate
        from ..concepts import is_floating

        entries = list()

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
        nlpsol_opts: dict = {},
    ):
        # N: number of discrete time steps

        self.nlp_g = list()  # constraints
        self.nlp_params = list()  # parameters

        y0 = ca.MX.sym('y0')
        g = ca.MX.sym('g')
        self.nlp_params += [y0, g]

        tau = ca.MX.sym('tau')
        w = ca.MX.sym('w', self.nb)
        Zs = ca.MX.sym('zs', N + 1)
        Ys = ca.MX.sym('ys', N + 1)
        self.nlp_x = ca.vertcat(tau, w, Zs, Ys)

        psi = ca.MX.zeros(self.nb)

        zcurr = ca.MX.sym('zcurr')
        ycurr = ca.MX.sym('ycurr')
        gg = ca.MX.sym('gg')
        ff = ca.MX.sym('forcing_term')
        z_dot = self.alpha * (self.beta * (gg-ycurr) - zcurr) + ff
        y_dot = zcurr
        z_ode = ca.Function('z_ode', [zcurr, ycurr, ff, gg], [z_dot])
        y_ode = ca.Function('y_ode', [zcurr], [y_dot])

        self.nlp_g += [Zs[0], Ys[0] - y0]
        dt = float(1 / N)

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
            self.nlp_g += [Zs[i + 1] / tau, Ys[i + 1] - yk_next, Zs[i + 1] - zk_next]

        self.nlp_g += [Zs[N] / tau, Ys[N] - g]

        prob_config = {
            'f': tau,
            'x': self.nlp_x,
            'g': ca.vertcat(*self.nlp_g),
            'p': ca.vertcat(*self.nlp_params)
        }
        self.nlp_prob = ca.nlpsol('solver', 'ipopt', prob_config, nlpsol_opts)
        self.nk = N
        return self.nlp_prob

    def _nlp_parameters(
        self,
        y0: float,
        g: float,
        wmin: Optional[np.ndarray] = None,
        wmax: Optional[np.ndarray] = None,
        wguess: Optional[np.ndarray] = None,
        options: dict = {}
    ):
        if self.nlp_prob is None:
            raise RuntimeError(
                "The NLP problem has not been written yet. "
                "Call write_nlp_problem() first"
            )

        self._prepare_nlp_props(wmin, wmax, wguess, options)

        # Set parameters
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

        x0 = [nlp_props["tau0"], *nlp_props["w0"]]
        lbx = [nlp_props["tau_min"], *nlp_props["wmin"]]
        ubx = [ca.inf, *nlp_props["wmax"]]
        x0 += [0.0, 0.0]
        lbx += [-ca.inf, -ca.inf]
        ubx += [ca.inf, ca.inf]

        lbg = [0.0, 0.0]
        ubg = [0.0, 0.0]

        nk = self.nk
        for _ in range(nk):
            x0 += [0.0, 0.0]
            lbx += [-ca.inf, -ca.inf]
            ubx += [ca.inf, ca.inf]
            lbg += [-nlp_props["vmax"], 0.0, 0.0]
            ubg += [nlp_props["vmax"], 0.0, 0.0]

        lbg += [-nlp_props["vf_max"], -nlp_props["ef_max"]]
        ubg += [nlp_props["vf_max"], nlp_props["ef_max"]]

        return params, x0, lbx, ubx, lbg, ubg, nlp_props

    def solve(self, y0: float, g: float, opts: dict = {}, verbose: bool = True):
        if self.nlp_prob is None:
            raise RuntimeError(
                "The NLP problem has not been written yet. "
                "Call write_nlp_problem() first"
            )
        p, x0, lbx, ubx, lbg, ubg, nlp_props = self._nlp_parameters(y0, g)
        if verbose:
            self._print_nlp_properties(nlp_props)
        sol = self.nlp_prob(
            p=p,
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
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
