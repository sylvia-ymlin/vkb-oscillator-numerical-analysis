import time

import numpy as np
from scipy.integrate import solve_ivp

try:
    from gillespy2 import Model, Species, Reaction, Parameter
except ImportError:
    Model = object
    Species = Reaction = Parameter = None

from config import DEFAULT_PARAMS, BIOLOGICAL_INITIAL_CONDITION, ODE_TOLERANCES, SPECIES
from vkb_model import vkb_ode, vkb_jac


def _build_simulation_inputs(s_R, y0, rtol, atol, default_y0):
    params = {**DEFAULT_PARAMS, "s_R": s_R}
    rtol = rtol if rtol is not None else ODE_TOLERANCES["rtol"]
    atol = atol if atol is not None else ODE_TOLERANCES["atol"]
    y0 = y0 if y0 is not None else default_y0
    return params, y0, rtol, atol


def run_ode_simulation(
    s_R=0.2,
    t_max=400,
    y0=None,
    method='BDF',
    rtol=None,
    atol=None,
    num_points=1000,
    use_jacobian=True,
    return_solver_stats=False,
):
    params, y0, rtol, atol = _build_simulation_inputs(
        s_R,
        y0,
        rtol,
        atol,
        BIOLOGICAL_INITIAL_CONDITION,
    )

    t_eval = np.linspace(0, t_max, int(num_points)) if num_points else None
    kwargs = {"method": method, "t_eval": t_eval, "rtol": rtol, "atol": atol}
    if use_jacobian and method in {"BDF", "Radau"}:
        kwargs["jac"] = lambda t, y: vkb_jac(t, y, params)

    t0  = time.perf_counter()
    sol = solve_ivp(lambda t, y: vkb_ode(t, y, params), (0, t_max), y0, **kwargs)
    elapsed = time.perf_counter() - t0

    result = {'time': sol.t, **{name: sol.y[i] for i, name in enumerate(SPECIES)}}

    if return_solver_stats:
        result["solver_stats"] = {
            "success": bool(sol.success),
            "nfev": int(getattr(sol, "nfev", -1)),
            "njev": int(getattr(sol, "njev", -1)),
            "nlu":  int(getattr(sol, "nlu",  -1)),
            "method": method,
            "elapsed_sec": elapsed,
        }
    return result


class VilarOscillator(Model):
    """GillesPy2 SSA model — one-to-one with vkb_ode propensities."""

    def __init__(self, s_R, t_max):
        if Species is None:
            raise ImportError("gillespy2 is required to run SSA simulations.")
        super().__init__(name="Vilar_Oscillator")
        self.volume = 1

        params = {**DEFAULT_PARAMS, "s_R": s_R}
        for k, v in params.items():
            self.add_parameter(Parameter(name=k, expression=v))

        for name in SPECIES:
            init = 1 if name in ["D_A", "D_R"] else 0
            self.add_species(Species(name=name, initial_value=init, mode="discrete"))

        reactions = [
            ("r1",  {'A':1, 'R':1},    {'C':1},              "r_C"),
            ("r2",  {'A':1},            {},                   "s_A"),
            ("r3",  {'C':1},            {'R':1},              "s_A"),
            ("r4",  {'R':1},            {},                   "s_R"),
            ("r5",  {'D_A':1, 'A':1},  {'D_A_':1},           "r_A"),
            ("r6",  {'D_R':1, 'A':1},  {'D_R_':1},           "r_R"),
            ("r7",  {'D_A_':1},        {'A':1, 'D_A':1},     "theta_A"),
            ("r8",  {'D_A':1},         {'D_A':1,  'MA':1},   "a_A"),
            ("r9",  {'D_A_':1},        {'D_A_':1, 'MA':1},   "a_A_"),
            ("r10", {'MA':1},           {},                   "s_MA"),
            ("r11", {'MA':1},           {'A':1, 'MA':1},     "b_A"),
            ("r12", {'D_R_':1},        {'A':1, 'D_R':1},     "theta_R"),
            ("r13", {'D_R':1},         {'D_R':1,  'MR':1},   "a_R"),
            ("r14", {'D_R_':1},        {'D_R_':1, 'MR':1},   "a_R_"),
            ("r15", {'MR':1},           {},                   "s_MR"),
            ("r16", {'MR':1},           {'MR':1, 'R':1},     "b_R"),
        ]
        for name, reac, prod, rate in reactions:
            self.add_reaction(Reaction(name=name, reactants=reac, products=prod, rate=rate))

        self.timespan(list(range(int(t_max) + 1)))


def run_ssa(s_R, trajectories, t_max, seed=None):
    model = VilarOscillator(s_R=s_R, t_max=t_max)
    try:
        from gillespy2.solvers.cpp.ssa_c_solver import SSACSolver
        return model.run(solver=SSACSolver, number_of_trajectories=trajectories, seed=seed)
    except Exception:
        from gillespy2.solvers.numpy.ssa_solver import NumPySSASolver
        return model.run(solver=NumPySSASolver, number_of_trajectories=trajectories, seed=seed)
