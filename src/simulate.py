import numpy as np
from scipy.integrate import solve_ivp
import gillespy2
from gillespy2 import Model, Species, Reaction, Parameter
from src.config import DEFAULT_PARAMS, BIOLOGICAL_INITIAL_CONDITION, ODE_TOLERANCES, ODE_TOLERANCES_LOOSE, SPECIES
from src.models import vkb_ode, vkb_jac, reduced_vkb_ode, reduced_vkb_jac, REDUCED_SPECIES


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
    """Run ODE simulation for the VKB oscillator."""
    params = {**DEFAULT_PARAMS, 's_R': s_R}
    
    # Use unified tolerances if not explicitly provided
    if rtol is None:
        rtol = ODE_TOLERANCES['rtol']
    if atol is None:
        atol = ODE_TOLERANCES['atol']
    
    if y0 is None:
        y0 = BIOLOGICAL_INITIAL_CONDITION

    t_span = (0, t_max)
    
    # If num_points is 0 or None, solve_ivp returns the adaptive solver steps
    t_eval = np.linspace(0, t_max, int(num_points)) if num_points else None

    solve_kwargs = {
        "method": method,
        "t_eval": t_eval,
        "rtol": rtol,
        "atol": atol,
    }
    if use_jacobian and method in {"BDF", "Radau"}:
        solve_kwargs["jac"] = lambda t, y: vkb_jac(t, y, params)

    import time
    t0 = time.perf_counter()
    sol = solve_ivp(
        lambda t, y: vkb_ode(t, y, params),
        t_span,
        y0,
        **solve_kwargs,
    )
    elapsed = time.perf_counter() - t0

    # Store time series first, then pack the 2D solution matrix (sol.y) into a dict 
    # using species names as keys, so it can be accessed like result['A']
    result = {'time': sol.t}
    for i, name in enumerate(SPECIES):
        result[name] = sol.y[i]

    if return_solver_stats:
        result["solver_stats"] = {
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": str(sol.message),
            "nfev": int(getattr(sol, "nfev", -1)),
            "njev": int(getattr(sol, "njev", -1)),
            "nlu": int(getattr(sol, "nlu", -1)),
            "method": method,
            "rtol": float(rtol),
            "atol": float(atol),
            "used_jacobian": bool(use_jacobian and method in {"BDF", "Radau"}),
            "elapsed_sec": elapsed,
        }

    return result





class VilarOscillator(Model):
    def __init__(self, s_R, t_max):
        super().__init__(name="Vilar_Oscillator")
        self.volume = 1

        params = {**DEFAULT_PARAMS, 's_R': s_R}
        for k, v in params.items():
            self.add_parameter(Parameter(name=k, expression=v))

        for name in SPECIES:
            init = 1 if name in ['D_A', 'D_R'] else 0
            self.add_species(Species(name=name, initial_value=init, mode='discrete'))

        reac_data = [
            ("r1", {'A':1, 'R':1}, {'C':1}, "r_C"), ("r2", {'A':1}, {}, "s_A"),
            ("r3", {'C':1}, {'R':1}, "s_A"), ("r4", {'R':1}, {}, "s_R"),
            ("r5", {'D_A':1, 'A':1}, {'D_A_':1}, "r_A"), ("r6", {'D_R':1, 'A':1}, {'D_R_':1}, "r_R"),
            ("r7", {'D_A_':1}, {'A':1, 'D_A':1}, "theta_A"), ("r8", {'D_A':1}, {'D_A':1, 'MA':1}, "a_A"),
            ("r9", {'D_A_':1}, {'D_A_':1, 'MA':1}, "a_A_"), ("r10", {'MA':1}, {}, "s_MA"),
            ("r11", {'MA':1}, {'A':1, 'MA':1}, "b_A"), ("r12", {'D_R_':1}, {'A':1, 'D_R':1}, "theta_R"),
            ("r13", {'D_R':1}, {'D_R':1, 'MR':1}, "a_R"), ("r14", {'D_R_':1}, {'D_R_':1, 'MR':1}, "a_R_"),
            ("r15", {'MR':1}, {}, "s_MR"), ("r16", {'MR':1}, {'MR':1, 'R':1}, "b_R")
        ]
        for name, reac, prod, rate in reac_data:
            self.add_reaction(Reaction(name=name, reactants=reac, products=prod, rate=rate))

        self.timespan(list(range(int(t_max) + 1)))

def run_ssa(s_R, trajectories, t_max, seed=None):
    """Run SSA simulation using GillesPy2 native solver handling."""
    model = VilarOscillator(s_R=s_R, t_max=t_max)
    kwargs = {'number_of_trajectories': trajectories}
    if seed is not None:
        kwargs['seed'] = int(seed)
    return model.run(**kwargs)






def run_reduced_simulation(
    s_R=0.2,
    t_max=400,
    y0=None,
    method="BDF",
    rtol=None,
    atol=None,
    num_points=1000,
):
    params = {**DEFAULT_PARAMS, "s_R": s_R}
    
    # Use unified tolerances if not explicitly provided
    if rtol is None:
        rtol = ODE_TOLERANCES['rtol']
    if atol is None:
        atol = ODE_TOLERANCES['atol']

    if y0 is None:
        y0 = [0.0, 0.0]

    t_eval = np.linspace(0, t_max, int(num_points))
    sol = solve_ivp(
        lambda t, y: reduced_vkb_ode(t, y, params),
        (0, t_max),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    result = {"time": sol.t}
    for index, name in enumerate(REDUCED_SPECIES):
        result[name] = sol.y[index]
    result["solver_stats"] = {
        "success": bool(sol.success),
        "status": int(sol.status),
        "message": str(sol.message),
        "method": method,
        "nfev": int(getattr(sol, "nfev", -1)),
        "njev": int(getattr(sol, "njev", -1)),
        "nlu": int(getattr(sol, "nlu", -1)),
    }
    return result
