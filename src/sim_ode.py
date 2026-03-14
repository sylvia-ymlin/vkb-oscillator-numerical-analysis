import numpy as np
from scipy.integrate import solve_ivp
from .ode_system import vkb_ode, vkb_jac
from .config import DEFAULT_PARAMS, SPECIES, ODE_TOLERANCES, BIOLOGICAL_INITIAL_CONDITION


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
    t_eval = np.linspace(0, t_max, int(num_points))

    solve_kwargs = {
        "method": method,
        "t_eval": t_eval,
        "rtol": rtol,
        "atol": atol,
    }
    if use_jacobian and method in {"BDF", "Radau"}:
        solve_kwargs["jac"] = lambda t, y: vkb_jac(t, y, params)

    sol = solve_ivp(
        lambda t, y: vkb_ode(t, y, params),
        t_span,
        y0,
        **solve_kwargs,
    )

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
        }

    return result
