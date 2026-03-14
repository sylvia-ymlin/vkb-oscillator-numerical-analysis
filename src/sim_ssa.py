from gillespy2 import Model, Species, Reaction, Parameter

from .config import DEFAULT_PARAMS, SPECIES


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

