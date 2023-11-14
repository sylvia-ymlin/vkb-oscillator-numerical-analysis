RC_PARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 13,
    'axes.titleweight': 'normal',
    'figure.titleweight': 'normal',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.color': '#EEEEEE',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'legend.frameon': False,
    'legend.loc': 'best',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.constrained_layout.use': False,
}

DEFAULT_PARAMS = {
    'a_A': 50.0, 'a_R': 0.01,
    'a_A_': 500.0, 'a_R_': 50.0,
    'b_A': 50.0, 'b_R': 5.0,
    's_MA': 10.0, 's_MR': 0.5,
    's_A': 1.0, 's_R': 0.2,
    'r_A': 1.0, 'r_R': 1.0, 'r_C': 2.0,
    'theta_A': 50.0, 'theta_R': 100.0,
}

ODE_TOLERANCES = {'rtol': 1e-8, 'atol': 1e-10}
ODE_TOLERANCES_LOOSE = {'rtol': 1e-6, 'atol': 1e-8}

SPECIES = ['D_A', 'D_R', 'D_A_', 'D_R_', 'MA', 'A', 'MR', 'R', 'C']
BIOLOGICAL_INITIAL_CONDITION = [1, 1, 0, 0, 0, 0, 0, 0, 0]

BASELINE_CASES = (
    {"label": "baseline", "s_R": 0.2},
    {"label": "low_sR",   "s_R": 0.05},
)

PLOT_CONFIG = {
    'FIGURES_DIR': 'figures',
    'IMG_DIR': 'figures',
    'DPI': 150,

    # A=coral, R=purple; three shades from dark to light for multiple trajectories
    'COLORS': {
        'A': ['#E17055', '#F4A492', '#FAD5CC'],
        'R': ['#9467BD', '#B08FD9', '#CCB7E5'],
    },

    'NEUTRAL_COLORS': {
        'black':    '#000000',
        'gray':     '#7f7f7f',
        'gray_dark': '#4d4d4d',
    },

    'FIGSIZE': {
        'single':             (8.0, 4.0),
        'dual_row':           (8.0, 5.0),
        'bifurcation_stack':  (6.8, 5.35),
    },
}
