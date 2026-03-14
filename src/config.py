RC_PARAMS = {
    # 字体系统 - 统一使用无衬线字体
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,           # 基础字体大小
    'axes.titlesize': 11,      # 子图标题
    'axes.labelsize': 11,      # 坐标轴标签
    'legend.fontsize': 10,     # 图例
    'xtick.labelsize': 10,     # x轴刻度
    'ytick.labelsize': 10,     # y轴刻度
    'figure.titlesize': 13,    # 主标题
    
    # 字体粗细 - 全部使用常规体，避免加粗
    'axes.titleweight': 'normal',
    'figure.titleweight': 'normal',
    
    # 坐标轴样式 - 统一无上右边框
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.0,
    
    # 网格线 - 统一淡灰色虚线
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.color': '#EEEEEE',
    
    # 背景色
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    
    # 图例样式 - 无边框
    'legend.frameon': False,
    'legend.loc': 'best',
    
    # 线条样式
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    
    # 布局
    'figure.constrained_layout.use': False,  # 禁用以避免与colorbar冲突
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

# --- Unified ODE Solver Tolerances ---
# Critical for consistency across all analyses, especially near bifurcation points
ODE_TOLERANCES = {
    'rtol': 1e-8,
    'atol': 1e-10,
}

# Looser tolerances for quick exploratory runs (not used in final analysis)
ODE_TOLERANCES_LOOSE = {
    'rtol': 1e-6,
    'atol': 1e-8,
}

SPECIES = ['D_A', 'D_R', 'D_A_', 'D_R_', 'MA', 'A', 'MR', 'R', 'C']

# Biological initial condition: both promoters unbound (D_A=1, D_R=1),
# all mRNA and protein concentrations at zero.
BIOLOGICAL_INITIAL_CONDITION = [1, 1, 0, 0, 0, 0, 0, 0, 0]

# --- Simulation Scenarios ---

BASELINE_CASES = (
    {"label": "baseline", "s_R": 0.2},
    {"label": "low_sR", "s_R": 0.03},
)

REDUCTION_CASES = (
    {"label": "oscillatory", "s_R": 0.2},
    {"label": "low_sR", "s_R": 0.03},
    {"label": "near_threshold", "s_R": 0.088},
)

NUMERICS_BENCHMARK_METHODS = ("RK45", "BDF", "Radau")
NUMERICS_IMPLICIT_METHODS = ("BDF", "Radau")
NUMERICS_TOLERANCE_GRID = (
    (1e-5, 1e-7),
    (1e-6, 1e-8),
    (1e-7, 1e-9),
)
NUMERICS_SENSITIVITY_CASES = (
    {"label": "oscillatory", "s_R": 0.2},
    {"label": "near_threshold", "s_R": 0.088},
)

# --- Unified Plotting Configuration ---
# 严格的设计系统，确保所有图表视觉一致性

PLOT_CONFIG = {
    'FIGURES_DIR': 'figures',
    'IMG_DIR': 'figures',
    'DPI': 150,
    
    # 统一配色方案 - 基于你的建议
    'PRIMARY_PALETTE': {
        # 核心对比色
        'full_model': '#333333',    # 深灰 - Full model
        'reduced_model': '#1F77B4', # 亮蓝 - Reduced model
        'repressor': '#9467BD',     # 紫色 - Repressor (R)
        'activator': '#E17055',     # 珊瑚红 - Activator (A)
        
        # 状态颜色
        'valid': '#2ca02c',         # 绿色 - Valid/Stable
        'invalid': '#d62728',       # 红色 - Invalid/Unstable
        'warning': '#ff7f0e',       # 橙色 - Warning
        'neutral': '#7f7f7f',       # 灰色 - Neutral
        
        # 辅助颜色
        'blue': '#1f77b4',
        'orange': '#ff7f0e',
        'green': '#2ca02c',
        'red': '#d62728',
        'purple': '#9467bd',
        'brown': '#8c564b',
        'pink': '#e377c2',
        'gray': '#7f7f7f',
        'olive': '#bcbd22',
        'cyan': '#17becf',
    },
    
    # Species-specific colors - 使用统一的紫色和黄绿色
    'COLORS': {
        'A': ['#E17055', '#F4A492', '#FAD5CC'],  # 珊瑚红系 - Activator
        'R': ['#9467BD', '#B08FD9', '#CCB7E5'],  # 紫色系 - Repressor
    },
    
    # Model comparison colors - 深灰 vs 亮蓝
    'MODEL_COLORS': {
        'full': '#333333',      # 深灰 for full model
        'reduced': '#1F77B4',   # 亮蓝 for reduced model
        'reference': '#333333',  # 深灰 for reference/baseline
    },
    
    # Status/validity colors
    'STATUS_COLORS': {
        'valid': '#2ca02c',      # 绿色
        'invalid': '#d62728',    # 红色
        'warning': '#ff7f0e',    # 橙色
        'neutral': '#7f7f7f',    # 灰色
    },
    
    # Attractor/fate colors - 使用模型颜色保持一致性
    'FATE_COLORS': {
        'equilibrium': '#1F77B4',  # 亮蓝（与reduced model一致）
        'limit_cycle': '#ff7f0e',  # 橙色
        'failed': '#7f7f7f',       # 灰色
    },
    
    # Bifurcation colors
    'BIFURCATION_COLORS': {
        'full_threshold': '#333333',    # 深灰（与full model一致）
        'reduced_threshold': '#1F77B4', # 亮蓝（与reduced model一致）
        'test_steady': '#1F77B4',       # 蓝色（→ 平衡态）
        'test_bistable': '#4d4d4d',     # 深灰（→ 双稳，中性）
        'test_oscillatory': '#ff7f0e',  # 橙色（→ 极限环）
    },
    
    # Neutral/utility colors
    'NEUTRAL_COLORS': {
        'black': '#000000',
        'white': '#FFFFFF',
        'gray_dark': '#4d4d4d',
        'gray': '#7f7f7f',
        'gray_light': '#CCCCCC',
        'grid': '#EEEEEE',
        'gold': '#FFD700',  # 添加gold颜色
    },
    
    # Eigenvalue visualization colors - 使用Set1调色板
    'EIGENVALUE_COLORS': {
        'dominant_1': '#d62728',  # 红色
        'dominant_2': '#1f77b4',  # 蓝色
        'background': '#7f7f7f',  # 灰色
    },
    
    # 线型规范
    'LINE_STYLES': {
        'full_solid': '-',          # Full model: 实线
        'reduced_solid': '-',       # Reduced model: 实线
        'reduced_dashed': '--',     # Reduced model alternative: 虚线
        'reference': ':',           # Reference: 点线
    },
    
    # 标记样式
    'MARKERS': {
        'full': 'o',               # Full model: 圆点
        'reduced': 's',            # Reduced model: 方块
        'data': 'o',               # Data points: 圆点
    },
}
