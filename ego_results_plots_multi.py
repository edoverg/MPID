'''
Generates the surrogate model contour plot for several filter radii, using a
common colormap range (global min/max of the surrogate over all radii) so that
the plots are directly comparable.
'''

import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
import pickle
from matplotlib.colors import Normalize

# Filter radii, matching the folders matern32_surrogate_filtered_<radius>_<strategy>
radii = ['0', '1_0', '2_0' , '3_0', '5_0', '7_0']
xlimits = np.array([[100, 700], [100, 700]])  # d0, d1 in [um]
n_grid = 100
strategy = 'epigraph'

def load_results(radius,strategy):
    folder = f'matern32_surrogate_filtered_{radius}_{strategy}'
    with open(f'{folder}/refined_surrogate_model.pkl', 'rb') as f:
        surrogate_model = pickle.load(f)
    # columns: d0, d1, result
    doe_points = np.loadtxt(f'{folder}/ego_evaluations.txt')
    # EGO dataset also contains the DOE points, which are removed here
    with open(f'{folder}/refined_dataset.pkl', 'rb') as f:
        ego_points = pickle.load(f)
    ego_locs = ego_points[0]
    mask = np.array([not any(np.all(loc == dp[:2]) for dp in doe_points)
                     for loc in ego_locs])
    # rows: d0, d1, best value
    best = np.loadtxt(f'{folder}/ego_opt_results.txt')
    return surrogate_model, doe_points, ego_locs[mask], best


x = np.linspace(xlimits[0, 0], xlimits[0, 1], n_grid)
y = np.linspace(xlimits[1, 0], xlimits[1, 1], n_grid)
X, Y = np.meshgrid(x, y)

data = {}
global_min, global_max = np.inf, -np.inf
for radius in radii:
    model, doe, ego, best = load_results(radius, strategy)
    Z = model.predict_values(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    data[radius] = (Z, doe, ego, best)
    global_min = min(global_min, Z.min())
    global_max = max(global_max, Z.max())
    print(f'radius {radius}: min={Z.min():.4f}, max={Z.max():.4f}')
print(f'Global range: [{global_min:.4f}, {global_max:.4f}]')

pad = 1e-6 * (global_max - global_min)
levels = np.linspace(global_min - pad, global_max + pad, 51)
norm = Normalize(vmin=global_min, vmax=global_max)
for radius in radii:
    Z, doe, ego, best = data[radius]
    fig, ax = plt.subplots()
    cp = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', norm=norm)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax,
                        label='Surrogate model')
    cbar.set_ticks(np.round(np.linspace(global_min, global_max, 5), 2))
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.yaxis.label.set_size(20)
    ax.set_xlabel(r'$d_0\: [\mu m]$', fontsize=20)
    ax.set_ylabel(r'$d_1\: [\mu m]$', fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.set_xlim(xlimits[0])
    ax.set_ylim(xlimits[1])
    ax.set_xticks(np.arange(100, 701, 200))
    ax.set_yticks(np.arange(100, 701, 200))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.scatter(doe[:, 0], doe[:, 1], c='red', label='DOE')
    ax.scatter(ego[:, 0], ego[:, 1], c='blue', label='EGO')
    ax.scatter(best[0], best[1], c='limegreen', s=120, edgecolors='k',
               linewidths=0.8, label='Best', zorder=5)
    ax.legend(loc='upper right', framealpha=0.7, fontsize=20)
    fig.tight_layout()
    fig.savefig(f'surrogate_model_contour_plot_{radius}_{strategy}.pdf')
    plt.show()
    plt.close(fig)
