'''
This script is used to generate plots from the results of the EGO optimization
It loads the surrogate model and the tested points to generate plots of the model
'''

import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
from smt.surrogate_models import KRG
import pickle
# Load the surrogate model and the tested points
surrogate_model_file_location = 'matern32_surrogate_filtered_1_0/refined_surrogate_model.pkl' 
#points are divided into DOE points and EGO points
doe_points_file_location = 'matern32_surrogate_filtered_1_0/ego_evaluations.txt'
ego_points_file_location = 'matern32_surrogate_filtered_1_0/refined_dataset.pkl'

# Load the surrogate model
with open(surrogate_model_file_location, 'rb') as f:
    surrogate_model = pickle.load(f)

# Load the DOE points
#first column is d0, second column is d1, third column is the result
doe_points = np.loadtxt(doe_points_file_location)


# Load the EGO points
#the first element is an array of point location, the second element is
#an array of the corresponding function values. EGO points
#contain also previous DOE points, which we shall remove
with open(ego_points_file_location, 'rb') as f:
    ego_points = pickle.load(f)
ego_points_locations = ego_points[0]
ego_points_values = ego_points[1]
#remove DOE points from EGO points
mask = np.ones(ego_points_locations.shape[0], dtype=bool)
for i, loc in enumerate(ego_points_locations):
    if any(np.all(loc == dp[:2]) for dp in doe_points):
        mask[i] = False
ego_points_locations = ego_points_locations[mask]
ego_points_values = ego_points_values[mask]

#make a contourplot of the surrogate model 
xlimits = np.array([[100,700], [100,700]]) # d0, d1 in [um]
seed = 777

x = np.linspace(xlimits[0,0], xlimits[0,1], 100)
y = np.linspace(xlimits[1,0], xlimits[1,1], 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        #predict values returns an np.atleast_2d, so we need to explode the value
        [[t]] = surrogate_model.predict_values(np.atleast_2d([X[i,j], Y[i,j]]))
        Z[i,j] = t

fig, ax = plt.subplots()
#levels = np.linspace(0, 1, 51)
cp = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
#the colorbar ticks should be rounded to max 2 decimal places

cbar = fig.colorbar(cp, ax=ax, label='Surrogate model',)
cbar.set_ticks(np.round(np.linspace(Z.min(), Z.max(), 5), 2))
#cbar.set_ticks(np.arange(0, 1.01, 0.25))
cbar.ax.tick_params(labelsize=20)
cbar.ax.yaxis.label.set_size(20)
ax.set_xlabel(r'$d_0\: [\mu m]$', fontsize=20)
ax.set_ylabel(r'$d_1\: [\mu m]$', fontsize=20)
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.set_xlim(xlimits[0])
ax.set_ylim(xlimits[1])
#the ticks for the axes should start from 100 and increase by 200
ax.set_xticks(np.arange(100, 701, 200))
ax.set_yticks(np.arange(100, 701, 200))
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(framealpha=0.5, fontsize=20)
fig.tight_layout()
ax.scatter(doe_points[:,0], doe_points[:,1], c='red', label='DOE Points')
ax.scatter(ego_points_locations[:,0], ego_points_locations[:,1], c='blue', label='EGO Points')
ax.legend(loc='upper right', framealpha=0.7, fontsize=20)
plt.savefig('surrogate_model_contour_plot.pdf')



