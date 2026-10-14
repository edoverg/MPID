'''
Utility script to test the gradients functionality of the unit cell class of mpid
Given the simulated transmittance phase values and associated tune parameter, 
the unit cell class should return the gradient of the simulated phase with respect to the tuning parameter.
the class also provides a method to compute the gradient at a specific tuning parameter value, through interpolation.
'''

import numpy as np
from matplotlib import pyplot as plt
from mpid import UnitCell

wavelength_um_1_53 = 1.53
wavelength_um_1_55 = 1.55
wavelength_um_1_57 = 1.57
unit_cell_diameters = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_diameters.npy")
unit_cell_phases_wvl_1530 = np.load("unit_cell/unit_cell_center_1530_h900nm_res50_wvl_mode_phase.npy")
unit_cell_phases_wvl_1550 = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_mode_phase.npy")
unit_cell_phases_wvl_1570 = np.load("unit_cell/unit_cell_center_1570_h900nm_res50_wvl_mode_phase.npy")

unit_cell_1530 = UnitCell(wavelength=wavelength_um_1_53, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1530)
unit_cell_1550 = UnitCell(wavelength=wavelength_um_1_55, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1550)
unit_cell_1570 = UnitCell(wavelength=wavelength_um_1_57, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1570)

fig, ax = plt.subplots(figsize=(6, 6))
plt.rcParams['mathtext.default'] = 'regular'
ax.plot(unit_cell_1530.get_tune_param(), unit_cell_1530.get_gradient(), color='r', label=r"$\lambda=1530\:[nm]$", linewidth=3)
ax.plot(unit_cell_1550.get_tune_param(), unit_cell_1550.get_gradient(), color='g', label=r"$\lambda=1550\:[nm]$", linewidth=3)
ax.plot(unit_cell_1570.get_tune_param(), unit_cell_1570.get_gradient(), color='b', label=r"$\lambda=1570\:[nm]$", linewidth=3)
ax.set_xlabel("diameter [um]", fontsize=20)
ax.set_ylabel(r"$\partial \phi/\partial d$", fontsize=20, rotation='vertical')
ax.yaxis.set_label_coords(-0.11, 0.5)
ax.set_aspect('auto')
ax.set_xticks(np.arange(min(unit_cell_diameters), max(unit_cell_diameters)+0.1, 0.1))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.tight_layout()
plt.savefig("unit_cell_gradient.pdf")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(unit_cell_1530.get_tune_param(), unit_cell_1530.get_sim_phase(), color='r', label=r"$\lambda=1530\:[nm]$", linewidth=3)
ax.plot(unit_cell_1550.get_tune_param(), unit_cell_1550.get_sim_phase(), color='g', label=r"$\lambda=1550\:[nm]$", linewidth=3)
ax.plot(unit_cell_1570.get_tune_param(), unit_cell_1570.get_sim_phase(), color='b', label=r"$\lambda=1570\:[nm]$", linewidth=3)
ax.set_xlabel("diameter [um]", fontsize=20)
ax.set_ylabel(r"$\phi\:$[rad]", fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20, loc='best')
ax.set_aspect('auto')
ax.set_xticks(np.arange(min(unit_cell_diameters), max(unit_cell_diameters)+0.1, 0.1))
plt.tight_layout()
plt.savefig("unit_cell_phase.pdf")