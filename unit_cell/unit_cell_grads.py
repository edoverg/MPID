'''
Utility script to test the gradients functionality of the unit cell class of mpid
Given the simulated transmittance phase values and associated tune parameter, 
the unit cell class should return the gradient of the simulated phase with respect to the tuning parameter.
the class also provides a method to compute the gradient at a specific tuning parameter value, through interpolation.
'''

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.default'] = 'regular'

from mpid import UnitCell

wavelength_um_1_50 = 1.50
wavelength_um_1_55 = 1.55
wavelength_um_1_60 = 1.60

#the diameters are the same across all wavelengths
unit_cell_diameters = np.load("unit_cell/final/export/unit_cell_diameter_P_0_65_h_0_9_1_5.npy")

unit_cell_phases_wvl_1500 = np.load("unit_cell/final/export/unit_cell_phase_P_0_65_h_0_9_1_5.npy")
unit_cell_magnitude_wvl_1500 = np.load("unit_cell/final/export/unit_cell_magnitude_P_0_65_h_0_9_1_5.npy")
unit_cell_phases_wvl_1530 = np.load("unit_cell/final/export/unit_cell_phase_P_0_65_h_0_9_1_53.npy")
unit_cell_magnitude_wvl_1530 = np.load("unit_cell/final/export/unit_cell_magnitude_P_0_65_h_0_9_1_53.npy")
unit_cell_phases_wvl_1550 = np.load("unit_cell/final/export/unit_cell_phase_P_0_65_h_0_9_1_55.npy")
unit_cell_magnitude_wvl_1550 = np.load("unit_cell/final/export/unit_cell_magnitude_P_0_65_h_0_9_1_55.npy")
unit_cell_phases_wvl_1570 = np.load("unit_cell/final/export/unit_cell_phase_P_0_65_h_0_9_1_57.npy")
unit_cell_magnitude_wvl_1570 = np.load("unit_cell/final/export/unit_cell_magnitude_P_0_65_h_0_9_1_57.npy")
unit_cell_phases_wvl_1600 = np.load("unit_cell/final/export/unit_cell_phase_P_0_65_h_0_9_1_6.npy")
unit_cell_magnitude_wvl_1600 = np.load("unit_cell/final/export/unit_cell_magnitude_P_0_65_h_0_9_1_6.npy")

unit_cell_1500 = UnitCell(wavelength=wavelength_um_1_50, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1500, sim_magnitude=unit_cell_magnitude_wvl_1500)
unit_cell_1530 = UnitCell(wavelength=1.53, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1530, sim_magnitude=unit_cell_magnitude_wvl_1530)
unit_cell_1550 = UnitCell(wavelength=wavelength_um_1_55, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1550, sim_magnitude=unit_cell_magnitude_wvl_1550)
unit_cell_1570 = UnitCell(wavelength=1.57, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1570, sim_magnitude=unit_cell_magnitude_wvl_1570)
unit_cell_1600 = UnitCell(wavelength=wavelength_um_1_60, tune_param=unit_cell_diameters, sim_phase=unit_cell_phases_wvl_1600, sim_magnitude=unit_cell_magnitude_wvl_1600)


colors = ['purple', 'blue', 'cyan', 'green', 'red']

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(unit_cell_1500.get_tune_param()*1e3, unit_cell_1500.get_gradient(), color=colors[0], label=r"$\lambda=1.50\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1530.get_tune_param()*1e3, unit_cell_1530.get_gradient(), color=colors[1], label=r"$\lambda=1.53\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1550.get_tune_param()*1e3, unit_cell_1550.get_gradient(), color=colors[2], label=r"$\lambda=1.55\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1570.get_tune_param()*1e3, unit_cell_1570.get_gradient(), color=colors[3], label=r"$\lambda=1.57\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1600.get_tune_param()*1e3, unit_cell_1600.get_gradient(), color=colors[4], label=r"$\lambda=1.60\:[\mu m]$", linewidth=3)
ax.set_xlabel("Diameter [nm]", fontsize=20)
ax.set_ylabel(r"$\partial \phi/\partial d$", fontsize=20, rotation='vertical')
ax.yaxis.set_label_coords(-0.11, 0.5)
ax.set_aspect('auto')
ax.set_xticks(np.arange(np.round(min(unit_cell_diameters)*1e3, 0), np.round(max(unit_cell_diameters)*1e3+100, 0), 100))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
#set the frame line thickness
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.tight_layout()
plt.savefig("unit_cell/final/unit_cell_gradient.pdf")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(unit_cell_1500.get_tune_param()*1e3, unit_cell_1500.get_sim_phase(), color=colors[0], label=r"$\lambda=1.50\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1530.get_tune_param()*1e3, unit_cell_1530.get_sim_phase(), color=colors[1], label=r"$\lambda=1.53\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1550.get_tune_param()*1e3, unit_cell_1550.get_sim_phase(), color=colors[2], label=r"$\lambda=1.55\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1570.get_tune_param()*1e3, unit_cell_1570.get_sim_phase(), color=colors[3], label=r"$\lambda=1.57\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1600.get_tune_param()*1e3, unit_cell_1600.get_sim_phase(), color=colors[4], label=r"$\lambda=1.60\:[\mu m]$", linewidth=3)
ax.set_xlabel("Diameter [nm]", fontsize=20)
ax.set_ylabel(r"$\phi\:$[rad]", fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
#ax.legend(fontsize=20, loc='best')
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.set_aspect('auto')
ax.set_xticks(np.arange(np.round(min(unit_cell_diameters)*1e3, 0), np.round(max(unit_cell_diameters)*1e3+100, 0), 100))
plt.tight_layout()
plt.savefig("unit_cell/final/unit_cell_phase.pdf")


fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(unit_cell_1500.get_tune_param()*1e3, unit_cell_1500.get_magnitude(), color=colors[0], label=r"$\lambda=1.50\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1530.get_tune_param()*1e3, unit_cell_1530.get_magnitude(), color=colors[1], label=r"$\lambda=1.53\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1550.get_tune_param()*1e3, unit_cell_1550.get_magnitude(), color=colors[2], label=r"$\lambda=1.55\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1570.get_tune_param()*1e3, unit_cell_1570.get_magnitude(), color=colors[3], label=r"$\lambda=1.57\:[\mu m]$", linewidth=3)
ax.plot(unit_cell_1600.get_tune_param()*1e3, unit_cell_1600.get_magnitude(), color=colors[4], label=r"$\lambda=1.60\:[\mu m]$", linewidth=3)
ax.set_xlabel("Diameter [nm]", fontsize=20)
ax.set_ylim(0,1.1)
ax.set_ylabel(r"Amplitude", fontsize=20)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
#ax.legend(fontsize=20, loc='best')
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.set_aspect('auto')
ax.set_xticks(np.arange(np.round(min(unit_cell_diameters)*1e3, 0), np.round(max(unit_cell_diameters)*1e3+100, 0), 100))
plt.tight_layout()
plt.savefig("unit_cell/final/unit_cell_magnitude.pdf")


#all in one plot with 1 row and 3 cols with magnitude, phase and gradient
fontsize_val = 30
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].plot(unit_cell_1500.get_tune_param()*1e3, unit_cell_1500.get_magnitude(), color=colors[0], label=r"$\lambda=1.50\:[\mu m]$", linewidth=3)
ax[0].plot(unit_cell_1530.get_tune_param()*1e3, unit_cell_1530.get_magnitude(), color=colors[1], label=r"$\lambda=1.53\:[\mu m]$", linewidth=3)
ax[0].plot(unit_cell_1550.get_tune_param()*1e3, unit_cell_1550.get_magnitude(), color=colors[2], label=r"$\lambda=1.55\:[\mu m]$", linewidth=3)
ax[0].plot(unit_cell_1570.get_tune_param()*1e3, unit_cell_1570.get_magnitude(), color=colors[3], label=r"$\lambda=1.57\:[\mu m]$", linewidth=3)
ax[0].plot(unit_cell_1600.get_tune_param()*1e3, unit_cell_1600.get_magnitude(), color=colors[4], label=r"$\lambda=1.60\:[\mu m]$", linewidth=3)
ax[0].set_xlabel("Diameter [nm]", fontsize=fontsize_val)
ax[0].set_ylabel("Amplitude", fontsize=fontsize_val)
ax[0].tick_params(axis='x', labelsize=fontsize_val)
ax[0].tick_params(axis='y', labelsize=fontsize_val)
ax[0].set_ylim(0, 1.1)
for spine in ax[0].spines.values():
    spine.set_linewidth(2)
ax[0].set_aspect('auto')
ax[0].set_xticks(np.arange(np.round(min(unit_cell_diameters)*1e3, 0), np.round(max(unit_cell_diameters)*1e3+100, 0), 100))
#ax[0].legend(fontsize=20, loc='best')
ax[1].plot(unit_cell_1500.get_tune_param()*1e3, unit_cell_1500.get_phase_at_param(unit_cell_1500.get_tune_param()), color=colors[0], label=r"$\lambda=1.50\:[\mu m]$", linewidth=3)
ax[1].plot(unit_cell_1530.get_tune_param()*1e3, unit_cell_1530.get_phase_at_param(unit_cell_1530.get_tune_param()), color=colors[1], label=r"$\lambda=1.53\:[\mu m]$", linewidth=3)
ax[1].plot(unit_cell_1550.get_tune_param()*1e3, unit_cell_1550.get_phase_at_param(unit_cell_1550.get_tune_param()), color=colors[2], label=r"$\lambda=1.55\:[\mu m]$", linewidth=3)
ax[1].plot(unit_cell_1570.get_tune_param()*1e3, unit_cell_1570.get_phase_at_param(unit_cell_1570.get_tune_param()), color=colors[3], label=r"$\lambda=1.57\:[\mu m]$", linewidth=3)
ax[1].plot(unit_cell_1600.get_tune_param()*1e3, unit_cell_1600.get_phase_at_param(unit_cell_1600.get_tune_param()), color=colors[4], label=r"$\lambda=1.60\:[\mu m]$", linewidth=3)
ax[1].set_xlabel("Diameter [nm]", fontsize=fontsize_val)
ax[1].set_ylabel("Phase [rad]", fontsize=fontsize_val)
ax[1].tick_params(axis='x', labelsize=fontsize_val)
ax[1].tick_params(axis='y', labelsize=fontsize_val)
for spine in ax[1].spines.values():
    spine.set_linewidth(2)
ax[1].set_aspect('auto')
ax[1].set_xticks(np.arange(np.round(min(unit_cell_diameters)*1e3, 0), np.round(max(unit_cell_diameters)*1e3+100, 0), 100))
ax[2].plot(unit_cell_1500.get_tune_param()*1e3, unit_cell_1500.get_gradient(), color=colors[0], label=r"$\lambda=1.50\:[\mu m]$", linewidth=3)
ax[2].plot(unit_cell_1530.get_tune_param()*1e3, unit_cell_1530.get_gradient(), color=colors[1], label=r"$\lambda=1.53\:[\mu m]$", linewidth=3)
ax[2].plot(unit_cell_1550.get_tune_param()*1e3, unit_cell_1550.get_gradient(), color=colors[2], label=r"$\lambda=1.55\:[\mu m]$", linewidth=3)
ax[2].plot(unit_cell_1570.get_tune_param()*1e3, unit_cell_1570.get_gradient(), color=colors[3], label=r"$\lambda=1.57\:[\mu m]$", linewidth=3)
ax[2].plot(unit_cell_1600.get_tune_param()*1e3, unit_cell_1600.get_gradient(), color=colors[4], label=r"$\lambda=1.60\:[\mu m]$", linewidth=3)
ax[2].set_xlabel("Diameter [nm]", fontsize=fontsize_val)
ax[2].set_ylabel(r"$\partial \phi/\partial d$", fontsize=fontsize_val, rotation='vertical')
ax[2].tick_params(axis='x', labelsize=fontsize_val)
ax[2].tick_params(axis='y', labelsize=fontsize_val)
for spine in ax[2].spines.values():
    spine.set_linewidth(2)
ax[2].set_aspect('auto')
ax[2].set_xticks(np.arange(np.round(min(unit_cell_diameters)*1e3, 0), np.round(max(unit_cell_diameters)*1e3+100, 0), 100))
#ax[2].legend(fontsize=20, loc='best')
plt.tight_layout()
plt.savefig("unit_cell/final/unit_cell_complete.pdf")