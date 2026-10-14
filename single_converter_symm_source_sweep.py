import mpid as pid
import numpy as np
import matplotlib.pyplot as plt
import os
if not os.path.exists("results"):
    os.makedirs("results")
#Space grid definition
N = 1025
unit_cell_pitch = 0.700 #i.e., 700 [nm]

n0 = 1.444
n1 = 1.0
n2 = 1.444
mat_refr_index = np.array([n0, n1, n2])

ms_max_available_diameter = 122 #[um]
output_fiber_core_diameter = 19 #[um]
opt_max_eval = 140

central_wavelength_um = 1.55 #[um]

wavelengths_to_test_um = [1.530, 1.550, 1.570] #[um]

source_files = ["fiber_modes/SMF/SMF_LP01_field_distribution_lda_1530_N_1025.npy",
                "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy",
                "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1570_N_1025.npy"]

target_files = ["fiber_modes/FMF/FMF_LP21_field_distribution_lda_1530_N_1025.npy",
                "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1550_N_1025.npy",
                "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1570_N_1025.npy"]

def setup_optical_system()->pid.Simulation:
    '''
    Builds the mode converter optical system using 
    the specified propagation distances `d`, runs the optimization
    and returns the best objective value.

    Args:
        xdoe_points (np.ndarray): Array containing the DOE points.
        Each point consists of the propagation distances d0 and d1

    Returns:
        float: The best objective value obtained from the optimization.
    '''
    sources = [pid.Source(input_field=np.load(file), 
                          wavelength=wvl, 
                          field_norm=True) for file, wvl in zip(source_files, wavelengths_to_test_um)]

    targets = [pid.Target(input_field=np.load(file), 
                          associated_source=sources[i], 
                          wavelength=wvl, 
                          field_norm=True) for i, (file, wvl) in enumerate(zip(target_files, wavelengths_to_test_um))]

    #import unit cell data
    unit_cell_diameters_wvl_1530 = np.load("unit_cell/unit_cell_center_1530_h900nm_res50_wvl_diameters.npy")
    unit_cell_diameters_wvl_1550 = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_diameters.npy")
    unit_cell_diameters_wvl_1570 = np.load("unit_cell/unit_cell_center_1570_h900nm_res50_wvl_diameters.npy")
    unit_cell_phases_wvl_1530 = np.load("unit_cell/unit_cell_center_1530_h900nm_res50_wvl_mode_phase.npy")
    unit_cell_phases_wvl_1550 = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_mode_phase.npy")
    unit_cell_phases_wvl_1570 = np.load("unit_cell/unit_cell_center_1570_h900nm_res50_wvl_mode_phase.npy")
    #construct the UnitCell objects
    unit_cell_1530 = pid.UnitCell(wavelength=1.530, tune_param=unit_cell_diameters_wvl_1530, sim_phase=unit_cell_phases_wvl_1530)
    unit_cell_1550 = pid.UnitCell(wavelength=1.550, tune_param=unit_cell_diameters_wvl_1550, sim_phase=unit_cell_phases_wvl_1550)
    unit_cell_1570 = pid.UnitCell(wavelength=1.570, tune_param=unit_cell_diameters_wvl_1570, sim_phase=unit_cell_phases_wvl_1570)
    unit_cells_list = [unit_cell_1530, unit_cell_1550, unit_cell_1570]
  
    initial_space_grid = pid.SpaceGrid(
        side_pixels=N, 
        unit_cell_pitch=unit_cell_pitch, 
        materials_refr_index=mat_refr_index,
        xy_symmetry=True,
    )

    #setup the simulation instance
    simulation = pid.Simulation(
        sources=sources,
        targets=targets,
        central_wavelength=central_wavelength_um,
        space_grid=initial_space_grid,
        n_cpu=1
    )

    simulation.set_optimization(
        opt_param_min=0,
        opt_param_max=1,
        ms_max_avail_diameter=ms_max_available_diameter,
        phase_mask_init_size=output_fiber_core_diameter/3,
        opt_max_eval=opt_max_eval,
        conic_filter_radius=1.0,
        unit_cells=unit_cells_list,
        results_while_solving = False,
        save_results = False,
    )

    return simulation


if __name__ == "__main__":
    simulation = setup_optical_system()
    
    results_folder_name = 'results'

    #phase_mask_1 = np.load(f"{results_folder_name}/filtered_phase_mask_1.npy").flatten()
    #phase_mask_2 = np.load(f"{results_folder_name}/filtered_phase_mask_2.npy").flatten()
    mask1 = np.load(f"{results_folder_name}/mask_1.npy").flatten()
    mask2 = np.load(f"{results_folder_name}/mask_2.npy").flatten()
    
    d0 = 195 #propagation distance: source - MS1 [um] FILTERED mf (using LP01 of SMF)
    d1 = 129 #propagation distance: MS1 - MS2 [um] FILTERED mf (using LP01 of SMF)

    d2 = d0 #propagation distance: MS2 - target [um]
    d = np.array([d0, d1, d2])
    results = simulation.source_sweep(distances = d, input_diameter_masks=[mask1, mask2], save_output=True, apply_filtering=False)
    #make a plot of the results, Use the keys as x axis and values as y axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.array(list(results.keys()))*1e9, np.array(list(results.values())), marker='o', linewidth=2)
    #make the y tick labels to only have two decimal places
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.set_xlabel('wavelength [nm]', fontsize=15)
    ax.set_ylabel('overlap', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.savefig(f"{results_folder_name}/source_sweep_results.pdf")

    #for each of the test wavelength, take the fields from the results folder and make plots of them
    for i, wvl in enumerate(wavelengths_to_test_um):
        save_filename = f"{results_folder_name}/output_field_{wvl}"
        save_filename = save_filename.replace('.', '_') 
        output_field = np.load(save_filename+ ".npy").reshape(simulation.space_grid.get_Nx(), simulation.space_grid.get_Ny())
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pcolormesh(simulation.space_grid.get_X()*1e6, simulation.space_grid.get_Y()*1e6, np.abs(output_field) / np.max(np.abs(output_field)), cmap='viridis', rasterized = True)
        colorbar = fig.colorbar(ax.collections[0], ax=ax, label='Magnitude [a.u.]')
        ax.set_aspect('equal')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        #set the tick fontsize to 20
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        ax.set_ylim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        #overlay a red circle indicating the output fiber core
        circle = plt.Circle((0, 0), output_fiber_core_diameter/2, color='red', fill=False, linewidth=2)
        ax.add_artist(circle)
        ax.set_xlabel('X [um]', fontsize=20)
        ax.set_ylabel('Y [um]', fontsize=20)
        plt.title(r'Output Field at $\lambda=$' + str(wvl) + r' $\mu$m')
        plt.tight_layout()
        plt.savefig(save_filename + ".pdf")
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pcolormesh(simulation.space_grid.get_X()*1e6, simulation.space_grid.get_Y()*1e6, (np.angle(output_field)+2*np.pi)%(2*np.pi), cmap='hsv', vmin=0, vmax=2*np.pi, rasterized = True)
        colorbar = fig.colorbar(ax.collections[0], ax=ax, label='Phase [rad]')
        ax.set_aspect('equal')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        #set the tick fontsize to 20
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        ax.set_ylim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        #overlay a red circle indicating the output fiber core
        circle = plt.Circle((0, 0), output_fiber_core_diameter/2, color='red', fill=False, linewidth=2)
        ax.add_artist(circle)
        ax.set_xlabel('X [um]', fontsize=20)
        ax.set_ylabel('Y [um]', fontsize=20)
        plt.title(r'Output Field Phase at $\lambda=$' + str(wvl) + r' $\mu$m')
        plt.tight_layout()
        plt.savefig(save_filename + "_phase.pdf")
        plt.close()


        target_field_filename = target_files[i]
        target_field = np.load(target_field_filename)
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        # First subplot: output field amplitude
        im = axs[0].pcolormesh(simulation.space_grid.get_X()*1e6, simulation.space_grid.get_Y()*1e6, np.abs(output_field)/np.max(np.abs(output_field)), cmap='viridis', shading='auto', rasterized=True)
        fig.colorbar(im, ax=axs[0], label='Magnitude [a.u.]')
        axs[0].set_aspect('equal')
        axs[0].set_xlim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        axs[0].set_ylim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        circle = plt.Circle((0, 0), output_fiber_core_diameter/2, color='red', fill=False, linewidth=2)
        axs[0].add_artist(circle)
        axs[0].set_xlabel('X [um]', fontsize=20)
        axs[0].set_ylabel('Y [um]', fontsize=20)
        axs[0].tick_params(axis='x', labelsize=20)
        axs[0].tick_params(axis='y', labelsize=20)
        axs[0].tick_params(axis='both', which='major', labelsize=20)
        axs[0].set_title('Output Field Amplitude')
        axs[0].set_aspect('equal')
        # Second subplot: y cross section slice
        y_slice = output_field[:, output_field.shape[1]//2]
        y_target_slice = target_field[:, target_field.shape[1]//2]
        axs[1].plot(simulation.space_grid.get_Y()[:, 0]*1e6, np.abs(y_slice)/np.max(np.abs(y_slice)), label='Output', c='blue', linewidth=2)
        axs[1].plot(simulation.space_grid.get_Y()[:, 0]*1e6, np.abs(y_target_slice)/np.max(np.abs(y_target_slice)), label='Target', c='red', linewidth=2)
        axs[1].set_xlabel('Y [um]', fontsize=20)
        axs[1].set_ylabel('Magnitude [a.u.]', fontsize=20)
        axs[1].set_title('Y Cross Section')
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        axs[1].legend(fontsize=20)
        axs[1].set_aspect('auto')
        axs[1].tick_params(axis='x', labelsize=20)
        axs[1].tick_params(axis='y', labelsize=20)
        axs[1].tick_params(axis='both', which='major', labelsize=20)
        # Third subplot: x cross section slice
        x_slice = output_field[output_field.shape[0]//2, :]
        x_target_slice = target_field[target_field.shape[0]//2, :]
        axs[2].plot(simulation.space_grid.get_X()[0, :]*1e6, np.abs(x_slice)/np.max(np.abs(x_slice)), label='Output', c='blue', linewidth=2)
        axs[2].plot(simulation.space_grid.get_X()[0, :]*1e6, np.abs(x_target_slice)/np.max(np.abs(x_target_slice)), label='Target', c='red', linewidth=2)
        axs[2].set_xlabel('X [um]', fontsize=20)
        axs[2].set_ylim(0, 1)
        axs[2].set_xlim(-output_fiber_core_diameter/2-10, output_fiber_core_diameter/2+10)
        axs[2].set_ylabel('Magnitude [a.u.]', fontsize=20)
        axs[2].set_title('X Cross Section')
        axs[2].legend(fontsize=20)
        axs[2].set_aspect('auto')
        axs[2].tick_params(axis='x', labelsize=20)
        axs[2].tick_params(axis='y', labelsize=20)
        axs[2].tick_params(axis='both', which='major', labelsize=20)
        #remove completely the fourth subplot axis since we only have three now
        fig.delaxes(axs[3])

        plt.tight_layout()
        plt.savefig(save_filename + "_comparison.pdf")
        plt.close()

