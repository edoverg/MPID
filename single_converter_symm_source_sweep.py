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
    central_wavelength_um = 1.55

    wavelengths_to_test_um = [1.530, 1.550, 1.570] #[um]

    source_files = ["fiber_modes/SMF/SMF_LP01_field_distribution_lda_1530_N_1025.npy",
                    "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy",
                    "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1570_N_1025.npy"]

    sources = [pid.Source(input_field=np.load(file), 
                          wavelength=wvl, 
                          field_norm=True) for file, wvl in zip(source_files, wavelengths_to_test_um)]

    target_files = ["fiber_modes/FMF/FMF_LP21_field_distribution_lda_1530_N_1025.npy",
                    "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1550_N_1025.npy",
                    "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1570_N_1025.npy"]

    targets = [pid.Target(input_field=np.load(file), 
                          associated_source=sources[i], 
                          wavelength=wvl, 
                          field_norm=True) for i, (file, wvl) in enumerate(zip(target_files, wavelengths_to_test_um))]
  
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
        #conic_filter_radius=1.0,
        results_while_solving = False,
        save_results = False,
    )

    return simulation


if __name__ == "__main__":
    simulation = setup_optical_system()
    
    results_folder_name = 'results_nf'

    phase_mask_1 = np.load(f"{results_folder_name}/filtered_phase_mask_1.npy").flatten()
    phase_mask_2 = np.load(f"{results_folder_name}/filtered_phase_mask_2.npy").flatten()
    
    d0 = 195 #propagation distance: source - MS1 [um] FILTERED mf (using LP01 of SMF)
    d1 = 129 #propagation distance: MS1 - MS2 [um] FILTERED mf (using LP01 of SMF)

    d2 = d0 #propagation distance: MS2 - target [um]
    d = np.array([d0, d1, d2])
    results = simulation.source_sweep(distances = d, input_phase_masks=[phase_mask_1, phase_mask_2])
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