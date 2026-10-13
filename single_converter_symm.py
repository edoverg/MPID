import mpid as pid
import numpy as np
import os
if not os.path.exists("results"):
    os.makedirs("results")
#Space grid definition
N = 1025
unit_cell_pitch = 0.700

n0 = 1.444
n1 = 1.0
n2 = 1.444
mat_refr_index = np.array([n0, n1, n2])


#d0 = 400 #propagation distance: source - MS1 [um] NO FILT
#d1 = 280 #propagation distance: MS1 - MS2 [um] NO FILT
#d0 = 516 #propagation distance: source - MS1 [um] FILTERED mf (using LP01 of FMF)
#d1 = 467 #propagation distance: MS1 - MS2 [um] FILTERED mf (using LP01 of FMF)
d0 = 195 #propagation distance: source - MS1 [um] FILTERED mf (using LP01 of SMF)
d1 = 129 #propagation distance: MS1 - MS2 [um] FILTERED mf (using LP01 of SMF)

d2 = d0 #propagation distance: MS2 - target [um]
d = np.array([d0, d1, d2])

ms_max_available_diameter = 122 #[um]
output_fiber_core_diameter = 19 #[um]
opt_max_eval = 1000

space_grid = pid.SpaceGrid(
    side_pixels=N, 
    unit_cell_pitch = unit_cell_pitch, 
    prop_distances=d, 
    materials_refr_index=mat_refr_index,
    xy_symmetry=True,
)

#define the source
wavelengths_to_test_um = [1.530,1.550,1.570] #[um]
source_files = ["fiber_modes/SMF/SMF_LP01_field_distribution_lda_1530_N_1025.npy",
                "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy",
                "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1570_N_1025.npy"]
sources = [pid.Source(input_field=np.load(file), 
                      wavelength=wvl, 
                      field_norm=True) for file, wvl in zip(source_files, wavelengths_to_test_um)]

#sources[0].plot2D(plot_zoom_x=38,plot_zoom_y=38,choose_quantity='amplitude', save_name='results/input_field')
#define targets
#import field data from files and create target objects
target_files = ["fiber_modes/FMF/FMF_LP21_field_distribution_lda_1530_N_1025.npy",
                "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1550_N_1025.npy",
                "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1570_N_1025.npy"]

target_fields = [np.load(file) for file in target_files]

targets = [pid.Target(input_field=target_fields[i], 
                      associated_source=sources[i], 
                      wavelength=wvl, 
                      field_norm = True) for i, (file, wvl) in enumerate(zip(target_files, wavelengths_to_test_um))]

#targets[0].plot2D(plot_zoom_x=38,plot_zoom_y=38,choose_quantity='amplitude', save_name='results/target_field')

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

#setup the simulation instance
simulation = pid.Simulation(
    sources=sources,
    targets=targets,
    central_wavelength=wavelengths_to_test_um[0],
    space_grid=space_grid,
    n_cpu=1
)

simulation.set_optimization(
    opt_param_min=0,
    opt_param_max=1,
    ms_max_avail_diameter=ms_max_available_diameter,
    phase_mask_init_size=output_fiber_core_diameter/3,
    unit_cells=unit_cells_list,
    opt_max_eval=opt_max_eval,
    #conic_filter_radius=1.0,    
)

simulation.run()

best_opt_val, distances = simulation.get_best_opt_and_distances()
print("Best optimization value:", best_opt_val)
print("Distances:", distances)