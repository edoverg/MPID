import mpid as pid
import numpy as np
import os
if not os.path.exists("results"):
    os.makedirs("results")
#Space grid definition
N = 1025
unit_cell_pitch = 700e-9

n0 = 1.45
n1 = 1.0
n2 = 1.45
mat_refr_index = np.array([n0, n1, n2])

d0 = 476e-6 #propagation distance: source - MS1 [m]
d1 = 510e-6 #propagation distance: MS1 - MS2 [m]
d2 = d0 #propagation distance: MS2 - target [m]
d = np.array([d0, d1, d2])

ms_max_available_diameter = 122e-6 #[um]
output_fiber_core_diameter = 19e-6 #[um]
opt_max_eval = 140

space_grid = pid.SpaceGrid(
    side_pixels=N, 
    unit_cell_pitch = unit_cell_pitch, 
    prop_distances=d, 
    materials_refr_index=mat_refr_index,
    xy_symmetry=True,
)

#define the source
beam_waist = 5.2e-6
gaussian_field = np.exp(-(space_grid.get_rho())**2 / (beam_waist)**2) #Gaussian input field, flattened
gaussian_field_normalized = (gaussian_field / np.sqrt(np.sum(np.abs(gaussian_field)**2))) #normalize the input field to have power = 1

wavelengths_to_test_nm = [1550] #[nm]
central_wavelength_m = 1550 * space_grid.nm
sources = [pid.Source(input_field=gaussian_field_normalized, 
                      wavelength=wvl, 
                      space_grid=space_grid, 
                      wvl_unit_measure=space_grid.nm) for wvl in wavelengths_to_test_nm]

sources[0].plot2D(plot_zoom_x=38,plot_zoom_y=38,choose_quantity='amplitude', save_name='results/input_field')
#define targets
#import field data from files and create target objects
target_files = ["fiber_modes/LP21_field_distribution_lda_1550_N_1025.npy",]

target_fields = [np.load(file) for file in target_files]

targets = [pid.Target(input_field=target_fields[i], 
                      associated_source=sources[i], 
                      wavelength=wvl, 
                      space_grid=space_grid, 
                      wvl_unit_measure=space_grid.nm,
                      field_norm = True) for i, (file, wvl) in enumerate(zip(target_files, wavelengths_to_test_nm))]

targets[0].plot2D(plot_zoom_x=38,plot_zoom_y=38,choose_quantity='amplitude', save_name='results/target_field')

#setup the simulation instance
simulation = pid.Simulation(
    sources=sources,
    targets=targets,
    central_wavelength=central_wavelength_m,
    space_grid=space_grid,
    n_cpu=1
)

simulation.set_optimization(
    opt_param_min=0,
    opt_param_max=1,
    ms_max_avail_diameter=ms_max_available_diameter,
    phase_mask_init_size=output_fiber_core_diameter/3,
    opt_max_eval=opt_max_eval,
    conic_filter_radius=1.5e-6,    
)

simulation.run()

best_opt_val, distances = simulation.get_best_opt_and_distances()
print("Best optimization value:", best_opt_val)
print("Distances:", distances)