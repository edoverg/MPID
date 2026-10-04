import mpid as pid
import numpy as np
import os
if not os.path.exists("results_sorter"):
    os.makedirs("results_sorter")

#Space grid definition
N = 1025
unit_cell_pitch = 0.700 #i.e., 700 [nm]

n0 = 1.45
n1 = 1.45
n2 = 1.45
n3 = 1.45
mat_refr_index = np.array([n0, n1, n2, n3])

d0 = 500 #propagation distance: source - MS1 [um]
d1 = d0 #propagation distance: MS1 - MS2 [um]
d2 = d0 #propagation distance: MS2 - MS3 [um]
d3 = d0 #propagation distance: MS3 - target [um]
d = np.array([d0, d1, d2, d3])

ms_max_available_diameter = 250 #[um]
output_fiber_core_diameter = 19 #[um]
opt_max_eval = 200

space_grid = pid.SpaceGrid(
    side_pixels=N, 
    unit_cell_pitch = unit_cell_pitch, 
    prop_distances=d, 
    materials_refr_index=mat_refr_index,
)

wavelength = 1.55

#define the sources
source1 = pid.Source(input_field='fiber_modes/FMF/FMF_LP01_field_distribution_lda_1550_N_1025.npy', 
                     wavelength=wavelength, 
                     #space_grid=space_grid, 
                     field_norm=True)
source2 = pid.Source(input_field='fiber_modes/FMF/FMF_LP11_field_distribution_lda_1550_N_1025.npy', 
                     wavelength=wavelength, 
                     #space_grid=space_grid, 
                     field_norm=True)
source3 = pid.Source(input_field='fiber_modes/FMF/FMF_LP21_field_distribution_lda_1550_N_1025.npy', 
                     wavelength=wavelength, 
                     #space_grid=space_grid, 
                     field_norm=True)


target1 = pid.Target(input_field='fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy', 
                    center=np.array([20,0]),
                    associated_source=source1, 
                    wavelength=wavelength, 
                    #space_grid=space_grid, 
                    field_norm=True)
target2 = pid.Target(input_field='fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy', 
                     center=np.array([0,0]),
                     associated_source=source2, 
                     wavelength=wavelength, 
                     #space_grid=space_grid, 
                     field_norm=True)

target3 = pid.Target(input_field='fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy', 
                     center=np.array([-20,0]),
                     associated_source=source3, 
                     wavelength=wavelength, 
                     #space_grid=space_grid, 
                     field_norm=True)


sim = pid.Simulation(
    sources=[source1, source2, source3],
    targets=[target1, target2, target3],
    central_wavelength=wavelength,
    space_grid=space_grid,
    n_cpu=1
)

opt_param_min = 0.0
opt_param_max = 1.0

sim.set_optimization(opt_param_min=opt_param_min, 
                     opt_param_max=opt_param_max,
                     ms_max_avail_diameter=ms_max_available_diameter,
                     phase_mask_init_size=output_fiber_core_diameter,
                     opt_max_eval=opt_max_eval,
                     strategy = 'mean',
                     #conic_filter_radius = 1.0,
                    )

sim.run()

best_opt_val, distances = sim.get_best_opt_and_distances()
print("Best optimization value:", best_opt_val)
print("Distances:", distances)