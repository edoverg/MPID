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
)

wavelength = 1.55e-6

#define the sources
source1 = pid.Source(input_field='fiber_modes/LP11_field_distribution_lda_1550_N_1025.npy', 
                     wavelength=wavelength, 
                     space_grid=space_grid, 
                     field_norm=True)
source2 = pid.Source(input_field='fiber_modes/LP21_field_distribution_lda_1550_N_1025.npy', 
                     wavelength=wavelength, 
                     space_grid=space_grid, 
                     field_norm=True)


target = pid.Target(input_field='fiber_modes/LP11_field_distribution_lda_1550_N_1025.npy', 
                    center=(20e-6,0),
                    associated_source=source1, 
                    wavelength=wavelength, 
                    space_grid=space_grid, 
                    field_norm=True)
target2 = pid.Target(input_field='fiber_modes/LP21_field_distribution_lda_1550_N_1025.npy', 
                     center=(-20e-6,0),
                     associated_source=source2, 
                     wavelength=wavelength, 
                     space_grid=space_grid, 
                     field_norm=True) 

#make a plot of sources and targets
target.plot2D(plot_zoom_x=100, plot_zoom_y=100, choose_quantity="amplitude", save_name="results/target_plot")
target2.plot2D(plot_zoom_x=100, plot_zoom_y=100, choose_quantity="amplitude", save_name="results/target2_plot")
source1.plot2D(plot_zoom_x=100, plot_zoom_y=100, choose_quantity="amplitude", save_name="results/source1_plot")
source2.plot2D(plot_zoom_x=100, plot_zoom_y=100, choose_quantity="amplitude", save_name="results/source2_plot")

sim = pid.Simulation(
    sources=[source1, source2],
    targets=[target, target2],
    central_wavelength=wavelength,
    space_grid=space_grid,
    n_cpu=1
)

opt_param_min = 0
opt_param_max = 1

sim.set_optimization(opt_param_min=opt_param_min, 
                     opt_param_max=opt_param_max,
                     ms_max_avail_diameter=ms_max_available_diameter,
                     phase_mask_init_size=output_fiber_core_diameter,
                     opt_max_eval=opt_max_eval,
                     conic_filter_radius=1.5e-6)

sim.run()

best_opt_val, distances = sim.get_best_opt_and_distances()
print("Best optimization value:", best_opt_val)
print("Distances:", distances)