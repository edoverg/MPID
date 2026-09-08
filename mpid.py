import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import os
import pyfftw
import nlopt
from autograd import tensor_jacobian_product, grad as gd
import autograd.numpy as npa
from filters import conic_filter

def make_2Dplot_of(given_field:np.ndarray, grid_X:np.ndarray, grid_Y:np.ndarray, plot_zoom_x:float,plot_zoom_y:float,choose_quantity="amplitude",save_name="plot") -> None:
    '''Makes a plot of the given 2D field in a zoomed region. 
    Saves the plot with the chosen filename to a PDF file. 
    Args:
        given_field: the 2D field to be plotted (flattened)
        choose_quantity: either "amplitude" or "phase" to select quantity to plot
        save_name: name of the file to save the plot (without extension)
        plot_zoom_x: zoom level for the x-axis (in micrometers)
        plot_zoom_y: zoom level for the y-axis (in micrometers)
    Returns:
        None'''
    
    given_field_2d = given_field.reshape((grid_X.shape[0], grid_X.shape[1])) #assure the field is 2D before plotting
    fig, ax = plt.subplots(figsize=(6, 6))
    if choose_quantity == "amplitude":
        ax.pcolormesh(grid_X*1e6,grid_Y*1e6,np.abs(given_field_2d),rasterized=True)
        #add a colorbar for amplitude
        plt.colorbar(ax.pcolormesh(grid_X*1e6,grid_Y*1e6,np.abs(given_field_2d),rasterized=True), ax=ax)
    elif choose_quantity == "phase":
        ax.pcolormesh(grid_X*1e6,grid_Y*1e6,(np.angle(given_field_2d)+2*np.pi)%(2*np.pi), cmap='hsv', vmin=0, vmax=2*np.pi, rasterized=True)
    elif choose_quantity == "phase_mask":
        ax.pcolormesh(grid_X*1e6,grid_Y*1e6,(given_field_2d+2*np.pi)%(2*np.pi), cmap='hsv', vmin=0, vmax=2*np.pi, rasterized=True)
    else:
        raise ValueError("Unexpected value for choose_quantity")

    ax.set_xlim(-plot_zoom_x, plot_zoom_x)
    ax.set_ylim(-plot_zoom_y, plot_zoom_y)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    plt.tight_layout()
    plt.savefig(save_name + ".pdf")
    plt.close()

def make_1Dplot_of(x_axis,y_axis,plot_zoom_x,save_name="plot_1D",) -> None:
    '''Makes a 1D plot of the given data and saves it to a PDF file.
    Args:
        x_axis: list of x-axis data (list of 1D arrays)
        y_axis: list of y-axis data (list of 1D arrays)
        plot_zoom_x: zoom level for the x-axis (in micrometers)
        save_name: name of the file to save the plot (without extension)
    Returns:
        None
    '''
    if len(x_axis) != len(y_axis):
        raise ValueError("Lists x_axis and y_axis must have the same length.")
    plt.figure()
    for i in range(len(x_axis)):
        plt.plot(x_axis[i], y_axis[i])
    plt.xlabel("x [um]")
    plt.tight_layout()
    plt.xlim(-plot_zoom_x, plot_zoom_x)
    plt.savefig(save_name + ".pdf")
    plt.close()

class SpaceGrid:
    '''
    '''
    def __init__(self, side_pixels: int, unit_cell_pitch: float, prop_distances: np.ndarray, materials_refr_index: np.ndarray, xy_symmetry=False):
        self.Nx = side_pixels#for the moment we consider only square grids
        self.Ny = side_pixels
        self.unit_cell_pitch = unit_cell_pitch
        self.nx = (self.Nx - 1) // 2
        self.ny = (self.Ny - 1) // 2
        self.xy_symmetry = xy_symmetry
        self.S = self.Nx * self.Ny

        if self.xy_symmetry:
            self.dof = 2*((self.nx + 1)**2)
        else:
            self.dof = 2 * self.S

        self.size_x = self.unit_cell_pitch * self.Nx
        self.size_y = self.unit_cell_pitch * self.Ny
        self.xs = np.arange(-self.Nx//2+1, self.Nx//2+1) * (self.size_x / self.Nx)
        self.ys = np.arange(-self.Ny//2+1, self.Ny//2+1) * (self.size_y / self.Ny)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.rho = np.sqrt(self.X**2 + self.Y**2)
        self.sampling_period = self.unit_cell_pitch
        self.d = prop_distances

        self.nm = 1e-9
        self.m_to_nm = 1e9
        self.m = 1
        self.um = 1e-6
        self.m_to_um = 1e6
        try:
            if materials_refr_index.shape != self.d.shape:
                raise ValueError("Materials array must have the same shape as d array.")
        except AttributeError:
            raise ValueError("Materials must be an Numpy array.")
        self.materials_refr_index = materials_refr_index

    def get_grid_shape(self):
        return (self.Nx, self.Ny)

    def get_materials_refr_index(self):
        return self.materials_refr_index

    def get_distances(self):
        return self.d

    def get_num_distances(self):
        return len(self.d)
    
    def get_sampling_period(self):
        return self.sampling_period

    def get_resolution(self):
        return self.unit_cell_pitch

    def get_unit_cell_pitch(self):
        return self.unit_cell_pitch

    def get_size_x(self):
        return self.size_x

    def get_size_y(self):
        return self.size_y

    def get_Nx(self):
        return self.Nx

    def get_Ny(self):
        return self.Ny

    def get_nx(self):
        return self.nx

    def get_ny(self):
        return self.ny

    def get_dof(self):
        return self.dof

    def get_S(self):
        return self.S

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def get_XY(self):
        return self.X, self.Y

    def get_rho(self):
        return self.rho
    
class Field:
    def __init__(self, input_field: np.ndarray, wavelength: float, space_grid: SpaceGrid, center=(0,0)):
        self.center = center
        self.space_grid = space_grid
        self.input_field = self._shift_field(input_field)
        self.wavelength = wavelength
        
        
        if self.input_field.shape != self.space_grid.get_grid_shape():
            raise ValueError("Input field must have the same shape as the space grid it is applied onto.")

    def get_field(self, flatten=False):
        if flatten:
            return self.input_field.flatten()
        return self.input_field

    def get_wavelength(self):
        return self.wavelength

    def get_center(self):
        return self.center

    def _shift_field(self,input_field):
        #take the input field and shift it according to the center
        shift_field_x = np.roll(input_field, shift=int(self.center[0]/self.space_grid.get_sampling_period()), axis=1)
        shift_field_y = np.roll(shift_field_x, shift=int(self.center[1]/self.space_grid.get_sampling_period()), axis=0)
        return shift_field_y

    def plot2D(self, plot_zoom_x:float, plot_zoom_y:float, choose_quantity="amplitude", save_name="results/plot_out"):
        make_2Dplot_of(self.input_field, 
                       self.space_grid.X, 
                       self.space_grid.Y,
                       plot_zoom_x, 
                       plot_zoom_y, 
                       choose_quantity, 
                       save_name)

class Source(Field):
    def __init__(self, input_field: np.ndarray | str, wavelength: float, space_grid: SpaceGrid, wvl_unit_measure = 1, field_norm = False, center = (0,0)):
        if isinstance(input_field, str):
            input_field_to_constr = np.load(input_field)
            input_field_shape = input_field_to_constr.shape
            input_field_to_constr = (input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))).reshape(input_field_shape)
        else:
            input_field_to_constr = input_field

        if field_norm:
            input_field_to_constr = input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))

        super().__init__(input_field_to_constr, wavelength * wvl_unit_measure, space_grid, center=center)
        self.type = "source"

    def get_type(self):
        return self.type

class GaussianBeam():
    def __init__(self, beam_waist: float, space_grid: SpaceGrid, center=np.array([0,0]), field_norm = False):
        self.beam_waist = beam_waist
        self.center = center
        self.space_grid = space_grid
        generated_field = self._generate_gaussian_beam()
        if field_norm:
            self.field = generated_field / np.sqrt(np.sum(np.abs(generated_field)**2))
        else:
            self.field = generated_field
        
    def _generate_gaussian_beam(self):
        X = self.space_grid.X - self.center[0]
        Y = self.space_grid.Y - self.center[1]
        r_squared = X**2 + Y**2
        return np.exp(-r_squared / self.beam_waist**2)

    def get_field(self):
        return self.field

    def get_space_grid(self):
        return self.space_grid

class Target(Field):
    def __init__(self, input_field: np.ndarray | str | Field ,associated_source: Source, wavelength: float, space_grid: SpaceGrid, wvl_unit_measure = 1, field_norm = False, center = (0,0)):
        if isinstance(input_field, str):
            input_field_to_constr = np.load(input_field)
            input_field_shape = input_field_to_constr.shape
            input_field_to_constr = (input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))).reshape(input_field_shape)
        else:
            input_field_to_constr = input_field

        if field_norm:
            input_field_to_constr = input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))

        super().__init__(input_field_to_constr, wavelength * wvl_unit_measure, space_grid, center=center)
        self.type = "target"

        if associated_source.get_wavelength() != wavelength * wvl_unit_measure:
            raise ValueError("Associated source must have the same wavelength as the target.")
        else:
            self.associated_source = associated_source
        
    def get_type(self):
        return self.type

    def get_associated_source(self):
        return self.associated_source

class PlotArray():
    def __init__(self, input_field: np.ndarray, space_grid: SpaceGrid):
        self.input_field = input_field
        self.array_shape = input_field.shape
        self.space_grid = space_grid
        self.type = "plot_array"

    def get_type(self):
        return self.type

    def plot2D(self, plot_zoom_x, plot_zoom_y, choose_quantity="amplitude", save_name="results/plot2D_out"):
        if len(self.array_shape) == 2:
            make_2Dplot_of(self.input_field, 
                        self.space_grid.X, 
                        self.space_grid.Y,
                        plot_zoom_x, 
                        plot_zoom_y, 
                        choose_quantity, 
                        save_name)
        else:
            raise ValueError("Input field must be 2D for plot2D.")

    def plot1D(self, plot_zoom_x, save_name="results/plot1D_out"):
        if len(self.array_shape) == 1:
            make_1Dplot_of([self.space_grid.X.flatten()], 
                           [self.input_field.flatten()], 
                           plot_zoom_x, 
                           save_name)
        else:
            raise ValueError("Input field must be 1D for plot1D.")

class Simulation:
    def  __init__(self, sources:list[Source], targets:list[Target], central_wavelength: float, space_grid: SpaceGrid, n_cpu: int = 1):
        self.sources = sources
        self.targets = targets
        self.space_grid = space_grid
        self.central_wavelength = central_wavelength #central_wavelength parameter has little sense to exist. Revise

        self.operating_wavelength = self.central_wavelength #initially, the operating_wavelength is the central wavelength
        self.lda_vector = [self.operating_wavelength / n for n in self.space_grid.get_materials_refr_index()]
        
        self.ks = 2 * np.pi / self.space_grid.get_sampling_period()
        self.kappas = np.arange(-self.space_grid.get_Nx()//2+1, self.space_grid.get_Nx()//2+1) * self.ks / self.space_grid.get_Nx()
        self.KX, self.KY = np.meshgrid(self.kappas, self.kappas)
        self.K_parallel = np.sqrt(self.KX**2 + self.KY**2)
        self.nu_parallel = self.K_parallel / (2 * np.pi)

        self.lda_vector = [self.operating_wavelength / n for n in self.space_grid.get_materials_refr_index()]
        self.k_vectors = [2 * np.pi / lda for lda in self.lda_vector]
        self.phase_factors = [self.k_vectors[i] * self.space_grid.get_distances()[i] * np.sqrt(1 - (self.lda_vector[i] * self.nu_parallel) ** 2 + 0*1j) for i in range(len(self.k_vectors))]

        self.P = [np.exp(1j * self.phase_factors[i]) for i in range(len(self.phase_factors))]
        self.P_nat = [np.fft.ifftshift(self.P[i]) for i in range(len(self.P))]

        #phase masks initialization
        self.w_norm = np.zeros((2*self.space_grid.get_S()))
        self.w_norm_symm = np.zeros(self.space_grid.get_dof())
        self.phase_mask = self._phase_given_w(self.w_norm)
        #initialize intermediate fields
        self.intermediate_fields = [None, None, None]

        #pyfftw init
        self.n_cpu = n_cpu
        self._configure_fftw(self.n_cpu)
        self.fft_input_array = pyfftw.empty_aligned((self.space_grid.get_Nx(), self.space_grid.get_Ny()), dtype='complex128', n=16)
        self.fft_output_array = pyfftw.empty_aligned((self.space_grid.get_Nx(), self.space_grid.get_Ny()), dtype='complex128', n=16)
        self.ifft_input_array = pyfftw.empty_aligned((self.space_grid.get_Nx(), self.space_grid.get_Ny()), dtype='complex128', n=16)
        self.ifft_output_array = pyfftw.empty_aligned((self.space_grid.get_Nx(), self.space_grid.get_Ny()), dtype='complex128', n=16)
        self.fft_input_array.fill(0)
        self.fft_output_array.fill(0)
        self.ifft_input_array.fill(0)
        self.ifft_output_array.fill(0)
        self.fft_operator = pyfftw.FFTW(self.fft_input_array, self.fft_output_array, axes=(0,1), direction='FFTW_FORWARD', threads=self.n_cpu, flags=('FFTW_ESTIMATE',))
        self.ifft_operator = pyfftw.FFTW(self.ifft_input_array, self.ifft_output_array, axes=(0,1), direction='FFTW_BACKWARD', threads=self.n_cpu, flags=('FFTW_ESTIMATE',))

        self.is_optimization_set = False
        self.is_optimized_flag = False
        self.results_while_solving = True
        self.opt_history = []
        self.results_folder = "results"
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def _configure_fftw(self, n_cpu = 1, planner_effort='FFTW_ESTIMATE'):
        pyfftw.config.NUM_THREADS = n_cpu
        pyfftw.interfaces.cache.disable()
        pyfftw.config.PLANNER_EFFORT = planner_effort

    def set_optimization(self, opt_param_min, opt_param_max, ms_max_avail_diameter, phase_mask_init_size, opt_max_eval, conic_filter_radius = None, results_while_solving=True, custom_mask = None):
        '''
        Sets the optimization parameters for the simulation.
        Args:
            optimization_params (dict): A dictionary containing optimization parameters.
        '''
        self.is_optimization_set = True
        #self.optimization_params = optimization_params
        self.iter_num = [0]
        self.norm_phase_min = opt_param_min
        self.norm_phase_max = opt_param_max
        self.ms_max_avail_diameter = ms_max_avail_diameter
        self.phase_mask_init_size = phase_mask_init_size
        self.opt_max_eval = opt_max_eval
        self.results_while_solving = results_while_solving
        self.custom_mask = custom_mask #future: one might be able to directly pass a custom mask
        if conic_filter_radius is None:
            self.use_filtering = False
        else:
            self.use_filtering = True
            self.filter_radius = None
            
        if self.use_filtering:
            self.filter_radius = conic_filter_radius
            filtered_phase_mask_1 = self._phase_filter(self.phase_mask[:self.space_grid.get_S()])
            filtered_phase_mask_2 = self._phase_filter(self.phase_mask[self.space_grid.get_S():])
            self.filtered_phase_mask = np.concatenate((filtered_phase_mask_1, filtered_phase_mask_2))
        else:
            self.filtered_phase_mask = self.phase_mask
        
        mask_upper_bounds = self.norm_phase_max * np.where( self.space_grid.get_X().flatten()**2 + self.space_grid.get_Y().flatten()**2 < (self.ms_max_avail_diameter/2)**2, 1, 0)
        mask_upper_bounds_full = np.concatenate((mask_upper_bounds, mask_upper_bounds),axis=0)
        mask_lower_bounds_full = np.zeros_like(mask_upper_bounds_full)
        
        if self.space_grid.xy_symmetry:
            mask_upper_bounds_symm = mask_upper_bounds.reshape((self.space_grid.get_Nx(),self.space_grid.get_Ny()))[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()+1].flatten()
            mask_upper_bounds_symm_full = np.concatenate((mask_upper_bounds_symm, mask_upper_bounds_symm),axis=0)
            mask_lower_bounds_symm_full = np.zeros_like(mask_upper_bounds_symm_full)
            self.chosen_mask_upper_bounds = mask_upper_bounds_symm_full
            self.chosen_mask_lower_bounds = mask_lower_bounds_symm_full
        else:
            self.chosen_mask_upper_bounds = mask_upper_bounds_full
            self.chosen_mask_lower_bounds = mask_lower_bounds_full

        
    def get_sources(self):
        return self.sources

    def get_targets(self):
        return self.targets

    def get_space_grid(self):
        return self.space_grid

    def get_ms_max_avail_diameter(self):
        return self.ms_max_avail_diameter

    def get_central_wavelength(self):
        return self.central_wavelength

    def get_operating_wavelength(self):
        return self.operating_wavelength

    def get_k_vectors(self):
        return self.k_vectors

    def get_wvl_all_targets(self, unit_measure = 1):
        return [target.get_wavelength() * unit_measure for target in self.targets]

    def _update_wavelength(self, new_wavelength):
        '''
        Updates the operating wavelength and related vectors (lda_vector and k_vectors) based on the new wavelength.
        The new_wavelength should be provided in meters.
        Args:
            new_wavelength (float): The new operating wavelength in meters.
        '''
        assert new_wavelength > 0, "Wavelength must be positive."
        self.operating_wavelength = new_wavelength
        self.lda_vector = [self.operating_wavelength / n for n in self.space_grid.get_materials_refr_index()]
        self.k_vectors = [2 * np.pi / lda for lda in self.lda_vector]
        self.phase_factors = [self.k_vectors[i] * self.space_grid.get_distances()[i] * np.sqrt(1 - (self.lda_vector[i] * self.nu_parallel) ** 2 + 0*1j) for i in range(len(self.k_vectors))]
        self.P = [np.exp(1j * self.phase_factors[i]) for i in range(len(self.phase_factors))]
        self.P_nat = [np.fft.ifftshift(self.P[i]) for i in range(len(self.P))]
        self.P_dagger_nat = [self.P_nat[i].T.conj() for i in range(len(self.P_nat))]

    def _phase_given_w(self):
        return 2 * np.pi * self.w_norm

    def _phase_given_w(self, input_normalized_phase):
        return 2 * np.pi * input_normalized_phase

    def _symmetrize(self, w):
        '''
        Takes an input array, which must be full in size, turns it into a 2D array, and symmetrizes it along the x and y axes.
        Returns the first quadrant of shape (nx,ny), including the zero axis, flattened
        '''
        try:
            w_2d = w.reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
        except:
            raise ValueError(f"Wrong shape for input array. Current shape: {w.shape}")

        w_symm = np.zeros((self.space_grid.get_nx()+1, self.space_grid.get_ny()+1))
        w_symm[:,:] = w_2d[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()+1]

        return w_symm.flatten()

    def _build_circular_mask(self, center, radius):
        '''
        Builds a circular mask centered at the given position with the specified radius.
        '''
        x_center, y_center = center
        X, Y = self.space_grid.get_XY()
        rho = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
        return np.where(rho.flatten() < radius, 1, 0)

    def _initialize_phase_masks(self):
        '''
        Initializes the phase masks for the simulation.
        This method should be called once, before iterating over the targets.
        If multiple source/targets are present in this simulation instance, the mask
        is locally initialized around each source and target.
        '''
    
        source_Efields = [source.get_field(flatten=True) for source in self.sources]
        source_centers = [source.get_center() for source in self.sources]

        target_Efields = [target.get_field(flatten=True) for target in self.targets]
        target_centers = [target.get_center() for target in self.targets]

        self.w_norm = np.zeros((2*self.space_grid.get_S()))

        circular_mask_source = [self._build_circular_mask(source_center, self.phase_mask_init_size) for source_center in source_centers]
        circular_mask_target = [self._build_circular_mask(target_center, self.phase_mask_init_size) for target_center in target_centers]

        source_pattern_phase = [(np.angle(source_Efields[i]) + 2 * np.pi) % (2 * np.pi) * circular_mask_source[i] for i in range(len(source_Efields))] #make sure the phase is between 0 and 2pi
        target_pattern_phase = [(np.angle(target_Efields[i]) + 2 * np.pi) % (2 * np.pi) * circular_mask_target[i] for i in range(len(target_Efields))] #make sure the phase is between 0 and 2pi

        #initialize mask on source side
        cumulated_source_phase = np.zeros_like(self.w_norm[:self.space_grid.get_S()])
        for _, source_phase in enumerate(source_pattern_phase):
            cumulated_source_phase += source_phase / (2 * np.pi) #normalize the source phase to be between 0 and 1

        self.w_norm[:self.space_grid.get_S()] = ((cumulated_source_phase + 2*np.pi) % (2*np.pi)) / (2*np.pi)

        #initialize mask on target side
        cumulated_target_phase = np.zeros_like(self.w_norm[self.space_grid.get_S():])
        for _, target_phase in enumerate(target_pattern_phase):
            cumulated_target_phase += target_phase / (2 * np.pi) #normalize the target phase to be between 0 and 1

        self.w_norm[self.space_grid.get_S():] = ((cumulated_target_phase + 2*np.pi) % (2*np.pi)) / (2*np.pi)
        
        if self.space_grid.xy_symmetry:
            w_norm_symm_1 = self._symmetrize(self.w_norm[:self.space_grid.get_S()])
            w_norm_symm_2 = self._symmetrize(self.w_norm[self.space_grid.get_S():])

            self.w_norm_symm[:] = np.concatenate((w_norm_symm_1, w_norm_symm_2), axis=0)

        self.phase_mask[:] = self._phase_given_w(self.w_norm)
                
    def _phase_filter(self, phase_mask: np.ndarray) -> np.ndarray:
        '''Applies a conic filter on the phase mask. The filtering action is equivalent to a local weighted averaging of the phase.
        Note that due to the circular behavior of phase, a direct averaging of phase values can lead to incorrect results. 
        The following code averages the sine and cosine components separately and then reconstructs the phase mask through the arctangent.
        A practical example might help: if the phase mask contains values close to 0 and 2π, a direct averaging would incorrectly yield a value around π. 
        By averaging the sine and cosine components separately, the correct average near 0 (or 2π) is obtained.

        Args:
            phase_mask (ndarray): The input phase mask to be filtered.

        Returns:
            ndarray: The filtered phase mask.
        '''

        phase_mask_cos = npa.cos(phase_mask)
        phase_mask_sin = npa.sin(phase_mask)

        phase_mask_filtered_cos = conic_filter(
            phase_mask_cos,
            self.filter_radius,
            self.space_grid,
        )

        phase_mask_filtered_sin = conic_filter(
            phase_mask_sin,
            self.filter_radius,
            self.space_grid,
        )
        return npa.arctan2(phase_mask_filtered_sin, phase_mask_filtered_cos)

    def _forward_propagate(self, source_object: Source):

        source_to_propagate = source_object.get_field()

        self.fft_input_array[:,:] = np.fft.ifftshift(source_to_propagate.reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny())))
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[0]
        self.ifft_operator()
        self.intermediate_fields[0] = np.fft.fftshift(self.ifft_operator.output_array)
        input_field = self.intermediate_fields[0].flatten()

        self.fft_input_array[:,:] = np.fft.ifftshift((input_field * np.exp(1j * self.filtered_phase_mask[:self.space_grid.get_S()])).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny())))
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[1] #element wise product
        self.ifft_operator()
        self.intermediate_fields[1] = np.fft.fftshift(self.ifft_operator.output_array.copy())
        #print("Field intensity after first phase mask and propagation: %.4e" % np.sum(np.abs(self.intermediate_fields[0])**2))

        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * np.exp(1j * self.filtered_phase_mask[self.space_grid.get_S():].reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))))
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[2] #element wise product
        self.ifft_operator()
        self.intermediate_fields[2] = np.fft.fftshift(self.ifft_operator.output_array.copy())

        self.output_field = self.intermediate_fields[-1].flatten()

        #make a plot of intermediate fields
        #for i, field in enumerate(self.intermediate_fields):
        #    make_2Dplot_of(field, self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=60, plot_zoom_y=60, choose_quantity="amplitude", save_name=f"results/intermediate_field_{i+1}_plot")


    def _reduce_symmetrized_gradient(self, g_full):
        '''
        Utility function that takes a spatial full size gradient and reduces it 
        to a smaller, symmetrized representation. In other words, it combines
        the contributions from the other quadrants into the selected quadrant.
        '''
        g_2d = g_full.reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))

        third = g_2d[self.space_grid.get_nx()+1:,:self.space_grid.get_ny()+1]
        second = g_2d[:self.space_grid.get_nx()+1, self.space_grid.get_ny()+1:]
        fourth = g_2d[self.space_grid.get_nx()+1:, self.space_grid.get_ny()+1:]

        g_2d[:self.space_grid.get_nx(), self.space_grid.get_ny()+1:] += np.flip(fourth, axis = 0)
        g_2d[:self.space_grid.get_nx(), :self.space_grid.get_ny()+1] += np.flip(third, axis = 0)
        g_2d[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()] += np.flip(second, axis = 1)
        
        return g_2d[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()+1].flatten()

    def _adjoint_propagate(self, target):
        Phi_1_dagger = np.exp(-1j * self.filtered_phase_mask[:self.space_grid.get_S()]).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
        Phi_2_dagger = np.exp(-1j * self.filtered_phase_mask[self.space_grid.get_S():]).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))

        #compute the adjoint source
        self.fft_input_array[:,:] = np.fft.ifftshift(np.sum(self.intermediate_fields[2] * np.conj(target.get_field())) * target.get_field())
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_dagger_nat[2]
        self.ifft_operator()
        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * Phi_2_dagger)

        grad_C_phi_2 = 2 * np.real(-1j * self.intermediate_fields[1].conj() * np.fft.fftshift(self.fft_input_array)).flatten() 

        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_dagger_nat[1] #element wise
        self.ifft_operator()

        grad_C_phi_1 = 2 * np.real(-1j * self.intermediate_fields[0].conj() * np.fft.fftshift(self.ifft_operator.output_array) * Phi_1_dagger).flatten()
        
        return grad_C_phi_1, grad_C_phi_2

    def _value_and_grad(self, target):

        source = target.get_associated_source()
        wavelength = source.get_wavelength()

        self._update_wavelength(wavelength)
        #we shall consider using a conic filter: before doing the forward propagation,
        #we filter the masks
        self._forward_propagate(source)

        C_s = np.real(np.abs(np.sum(self.output_field * np.conj(target.get_field(flatten=True)))) ** 2)
        grad_C_phi_1, grad_C_phi_2 = self._adjoint_propagate(target)

        return C_s, grad_C_phi_1, grad_C_phi_2

    def _expand_symmetrize(self, w_symm):
        '''
        Takes a array of shape (nx+1, ny+1), representing the first quadrant, including zero axes, and 
        expands it to a full 2D array of shape (Nx, Ny) by mirroring, flattened.
        '''
        w_symm_2d = w_symm.reshape((self.space_grid.get_nx()+1, self.space_grid.get_ny()+1))

        w_full = np.zeros((self.space_grid.get_Nx(), self.space_grid.get_Ny()))

        w_full[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()+1] = w_symm_2d
        w_full[:self.space_grid.get_nx()+1, self.space_grid.get_ny()+1:] = np.flip(w_full[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()], axis=1)
        w_full[self.space_grid.get_nx()+1:, :self.space_grid.get_ny()+1] = np.flip(w_full[:self.space_grid.get_nx(), :self.space_grid.get_ny()+1], axis=0)
        w_full[self.space_grid.get_nx()+1:, self.space_grid.get_ny()+1:] = np.flip(np.flip(w_full[:self.space_grid.get_nx(), :self.space_grid.get_ny()], axis=1), axis=0)
        w_full[self.space_grid.get_nx(), self.space_grid.get_ny()] = w_symm_2d[-1, -1]

        return w_full.flatten()

    def _cost_fun(self, x, grad):
        if self.space_grid.xy_symmetry:
            ms1_first_quadrant = x[:self.space_grid.get_dof()//2].reshape(self.space_grid.get_Nx()//2+1, self.space_grid.get_Ny()//2+1)
            ms2_first_quadrant = x[self.space_grid.get_dof()//2:].reshape(self.space_grid.get_Nx()//2+1, self.space_grid.get_Ny()//2+1)
            ms1 = self._expand_symmetrize(ms1_first_quadrant)
            ms2 = self._expand_symmetrize(ms2_first_quadrant)
        else:
            ms1 = x[:self.space_grid.get_dof()//2]
            ms2 = x[self.space_grid.get_dof()//2:]

        all_ms = np.concatenate((ms1, ms2))

        self.w_norm[:] = all_ms
        self.phase_mask[:] = self._phase_given_w(self.w_norm)
        phase_mask_1 = self.phase_mask[:self.space_grid.get_S()]
        phase_mask_2 = self.phase_mask[self.space_grid.get_S():]
        if self.use_filtering:
            filtered_phase_mask_1 = self._phase_filter(phase_mask_1)
            filtered_phase_mask_2 = self._phase_filter(phase_mask_2)

            self.filtered_phase_mask[:] = npa.concatenate((filtered_phase_mask_1, filtered_phase_mask_2))
        else:
            self.filtered_phase_mask[:] = self.phase_mask

        #after the phase mask is set, we test the same mask across all targets
        C_all_wavelengths = []
        grad_partial = np.zeros_like(grad)

        for target in self.targets:

            C, full_grad_1, full_grad_2 = self._value_and_grad(target) 

            C_all_wavelengths.append(C)

            jac_grad_1 = tensor_jacobian_product(self._phase_filter)(phase_mask_1, full_grad_1)
            jac_grad_2 = tensor_jacobian_product(self._phase_filter)(phase_mask_2, full_grad_2)

            if self.space_grid.xy_symmetry:
                jac_grad_1_symm = self._reduce_symmetrized_gradient(jac_grad_1)
                jac_grad_2_symm = self._reduce_symmetrized_gradient(jac_grad_2)
                jac_grad = np.concatenate((jac_grad_1_symm, jac_grad_2_symm), axis=0)
            else:
                jac_grad = np.concatenate((jac_grad_1, jac_grad_2),axis=0)

            grad_partial[:] += jac_grad

        if grad.size > 0:
            grad[:] = grad_partial * 2 * np.pi / len(self.targets) 

        #the new cost function considers the average overlap across all wavelengths.
        self.opt_history.append(np.mean(C_all_wavelengths)) 
        self.iter_num[0] += 1
        print("Iteration:", self.iter_num[0])
        if self.iter_num[0] % 5 == 0 and self.results_while_solving:
            fig, ax = plt.subplots()
            ax.plot(self.opt_history)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cost Function Value')
            ax.set_title('Optimization History')
            fig.tight_layout()
            fig.savefig(f"{self.results_folder}/optimization_history.pdf")
            plt.close(fig)

            #make a plot of the cost function values across 
            # all wavelengths for this iteration
            fig, ax = plt.subplots()
            ax.plot(self.get_wvl_all_targets(unit_measure=self.space_grid.m_to_nm), C_all_wavelengths, marker='o')
            ax.set_xlabel(f'Wavelength [nm]')
            ax.set_ylabel('Cost function value')
            fig.tight_layout()
            fig.savefig(f"{self.results_folder}/cost_function_across_wavelengths.pdf")
            plt.close(fig)

        return np.mean(C_all_wavelengths) 

    def _export_results(self):
        '''
        Placeholder method for exporting the results of the optimization.
        This method should be implemented to save the relevant data.
        '''
        #output fields
        output_field = []
        for i,source in enumerate(self.sources):
            self._forward_propagate(source)
            output_field.append(self.intermediate_fields[-1])

            np.save(f"{self.results_folder}/output_fields_{i}.npy", output_field[-1])
            make_2Dplot_of(output_field[-1], self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=self.get_ms_max_avail_diameter()/2, plot_zoom_y=self.get_ms_max_avail_diameter()/2, choose_quantity="amplitude", save_name=f"{self.results_folder}/output_fields_{i}_plot")
            make_2Dplot_of(output_field[-1], self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=self.get_ms_max_avail_diameter()/2, plot_zoom_y=self.get_ms_max_avail_diameter()/2, choose_quantity="phase", save_name=f"{self.results_folder}/output_fields_{i}_phase_plot")

        #phase masks
        phase_mask_1 = self._phase_given_w(self.w_norm[:self.space_grid.get_S()]).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
        phase_mask_2 = self._phase_given_w(self.w_norm[self.space_grid.get_S():]).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
        filtered_phase_mask_1 = self._phase_filter(phase_mask_1)
        filtered_phase_mask_2 = self._phase_filter(phase_mask_2)
        make_2Dplot_of(phase_mask_1, self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=self.get_ms_max_avail_diameter()/2, plot_zoom_y=self.get_ms_max_avail_diameter()/2, choose_quantity="phase_mask", save_name=f"{self.results_folder}/phase_mask_1_plot")
        make_2Dplot_of(phase_mask_2, self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=self.get_ms_max_avail_diameter()/2, plot_zoom_y=self.get_ms_max_avail_diameter()/2, choose_quantity="phase_mask", save_name=f"{self.results_folder}/phase_mask_2_plot")
        make_2Dplot_of(filtered_phase_mask_1, self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=self.get_ms_max_avail_diameter()/2, plot_zoom_y=self.get_ms_max_avail_diameter()/2, choose_quantity="phase_mask", save_name=f"{self.results_folder}/filtered_phase_mask_1_plot")
        make_2Dplot_of(filtered_phase_mask_2, self.space_grid.get_X(), self.space_grid.get_Y(), plot_zoom_x=self.get_ms_max_avail_diameter()/2, plot_zoom_y=self.get_ms_max_avail_diameter()/2, choose_quantity="phase_mask", save_name=f"{self.results_folder}/filtered_phase_mask_2_plot")

        np.save(f"{self.results_folder}/optimized_phase_mask_1.npy", phase_mask_1)
        np.save(f"{self.results_folder}/optimized_phase_mask_2.npy", phase_mask_2)
        np.save(f"{self.results_folder}/filtered_phase_mask_1.npy", filtered_phase_mask_1)
        np.save(f"{self.results_folder}/filtered_phase_mask_2.npy", filtered_phase_mask_2)

    def get_best_opt_val(self):
        if self.is_optimized_flag:
            return self.best_opt_val
        else:
            raise RuntimeError("Simulation object has not been optimized")

    def get_best_opt_and_distances(self):
        if self.is_optimized_flag:
            return self.best_opt_val, self.space_grid.d
        else:
            raise RuntimeError("Simulation object has not been optimized")

    def is_optimized(self):
        return self.is_optimized_flag

    def update_space_grid(self, new_space_grid: SpaceGrid):
        self.is_optimized_flag = False
        self.best_opt_val = 0
        self.space_grid = new_space_grid
        self.ks = 2 * np.pi / self.space_grid.get_sampling_period()
        self.kappas = np.arange(-self.space_grid.get_Nx()//2+1, self.space_grid.get_Nx()//2+1) * self.ks / self.space_grid.get_Nx()
        self.KX, self.KY = np.meshgrid(self.kappas, self.kappas)
        self.K_parallel = np.sqrt(self.KX**2 + self.KY**2)
        self.nu_parallel = self.K_parallel / (2 * np.pi)

    def run(self):
        '''
        Placeholder method for running the simulation.
        This method should be implemented with the actual simulation logic.
        '''
        if self.is_optimization_set:
            print("Starting optimization...")
            self._initialize_phase_masks()
            #initialize nlopt solver
            solver = nlopt.opt(nlopt.LD_CCSAQ, self.space_grid.get_dof())
            solver.set_lower_bounds(self.chosen_mask_lower_bounds.flatten())
            solver.set_upper_bounds(self.chosen_mask_upper_bounds.flatten())
            solver.set_max_objective(self._cost_fun)
            solver.set_maxeval(self.opt_max_eval)
            solver.set_param("dual_ftol_rel", 1e-7)
            solver.set_param("verbosity",0)

            if self.space_grid.xy_symmetry:
                self.w_norm_symm[:] = solver.optimize(self.w_norm_symm)
                self.is_optimized_flag = True
                self.best_opt_val = solver.last_optimum_value()
                self.w_norm[:self.space_grid.get_S()] = self._expand_symmetrize(self.w_norm_symm[:self.space_grid.get_dof()//2])
                self.w_norm[self.space_grid.get_S():] = self._expand_symmetrize(self.w_norm_symm[self.space_grid.get_dof()//2:])
            else:
                self.w_norm[:] = solver.optimize(self.w_norm)
                self.is_optimized_flag = True
                self.best_opt_val = solver.last_optimum_value()

            self._export_results()
        else:
            raise RuntimeError("Optimization settings have not been specified.")
        