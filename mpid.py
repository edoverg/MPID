from kiwisolver import Constraint
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import os
import pyfftw
import nlopt
from autograd import tensor_jacobian_product, grad as gd
import autograd.numpy as npa
from filters import Filter

def make_2Dplot_of(given_field:np.ndarray, grid_X:np.ndarray, grid_Y:np.ndarray, plot_zoom_x:float, plot_zoom_y:float, choose_quantity="amplitude", save_name="plot") -> None:
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
        if x_axis[i] is None:
            plt.plot(y_axis[i])
        else:
            plt.plot(x_axis[i], y_axis[i])
    plt.xlabel("x [um]")
    plt.tight_layout()
    if plot_zoom_x is None:
        pass
    else:
        plt.xlim(-plot_zoom_x, plot_zoom_x)
    plt.savefig(save_name + ".pdf")
    plt.close()

class SpaceGrid:
    '''
    Class representing the spatial grid for the simulation.
    Spatial dimensions are intended to be measured in micrometers [um].
    '''
    def __init__(self, side_pixels: int, unit_cell_pitch: float, materials_refr_index: np.ndarray, prop_distances: np.ndarray = None, xy_symmetry=False):
        if prop_distances is None:
            self.propagation_distances_set = False
            self.d = None
        else:
            self.propagation_distances_set = True
            self.d = prop_distances * 1e-6
        
        self.materials_refr_index = materials_refr_index

        self.num_phase_masks = 0
        if self.propagation_distances_set:
            self.num_phase_masks = len(self.d) - 1

        #determine the number of phase masks required based on the number of propagation distances
        self.num_phase_masks = 0
        if self.propagation_distances_set:
            self.num_phase_masks = len(self.d) - 1
        
        self.Nx = side_pixels#for the moment we consider only square grids
        self.Ny = side_pixels
        self.unit_cell_pitch = unit_cell_pitch * 1e-6 # convert to [m]
        self.nx = (self.Nx - 1) // 2
        self.ny = (self.Ny - 1) // 2
        self.xy_symmetry = xy_symmetry
        self.S = self.Nx * self.Ny

        if self.xy_symmetry:
            self.dof = self.num_phase_masks * ((self.nx + 1)**2)
            self.S_symm = (self.nx + 1)**2
        else:
            self.dof = self.num_phase_masks * self.S

        self.size_x = self.unit_cell_pitch * self.Nx
        self.size_y = self.unit_cell_pitch * self.Ny
        self.xs = np.arange(-self.Nx//2+1, self.Nx//2+1) * (self.size_x / self.Nx)
        self.ys = np.arange(-self.Ny//2+1, self.Ny//2+1) * (self.size_y / self.Ny)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)
        self.rho = np.sqrt(self.X**2 + self.Y**2)
        self.sampling_period = self.unit_cell_pitch

    def get_num_phase_masks(self):
        return self.num_phase_masks
        
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
        '''
        Returns the resolution of the spatial grid in micrometers [um].
        '''
        return self.unit_cell_pitch * 1e6 # returns [um]

    def get_unit_cell_pitch(self) -> float:
        '''
        Returns the unit cell pitch of the spatial grid in micrometers [um].
        '''
        return self.unit_cell_pitch * 1e6 # returns [um]

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

    def get_S_symm(self):
        if not self.xy_symmetry:
            raise ValueError("S_symm is only defined when xy_symmetry is set True")
        return self.S_symm

    def get_X(self):
        return self.X

    def get_Y(self):
        return self.Y

    def get_XY(self):
        return self.X, self.Y

    def get_rho(self):
        return self.rho

class UnitCell:
    def __init__(self,wavelength,tune_param:np.ndarray,sim_phase:np.ndarray, sim_magnitude:np.ndarray = None):
        '''
        Constructs a unit cell object with a given tuning parameter and associated simulated transmission phase and magnitude, at the specified wavelength.
        Once initialized, the gradient of the phase with respect to the tuning parameter is computed.

        Parameters:
            wavelength (float): The wavelength associated with the simulation in micrometers [um].
            tune_param (ndarray): The tuning parameter for the unit cell.
            sim_phase (ndarray): The associated simulated phase value.
            sim_magnitude (ndarray, optional): The associated simulated magnitude value. Defaults to None.
        '''
        self.wavelength = wavelength * 1e-6
        self.tune_param = tune_param
        self.sim_phase_raw = np.unwrap(sim_phase - sim_phase[0])
        moving_average_size = 3
        #before filtering, extend the signal by a number of elements equal to the moving average size to reduce edge effects
        #and set those elements equal to the last value of the original signal
        self.sim_phase_raw_extended = np.concatenate([self.sim_phase_raw, np.full(moving_average_size, self.sim_phase_raw[-1])])
        self.sim_phase_filtered = np.convolve(self.sim_phase_raw_extended, np.ones(moving_average_size)/moving_average_size, mode='same')
        #the final value needs to remove the extra elements added for the moving average
        self.sim_phase_filtered = self.sim_phase_filtered[:-moving_average_size]
        self.sim_magnitude = sim_magnitude
        self._compute_gradient()

    def get_wavelength(self):
        '''
        Returns the wavelength, in meters, associated to this unit-cell
        
        Returns:
            float: The wavelength in meters associated with this unit-cell.
        '''
        return self.wavelength

    def _compute_gradient(self)->None:
        '''
        Computes the raw gradient of the simulated phase with respect to the tuning parameter.
        The numpy gradient method is used for the calculation. 
        It uses central differences in the interior and first differences at the boundaries.
        '''
        self.raw_gradient = np.gradient(self.sim_phase_filtered, self.tune_param)

    def compute_gradient_at(self, tune_param_value, is_normalized = False):
        '''
        Computes the gradient of the simulated phase at a specific tuning parameter value.

        Parameters:
            tune_param_value (float): The tuning parameter value at which to compute the gradient.

        Returns:
            float: The gradient of the simulated phase at the specified tuning parameter value.
        '''
        if is_normalized:
            tune_param_value = self.get_param_from_norm(tune_param_value)
        return np.interp(tune_param_value, self.tune_param, self.raw_gradient)

    def get_magnitude_at_param(self, tune_param_value):
        '''
        Retrieves the simulated magnitude at a specific tuning parameter value.

        Parameters:
            tune_param_value (np.ndarray or float): The tuning parameter value at which to retrieve the simulated magnitude.

        Returns:
            np.ndarray or float: The simulated magnitude at the specified tuning parameter value.
        '''
        return np.interp(tune_param_value, self.tune_param, self.sim_magnitude)

    def get_phase_at_param(self, tune_param_value):
        '''
        Retrieves the simulated phase at a specific tuning parameter value.

        Parameters:
            tune_param_value (np.ndarray or float): The tuning parameter value at which to retrieve the simulated phase.

        Returns:
            np.ndarray or float: The simulated phase at the specified tuning parameter value.
        '''
        return np.interp(tune_param_value, self.tune_param, self.sim_phase_filtered)

    def get_param_from_norm(self, normalized_value):
        '''
        Converts a normalized parameter value to the corresponding actual tuning parameter value.

        Parameters:
            normalized_value (np.ndarray or float): The normalized parameter value (between 0 and 1).

        Returns:
            np.ndarray or float: The corresponding actual tuning parameter value.
        '''
        return self.get_minimum_param() + normalized_value * (self.get_maximum_param() - self.get_minimum_param())

    def dw_dw_norm(self):
        '''
        Computes the derivative of the actual tuning parameter with respect to the normalized parameter.
        This is useful for converting gradients from normalized space to physical space.

        Returns:
            float: The derivative of the tuning parameter with respect to the normalized one.
        '''
        return (self.get_maximum_param() - self.get_minimum_param())
        
    def get_gradient(self):
        '''
        Returns the raw gradient of the simulated phase with respect to the tuning parameter.

        Returns:
            np.ndarray: The raw gradient of the simulated phase.
        '''
        return self.raw_gradient

    def get_complete_gradient(self):
        '''
        Returns the complete gradient of the simulated phase with respect to the normalized tuning parameter.

        Returns:
            np.ndarray: The complete gradient of the simulated phase.
        '''
        return self.get_gradient() * self.dw_dw_norm()

    def get_complete_gradient_from_norm_param(self, norm_params):
        '''
        Computes the complete gradient of the simulated phase with respect to the normalized tuning parameter at the specified normalized parameter values.

        Parameters:
            norm_params (np.ndarray or float): The normalized parameter values (between 0 and 1).

        Returns:
            np.ndarray or float: The complete gradient of the simulated phase at the specified normalized parameter values.
        '''
        return self.compute_gradient_at(norm_params, is_normalized=True) * self.dw_dw_norm()

    def get_phase_from_norm_param(self, norm_param):
        '''
        Computes the phase value given a normalized tuning parameter.

        Parameters:
            norm_param (np.ndarray or float): The normalized parameter value (between 0 and 1).

        Returns:
            np.ndarray or float: The corresponding phase value.
        '''
        return self.get_phase_at_param(self.get_param_from_norm(norm_param))

    def get_tune_param(self):
        return self.tune_param

    def get_magnitude(self):
        return self.sim_magnitude

    def get_raw_sim_phase(self):
        return self.sim_phase_raw
    
    def get_filtered_sim_phase(self):
        return self.sim_phase_filtered

    def get_sim_phase(self):
        return self.sim_phase_filtered

    def get_wavelength(self):
        return self.wavelength

    def get_minimum_param(self):
        return np.min(self.tune_param)

    def get_maximum_param(self):
        return np.max(self.tune_param)
    

class Field:
    def __init__(self, input_field: np.ndarray, wavelength: float, center: np.ndarray = np.array([0,0])):
        '''
        Initializes a Field object.

        Parameters:
            input_field (np.ndarray): The input field array.
            wavelength (float): The wavelength of the field in micrometers.
            center (np.ndarray, optional): The center coordinates of the field. Defaults to np.array([0,0]).
        '''
        self.center = center * 1e-6
        self.input_field = input_field
        self.wavelength = wavelength * 1e-6

    def get_field(self, space_grid: SpaceGrid = None, flatten=False):
        if space_grid is None:
            raise ValueError("A SpaceGrid must be provided when invoking get_field()")
        self.space_grid = space_grid
        if flatten:
            return self._shift_field(self.input_field).flatten()
        return self._shift_field(self.input_field)

    def get_field_arr(self):
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
        return 0 #broken atm
        #make_2Dplot_of(self.input_field, 
        #               self.space_grid.X, 
        #               self.space_grid.Y,
        #               plot_zoom_x, 
        #               plot_zoom_y, 
        #               choose_quantity, 
        #               save_name)

class Source(Field):
    def __init__(self, input_field: np.ndarray | str, wavelength: float, field_norm = False, center: np.ndarray = np.array([0,0])):
        if isinstance(input_field, str):
            input_field_to_constr = np.load(input_field)
            input_field_shape = input_field_to_constr.shape
            input_field_to_constr = (input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))).reshape(input_field_shape)
        else:
            input_field_to_constr = input_field

        if field_norm:
            input_field_to_constr = input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))

        super().__init__(input_field_to_constr, wavelength, center=center)
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
    def __init__(self, input_field: np.ndarray | str | Field ,
                 associated_source: Source, 
                 wavelength: float, 
                 field_norm = False, 
                 center = np.array([0,0])):
        if isinstance(input_field, str):
            input_field_to_constr = np.load(input_field)
            input_field_shape = input_field_to_constr.shape
            input_field_to_constr = (input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))).reshape(input_field_shape)
        else:
            input_field_to_constr = input_field

        if field_norm:
            input_field_to_constr = input_field_to_constr / np.sqrt(np.sum(np.abs(input_field_to_constr)**2))

        super().__init__(input_field_to_constr, wavelength, center=center)
        self.type = "target"

        if associated_source.get_wavelength() != self.get_wavelength():
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
    def  __init__(self, sources:list[Source], 
                  targets:list[Target], 
                  central_wavelength: float, 
                  space_grid: SpaceGrid = None, 
                  n_cpu: int = 1):

        self.sources = sources
        self.targets = targets
        self.eps_0 = 8.854e-12
        self.mu_0 = 4*np.pi*1e-7
        self.central_wavelength = central_wavelength * 1e-6 #central_wavelength parameter has little sense to exist. Revise
        self.operating_wavelength = self.central_wavelength #initially, the operating_wavelength is the central wavelength
        self.current_unit_cell = None
        self.use_unit_cells = False #must be set before set_space_grid, which calls _phase_given_w
        self.n_cpu = n_cpu

        if space_grid is not None:
            self.set_space_grid(space_grid)

        self._configure_fftw(self.n_cpu)
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

    def set_optimization(self, 
                         opt_param_min: float, 
                         opt_param_max: float, 
                         ms_max_avail_diameter: float, 
                         phase_mask_init_size: float, 
                         opt_max_eval: int, 
                         conic_filter_radius: float = None,
                         unit_cells: list[UnitCell] = None, 
                         results_while_solving: bool=True, 
                         custom_mask: np.ndarray = None, 
                         strategy: str = 'mean',
                         save_results: bool = True):
        '''
        Sets the optimization parameters for the simulation.

        Arguments related to physical dimensions are expected to be in micrometers [um].

        Args:
            opt_param_min (float): Minimum value of the optimization parameter.
            opt_param_max (float): Maximum value of the optimization parameter.
            ms_max_avail_diameter (float): Maximum available diameter of the metasurface.
            phase_mask_init_size (float): Initial size of the phase mask.
            opt_max_eval (int): Maximum number of optimization evaluations.
            conic_filter_radius (float, optional): Radius of the conic filter. Defaults to None.
            unit_cells (list[UnitCell], optional): List of unit cells to be used. If the list
                is empty, the optimization is performed on phase values. If a list of unit-cell objects is provided,
                the optimization is carried over the provided unit cells, assuming they only have one degree of freedom each.
            results_while_solving (bool, optional): Whether to store results while solving. Defaults to True.
            custom_mask (array-like, optional): Custom mask for the optimization. Defaults to None.
            strategy (str, optional): Optimization strategy. In a multiobjective optimization, this option defines
                how the multiple objectives are combined: 'mean' computes the mean of all objectives;
                'epigraph' takes the epigraph reformulation of the problem.
            save_results (bool, optional): Whether to save the optimization results. Defaults to True.
        '''
        self.is_optimization_set = True
        self.iter_num = [0]
        self.norm_phase_min = opt_param_min
        self.norm_phase_max = opt_param_max
        self.ms_max_avail_diameter = ms_max_avail_diameter * 1e-6
        self.phase_mask_init_size = phase_mask_init_size * 1e-6
        self.opt_max_eval = opt_max_eval
        self.results_while_solving = results_while_solving
        self.custom_mask = custom_mask #future: one might be able to directly pass a custom mask
        self.strategy = strategy
        self.save_results = save_results
        self.unit_cells = unit_cells

        if unit_cells is None or len(unit_cells) == 0:
            self.use_unit_cells = False
        else:
            #if unit_cells are provided, we need to check that there is one unit cell for each wavelength in 
            #the simulation. Moreover, we need to check if minimum and maximum parameters are equal across all the unit cells
            #first: check if the number of unit cells matches the number of wavelengths in the simulation
            if len(unit_cells) != len(self.get_wvl_all_targets()):
                raise ValueError("The number of unit cells must match the number of wavelengths in the simulation.")
            #second: check if the minimum and maximum parameters are equal across all unit cells
            min_param = self.unit_cells[0].get_minimum_param()
            max_param = self.unit_cells[0].get_maximum_param()
            for uc in self.unit_cells:
                if not (np.abs(uc.get_minimum_param() - min_param) < 1e-12) or not (np.abs(uc.get_maximum_param() - max_param) < 1e-12):
                    raise ValueError("All unit cells must have the same minimum and maximum parameters.")
            #third: check if, for each wavelength in the simulation, there exists a unit cell with the same wavelength
            for wvl in self.get_wvl_all_targets():
                if not any(np.abs(uc.get_wavelength() - wvl) < 1e-12 for uc in self.unit_cells):
                    raise ValueError(f"No unit cell found for wavelength {wvl}.")
            self.use_unit_cells = True

        if conic_filter_radius is None:
            self.use_filtering = False
        else:
            self.use_filtering = True
            self.filter_radius = None

        if conic_filter_radius is not None:
            self.filter_radius = conic_filter_radius
            self.filter_obj = Filter(conic_filter_radius, self.space_grid)
        else:
            self.filter_radius = None
            self.filter_obj = None

    def get_sources(self):
        return self.sources

    def get_targets(self):
        return self.targets

    def get_num_targets(self):
        return len(self.targets)

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

    def get_unit_cells(self):
        return self.unit_cells

    def _update_wavelength(self, new_wavelength):
        '''
        Updates the operating wavelength and related vectors (lda_vector and k_vectors) based on the new wavelength.
        The new_wavelength should be provided in meters. If use_unit_cells is True, updating the wavelength
        will also update the current active unit cell.
        Args:
            new_wavelength (float): The new operating wavelength in meters.
        '''
        assert new_wavelength > 0, "Wavelength must be positive."
        self.operating_wavelength = new_wavelength

        #if using unit_cells, update the current unit cell based on the new wavelength
        if self.use_unit_cells:
            self.current_unit_cell = None
            for uc in self.unit_cells:
                if np.abs(uc.get_wavelength() - self.operating_wavelength) < 1e-12:
                    self.current_unit_cell = uc
                    break
            if self.current_unit_cell is None:
                raise ValueError("No matching unit cell found for the current operating wavelength.")
        
        self.lda_vector = [self.operating_wavelength / n for n in self.space_grid.get_materials_refr_index()]
        self.eps_r = (self.space_grid.get_materials_refr_index())**2
        self.k_vectors = [2 * np.pi / lda for lda in self.lda_vector]
        
        self.phase_factors = [self.k_vectors[i] * self.space_grid.get_distances()[i] * np.sqrt(1 - (self.lda_vector[i] * self.nu_parallel) ** 2 + 0*1j) for i in range(len(self.k_vectors))]
        
        self.P = [np.exp(1j * self.phase_factors[i]) for i in range(len(self.phase_factors))]
        self.P_nat = [np.fft.ifftshift(self.P[i]) for i in range(len(self.P))]
        self.P_dagger_nat = [self.P_nat[i].T.conj() for i in range(len(self.P_nat))]

    def _phase_given_w(self, input_normalized_param):
        '''
        Computes the phase corresponding to the given input normalized parameter.
        If unit cells are used, it retrieves the phase values from the unit cell data.
        If unit cells are not used, then the normalized parameters are turned into phase values by simply
        multiplying them by 2 * pi.
        The unit cell object that is used is selected from the available ones according to the current operating wavelength.
        For multi-wavelength optimization, if use_unit_cells is enabled, then there must be a matching unit cell for each operating wavelength.
        '''
        if self.use_unit_cells:
            matching_unit_cell = None
            for uc in self.unit_cells:
                if np.abs(uc.wavelength - self.operating_wavelength) < 1e-12:
                    matching_unit_cell = uc
                    break
            if matching_unit_cell is None:
                raise ValueError("No matching unit cell found for the current operating wavelength.")
            #once the matching unit cell is found, extract the phase information based on the input normalized phase
            true_diameter = matching_unit_cell.get_param_from_norm(input_normalized_param)
            phase = matching_unit_cell.get_phase_at_param(true_diameter)
            return phase
        else:
            return 2 * np.pi * input_normalized_param

    def _w_given_phase(self, input_phase):
        return input_phase / (2 * np.pi)

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
        source_Efields = [source.get_field(space_grid=self.space_grid, flatten=True) for source in self.sources]
        source_centers = [source.get_center() for source in self.sources]

        target_Efields = [target.get_field(space_grid=self.space_grid, flatten=True) for target in self.targets]
        target_centers = [target.get_center() for target in self.targets]

        self.w_norm = np.zeros((self.num_phase_masks*self.space_grid.get_S()))
        self.optimal_w_norm = np.zeros_like(self.w_norm)
        self.best_epigraph = 0
        self.best_opt_val = 0
        circular_mask_source = [self._build_circular_mask(source_center, self.phase_mask_init_size) for source_center in source_centers]
        circular_mask_target = [self._build_circular_mask(target_center, self.phase_mask_init_size) for target_center in target_centers]

        source_pattern_phase = [(np.angle(source_Efields[i]) + 2 * np.pi) % (2 * np.pi) * circular_mask_source[i] for i in range(len(source_Efields))] #make sure the phase is between 0 and 2pi
        target_pattern_phase = [(np.angle(target_Efields[i]) + 2 * np.pi) % (2 * np.pi) * circular_mask_target[i] for i in range(len(target_Efields))] #make sure the phase is between 0 and 2pi

        #initialize mask on source side
        cumulated_source_phase = np.zeros_like(self.w_norm[:self.space_grid.get_S()])
        for _, source_phase in enumerate(source_pattern_phase):
            cumulated_source_phase += source_phase / (2 * np.pi) #normalize the source phase to be between 0 and 1

        #notice that w_norm might either be normalized phase values or normalized diameters. We shall
        #look to see if this difference leads to issues during the optimization.
        self.w_norm[:self.space_grid.get_S()] = ((cumulated_source_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)

        #initialize mask on target side
        cumulated_target_phase = np.zeros(self.space_grid.get_S())
        for _, target_phase in enumerate(target_pattern_phase):
            cumulated_target_phase += target_phase / (2 * np.pi) #normalize the target phase to be between 0 and 1

        #this initialization was intended when only dealing with phase values.
        #in practice, when use_unit_cell is enabled, the value of w_norm represents normalized diameters,
        #not phase values. Let's see how this affects the initialization. 12/10/2026
        if self.num_phase_masks == 2:
            self.w_norm[self.space_grid.get_S():] = ((cumulated_target_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)
        elif self.num_phase_masks == 3:
            self.w_norm[2 * self.space_grid.get_S():] = ((cumulated_target_phase + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)

        if self.space_grid.xy_symmetry:
            if self.num_phase_masks == 2:
                w_norm_symm_1 = self._symmetrize(self.w_norm[:self.space_grid.get_S()])
                w_norm_symm_2 = self._symmetrize(self.w_norm[self.space_grid.get_S():])
            elif self.num_phase_masks == 3:
                w_norm_symm_1 = self._symmetrize(self.w_norm[:self.space_grid.get_S()])
                w_norm_symm_2 = self._symmetrize(self.w_norm[self.space_grid.get_S(): 2 * self.space_grid.get_S()])
                w_norm_symm_3 = self._symmetrize(self.w_norm[2 * self.space_grid.get_S():])
            if self.num_phase_masks == 2:
                self.w_norm_symm[:] = np.concatenate((w_norm_symm_1, w_norm_symm_2), axis=0)
            elif self.num_phase_masks == 3:
                self.w_norm_symm[:] = np.concatenate((w_norm_symm_1, w_norm_symm_2, w_norm_symm_3), axis=0)

        self.phase_mask[:] = self._phase_given_w(self.w_norm)

    def _normParam_filter(self, norm_param: np.ndarray) -> np.ndarray:
        '''Applies a conic filter on normalized parameters. Normalized parameters are the ones the optimization algorithm
        uses and are always in the range [0,1].'''
        # return conic_filter(norm_param,
        #                     self.filter_radius,
        #                     self.space_grid,)
        return self.filter_obj.apply_filter(norm_param)
    
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
        if self.filter_radius is None:
            raise ValueError("Filter radius is not set.")

        phase_mask_cos = npa.cos(phase_mask)
        phase_mask_sin = npa.sin(phase_mask)

        # phase_mask_filtered_cos = conic_filter(
        #     phase_mask_cos,
        #     self.filter_radius,
        #     self.space_grid,
        # )
        phase_mask_filtered_cos = self.filter_obj.apply_filter(phase_mask_cos)
        # phase_mask_filtered_sin = conic_filter(
        #     phase_mask_sin,
        #     self.filter_radius,
        #     self.space_grid,
        # )
        phase_mask_filtered_sin = self.filter_obj.apply_filter(phase_mask_sin)
        return npa.arctan2(phase_mask_filtered_sin, phase_mask_filtered_cos)

    def _forward_propagate(self, source_object: Source, returns_field: bool = False):
        #source field
        source_to_propagate = source_object.get_field(space_grid=self.space_grid)

        #we retrieve the phase masks by managing the filtered phase mask array
        #depending on the number of phase masks, we will split the filtered phase mask array accordingly
        if self.num_phase_masks == 2:
            filtered_phase_mask_1 = self.filtered_phase_mask[:self.space_grid.get_S()]
            filtered_phase_mask_2 = self.filtered_phase_mask[self.space_grid.get_S():]
        elif self.num_phase_masks == 3:
            filtered_phase_mask_1 = self.filtered_phase_mask[:self.space_grid.get_S()]
            filtered_phase_mask_2 = self.filtered_phase_mask[self.space_grid.get_S():2*self.space_grid.get_S()]
            filtered_phase_mask_3 = self.filtered_phase_mask[2*self.space_grid.get_S():]

        self.fft_input_array[:,:] = np.fft.ifftshift(source_to_propagate.reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny())))
        self.fft_operator()

        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[0]
        
        self.ifft_operator()

        #field before MS1
        self.intermediate_electric_fields[0] = np.fft.fftshift(self.ifft_operator.output_array)

        input_field = self.intermediate_electric_fields[0].flatten()

        self.fft_input_array[:,:] = np.fft.ifftshift((input_field * np.exp(1j * filtered_phase_mask_1)).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny())))
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[1] #element wise product
        self.ifft_operator()

        #field before MS2
        self.intermediate_electric_fields[1] = np.fft.fftshift(self.ifft_operator.output_array.copy())

        #print("Field intensity after first phase mask and propagation: %.4e" % np.sum(np.abs(self.intermediate_electric_fields[0])**2))

        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * np.exp(1j * filtered_phase_mask_2.reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))))
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[2] #element wise product
        self.ifft_operator()

        #output field if the number of phase masks is 2, otherwise this is the input field on MS3
        self.intermediate_electric_fields[2] = np.fft.fftshift(self.ifft_operator.output_array.copy())

        #if the number of phase masks is 2, stop here
        if self.num_phase_masks == 2:
            self.output_field = self.intermediate_electric_fields[-1].flatten()
            if returns_field:
                return self.output_field
            else:
                return None

        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * np.exp(1j * filtered_phase_mask_3.reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))))
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_nat[3] #element wise product
        self.ifft_operator()

        self.intermediate_electric_fields[3] = np.fft.fftshift(self.ifft_operator.output_array.copy())

        self.output_field = self.intermediate_electric_fields[-1].flatten()        

        if returns_field:
            return self.output_field
        else:
            return None
        #make a plot of intermediate fields
        #for i, field in enumerate(self.intermediate_electric_fields):
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
        #we need to manage the case of two or three phase masks by defining the associated filtered phase masks
        if self.num_phase_masks == 2:
            filtered_phase_mask_1 = self.filtered_phase_mask[:self.space_grid.get_S()]
            filtered_phase_mask_2 = self.filtered_phase_mask[self.space_grid.get_S():]
        elif self.num_phase_masks == 3:
            filtered_phase_mask_1 = self.filtered_phase_mask[:self.space_grid.get_S()]
            filtered_phase_mask_2 = self.filtered_phase_mask[self.space_grid.get_S():2*self.space_grid.get_S()]
            filtered_phase_mask_3 = self.filtered_phase_mask[2*self.space_grid.get_S():]

        Phi_dagger = []
        Phi_1_dagger = np.exp(-1j * filtered_phase_mask_1).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
        Phi_2_dagger = np.exp(-1j * filtered_phase_mask_2).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
        Phi_dagger.append(Phi_1_dagger)
        Phi_dagger.append(Phi_2_dagger)
        if self.num_phase_masks == 3:
            Phi_3_dagger = np.exp(-1j * filtered_phase_mask_3).reshape((self.space_grid.get_Nx(), self.space_grid.get_Ny()))
            Phi_dagger.append(Phi_3_dagger)


        grad_C_phi = [None] * self.num_phase_masks
        #compute the adjoint source
        S = np.sum(self.intermediate_electric_fields[-1] * np.conj(target.get_field(space_grid=self.space_grid)))
        a = S * target.get_field(space_grid=self.space_grid)
        
        self.fft_input_array[:,:] = np.fft.ifftshift(a)
        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_dagger_nat[-1]
        self.ifft_operator()
        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * Phi_dagger[-1])

        grad_C_phi[-1] = 2 * np.real(-1j * self.intermediate_electric_fields[-2].conj() * np.fft.fftshift(self.fft_input_array)).flatten() 
        #grad_C_phi_2 = 2 * np.real(-1j * self.intermediate_electric_fields[1].conj() * np.fft.fftshift(self.fft_input_array)).flatten() 

        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_dagger_nat[-2] #element wise
        self.ifft_operator()
        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * Phi_dagger[-2])

        grad_C_phi[-2] = 2 * np.real(-1j * self.intermediate_electric_fields[-3].conj() * np.fft.fftshift(self.fft_input_array)).flatten()

        if self.num_phase_masks == 2:
            return grad_C_phi

        self.fft_operator()
        self.ifft_input_array[:,:] = self.fft_operator.output_array * self.P_dagger_nat[-3] #element wise
        self.ifft_operator()
        self.fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(self.ifft_operator.output_array) * Phi_dagger[-3])

        grad_C_phi[-3] = 2 * np.real(-1j * self.intermediate_electric_fields[-4].conj() * np.fft.fftshift(self.fft_input_array)).flatten()

        return grad_C_phi
        #return grad_C_phi_1, grad_C_phi_2

    def _value_and_grad(self, target):
        '''
        Propagates the source field associated with the given target. 
        Returns the value of the cost function and its gradient with respect to phase parameters.

        Args:
            target: The target object for which the cost function and gradient are computed.

        Returns:
            C_s: The value of the cost function.
            grads: The gradient of the cost function with respect to phase parameters.
        '''
        source = target.get_associated_source()
        wavelength = source.get_wavelength()

        self._update_wavelength(wavelength)

        #if use_unit_cells is active, then we the normalized parameters, which
        #represent (filtered) normalized parameters of the unit cells, need to be converted to actual phase values.
        #The actual phase values those diameters correspond to depend on the current unit cell and wavelength
        if self.use_unit_cells: #only necessary if using normalized diameters, otherwise the filtered phase mask is already set.
            self.filtered_phase_mask[:] = self.current_unit_cell.get_phase_from_norm_param(self.filtered_w_norm)

        self._forward_propagate(source)
        #i take the real part just to be sure
        C_s = np.real(np.abs(np.sum(self.output_field * np.conj(target.get_field(space_grid=self.space_grid, flatten=True)))) ** 2)
        grads = self._adjoint_propagate(target)

        return C_s, grads

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

    def _epigraph_constraint(self, result, epigraph_and_weights, gradient)->None:
        '''
        Constraint function for the epigraph formulation. 
        Check nlopt documentation about Vector-valued constraints for additional details.
        
        Args:
            result: 1D array, modified in place, with the results of this constraint
                evaluation. Dimensionality should match the number of constraints.
        
            epigraph_and_weights: 1D array of epigraph (first element) and 
                design weights (remaining elements)
            
            gradient: a 2D array of gradients, modified in place, of the constraint function with respect to the design weights.
            Shape should be mxn, where m is the number of constraints and n is the number of design weights.

        '''
        epigraph = epigraph_and_weights[0]
        x = epigraph_and_weights[1:]

        if self.space_grid.xy_symmetry:
            #if symmetry is used, then x contains the first quadrant of all the metasurfaces involved
            ms = [self._expand_symmetrize(quadrant) for quadrant in self._split_masks(x,self.space_grid.get_S_symm())]
		#the method _split_masks automatically splits a given array into the correct number of metasurfaces
        else:
            ms = self._split_masks(x,self.space_grid.get_dof() // self.num_phase_masks)
            
        self.w_norm[:] = np.concatenate(ms) #at this point w_norm is a 1d array contains the FULL phase masks (all phase masks)
        
        w_masks = self._split_masks(self.w_norm) #w_masks is a list with the individual FULL masks
        if self.use_unit_cells: #conditional to use unit cell gradients
            if self.use_filtering:
                filtered_w_masks = [self._normParam_filter(w) for w in w_masks] #if using cells and filtering, then the correct filter kernel is _normParam_filter
            else:
                filtered_w_masks = [np.copy(w) for w in w_masks]
            self.filtered_w_norm = np.concatenate(filtered_w_masks)
        else:
            #if we do not use unit cells, then simply 
            phase_masks = [self._phase_given_w(w) for w in w_masks] #constructs the phase masks from normalized parameters
            #note that _phase_give_w automatically checks if unit cells are being used. If we do not use unit cells, then normalized 
            #parameters are turned into phase by simply multiplying them by 2pi
            if self.use_filtering:
                filtered_phase_masks = [self._phase_filter(phase) for phase in phase_masks]
            else:
                filtered_phase_masks = phase_masks
            self.filtered_phase_mask[:] = np.concatenate(filtered_phase_masks)
        
        results_array = np.zeros(len(self.targets)) #used to store the overlap value at each target
        grads_matrix = np.zeros((len(self.targets), self.space_grid.get_dof())) #used to store gradients at each target
        
        mC_at_wavelengths = []
        tested_wavelength = []

        for idx, target in enumerate(self.targets):

            C, full_grads = self._value_and_grad(target) 
            mC_at_wavelengths.append(-C)
            tested_wavelength.append(target.wavelength)
            #minus sign required because the constraint is in the form of f - t < 0 and we MINIMIZE the epigraph t
            results_array[idx] = - C

            #apply chain rule if use_unit_cells is active
            #in practice, from dC / dPhi we compute dPhi / dwf * dC / dPhi to get dC / dwf
            if self.use_unit_cells:
                full_grads = [g * self.current_unit_cell.get_complete_gradient_from_norm_param(fw) for g, fw in zip(full_grads, filtered_w_masks)]

            #we now need to apply the tensor jacobian to account for filtering, if active
            if self.use_filtering:
                if self.use_unit_cells:
                    filter_fun, filter_inputs = self._normParam_filter, w_masks
                else:
                    filter_fun, filter_inputs = self._phase_filter, phase_masks
                jac_grads = [tensor_jacobian_product(filter_fun)(inp, g) for inp, g in zip(filter_inputs, full_grads)]
            else:
                jac_grads = full_grads

            if self.space_grid.xy_symmetry:
                jac_grads = [self._reduce_symmetrized_gradient(j) for j in jac_grads]

            if self.use_unit_cells:
                grads_matrix[idx,:] = - np.concatenate(jac_grads)
            else:
                grads_matrix[idx, :] = - np.concatenate(jac_grads) * 2 * np.pi

        result[:] = results_array - epigraph
        
        #check if the current case satisfies the constraints
        epigraph_constraint_satisfied = all(mC <= epigraph + 1e-4 for mC in mC_at_wavelengths)
        if epigraph_constraint_satisfied:
            if max(mC_at_wavelengths) < self.best_epigraph:
                self.best_epigraph = max(mC_at_wavelengths)
                # note that this is done here and not after run(). This is different compared to how
                # the 'mean' strategy is handled.
                # because I have encountered some issues in managing the Nlopt best_opt_val. 
                # In other words, that method does not seem to provide the answer I am after.
                self.best_opt_val = - max(mC_at_wavelengths) #this is turned positive for compatibility with EGO
                self.optimal_w_norm[:] = self.w_norm

        if gradient.size > 0:
            gradient[:,0] = -1
            gradient[:,1:] = grads_matrix


    def _obj_fun(self, epigraph_and_weights, grad):
        ''' Objective function for the epigraph formulation
            
            Args:
                epigraph_and_weights: 1D array of epigraph (first element) and 
                    design weights (remaining elements)
                grad: gradient of the objective function with respect to the design weights, modified in place
            Returns:
                The scalar epigraph variable (the objective function)
        '''

        epigraph = epigraph_and_weights[0]

        if grad.size > 0:
            grad[0] = 1
            grad[1:] = 0

        self.iter_num[0] = self.iter_num[0] + 1
        self.opt_history.append(epigraph)

        self._plot_history()
        self.iter_num[:] += [self.iter_num[0] + 1]
        
        return epigraph

    def _split_masks(self, arr, mask_size=None)->list:
        '''
        Splits an array that concatenates all the phase masks into a list with one entry (a view) per phase mask.
        mask_size defaults to the full size S of a single mask.
        '''
        if mask_size is None:
            mask_size = self.space_grid.get_S()
        return [arr[i*mask_size:(i+1)*mask_size] for i in range(self.num_phase_masks)]

    def _cost_fun(self, x, grad):
        self.iter_num[0] = self.iter_num[0] + 1

        #x holds one first quadrant per mask if xy_symmetry is used, otherwise the full masks
        if self.space_grid.xy_symmetry:
            ms = [self._expand_symmetrize(quadrant) for quadrant in self._split_masks(x, self.space_grid.get_S_symm())]
        else:
            ms = self._split_masks(x, self.space_grid.get_dof() // self.num_phase_masks)

        self.w_norm[:] = np.concatenate(ms)

        #after having set w_norm, we filter. With unit cells the filter acts on the normalized parameters and the phase
        #is retrieved per target in _value_and_grad (it depends on the wavelength); otherwise the filter acts on the phase.
        w_masks = self._split_masks(self.w_norm)
        if self.use_unit_cells:
            if self.use_filtering:
                filtered_w_masks = [self._normParam_filter(w) for w in w_masks]
            else:
                filtered_w_masks = [np.copy(w) for w in w_masks]
            self.filtered_w_norm = np.concatenate(filtered_w_masks)
        else:
            phase_masks = [self._phase_given_w(w) for w in w_masks]
            if self.use_filtering:
                filtered_phase_masks = [self._phase_filter(phase) for phase in phase_masks]
            else:
                filtered_phase_masks = phase_masks
            self.filtered_phase_mask[:] = np.concatenate(filtered_phase_masks)

        #after the phase mask is set, we test the same mask across all targets
        C_all_wavelengths = []
        tested_wavelengths = []
        #per-mask gradient w.r.t. the filter output, accumulated over targets (the filter Jacobian is target-independent)
        grads_wrt_filtered = None

        for target in self.targets:

            C, full_grads = self._value_and_grad(target)  # full_grads = dC / dPhi

            C_all_wavelengths.append(C)
            tested_wavelengths.append(target.get_wavelength())
            if self.use_unit_cells:
                #dC/dw = dC/dphi * dphi/dw, evaluated at the filtered parameters
                full_grads = [g * self.current_unit_cell.get_complete_gradient_from_norm_param(fw)
                              for g, fw in zip(full_grads, filtered_w_masks)]

            #the gradient depends on wavelength, because phase of the unit cell is wavelength-dependent.
            #Each mask accumulates only its own gradient.
            if grads_wrt_filtered is None:
                grads_wrt_filtered = [np.array(g, dtype=float) for g in full_grads]
            else:
                for k, g in enumerate(full_grads):
                    grads_wrt_filtered[k] += g

        #a single backward pass per mask through the filter, which acts on w (unit cells) or on the phase (no unit cells)
        if self.use_filtering:
            if self.use_unit_cells:
                filter_fun, filter_inputs = self._normParam_filter, w_masks
            else:
                filter_fun, filter_inputs = self._phase_filter, phase_masks
            jac_grads = [tensor_jacobian_product(filter_fun)(inp, g) for inp, g in zip(filter_inputs, grads_wrt_filtered)]
        else:
            jac_grads = list(grads_wrt_filtered)

        if self.space_grid.xy_symmetry:
            jac_grads = [self._reduce_symmetrized_gradient(j) for j in jac_grads]
        grad_partial = np.concatenate(jac_grads, axis=0) # = sum over targets of (dC / dPhi) * (dw_f / dw), recall Phi = 2pi * w_f

        if grad.size > 0:

            if self.use_unit_cells:
                grad[:] = grad_partial / len(self.targets)
            else:
                grad[:] = grad_partial * 2 * np.pi / len(self.targets) 

        #the new cost function considers the average overlap across all wavelengths.
        self.opt_history.append(np.mean(C_all_wavelengths))

        self._plot_history()

        if self.iter_num[0] % 5 == 0 and self.results_while_solving:
            fig, ax = plt.subplots()
            ax.plot(np.array(tested_wavelengths)*1e6, C_all_wavelengths, 'o-')
            ax.set_xlabel('Wavelength [um]')
            ax.set_ylabel('Overlap')
            ax.set_title('Overlap Across Wavelengths')
            fig.tight_layout()
            fig.savefig(f"{self.results_folder}/overlap_across_wavelengths.pdf")
            plt.close(fig)

        return np.mean(C_all_wavelengths) 

    def _plot_history(self):
        '''
        Utility function to plot the optimization history and the cost function
        '''
        print("Iteration:", self.iter_num[0])
        if self.iter_num[0] % 5 == 0 and self.results_while_solving:
            fig, ax = plt.subplots()
            ax.plot(self.opt_history)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Epigraph value')
            ax.set_title('Optimization History')
            fig.tight_layout()
            fig.savefig(f"{self.results_folder}/optimization_history.pdf")
            plt.close(fig)

    def _solver_init(self):
        '''
        This method sets up the optimizer based on the chosen strategy.

        '''
        #this filtering seems unnecessary, commenting it out for the moment 12/10/2026
        # if self.use_filtering:
        #     if self.num_phase_masks == 2:
        #         filtered_phase_mask_1 = self._phase_filter(self.phase_mask[:self.space_grid.get_S()])
        #         filtered_phase_mask_2 = self._phase_filter(self.phase_mask[self.space_grid.get_S():])
        #         self.filtered_phase_mask = np.concatenate((filtered_phase_mask_1, filtered_phase_mask_2))
        #     elif self.num_phase_masks == 3:
        #         filtered_phase_mask_1 = self._phase_filter(self.phase_mask[:self.space_grid.get_S()])
        #         filtered_phase_mask_2 = self._phase_filter(self.phase_mask[self.space_grid.get_S():2*self.space_grid.get_S()])
        #         filtered_phase_mask_3 = self._phase_filter(self.phase_mask[2*self.space_grid.get_S():])
        #         self.filtered_phase_mask = np.concatenate((filtered_phase_mask_1, filtered_phase_mask_2, filtered_phase_mask_3))
        # else:
        #     self.filtered_phase_mask = self.phase_mask

        mask_upper_bounds = self.norm_phase_max * np.where(self.space_grid.get_X().flatten()**2 + self.space_grid.get_Y().flatten()**2 < (self.ms_max_avail_diameter/2)**2, 1.0, 0.0)
        mask_upper_bounds_full = np.tile(mask_upper_bounds,self.num_phase_masks)
        mask_lower_bounds_full = np.zeros_like(mask_upper_bounds_full)
        
        if self.space_grid.xy_symmetry:
            mask_upper_bounds_symm = mask_upper_bounds.reshape((self.space_grid.get_Nx(),self.space_grid.get_Ny()))[:self.space_grid.get_nx()+1, :self.space_grid.get_ny()+1].flatten()
            mask_upper_bounds_symm_full = np.tile(mask_upper_bounds_symm,self.num_phase_masks)
            mask_lower_bounds_symm_full = np.zeros_like(mask_upper_bounds_symm_full)
            self.chosen_mask_upper_bounds = mask_upper_bounds_symm_full
            self.chosen_mask_lower_bounds = mask_lower_bounds_symm_full
        else:
            self.chosen_mask_upper_bounds = mask_upper_bounds_full
            self.chosen_mask_lower_bounds = mask_lower_bounds_full

        if self.strategy == 'mean':

            self.solver = nlopt.opt(nlopt.LD_CCSAQ, self.space_grid.get_dof())

            self.solver.set_lower_bounds(self.chosen_mask_lower_bounds.flatten())
            self.solver.set_upper_bounds(self.chosen_mask_upper_bounds.flatten())

            self.solver.set_max_objective(self._cost_fun)
            self.solver.set_maxeval(self.opt_max_eval)

        elif self.strategy == 'epigraph':

            self.solver = nlopt.opt(nlopt.LD_MMA, self.space_grid.get_dof() + 1)

            epigraph_and_weights_lower_bounds = np.insert(self.chosen_mask_lower_bounds.flatten(), 0, -np.inf)
            epigraph_and_weights_upper_bounds = np.insert(self.chosen_mask_upper_bounds.flatten(), 0, np.inf)

            self.solver.set_lower_bounds(epigraph_and_weights_lower_bounds)
            self.solver.set_upper_bounds(epigraph_and_weights_upper_bounds)

            self.solver.set_min_objective(self._obj_fun)
            epigraph_tolerance = np.array([1e-4] * self.get_num_targets())
            self.solver.add_inequality_mconstraint(
                lambda result_, epigraph_and_weights_, grad_: self._epigraph_constraint(
                    result_, 
                    epigraph_and_weights_, 
                    grad_
                ),
                epigraph_tolerance
            )
            self.solver.set_maxeval(self.opt_max_eval)
        else:
            raise ValueError(f"Unknown optimization strategy: {self.strategy}")
        
        self.solver.set_param("dual_ftol_rel", 1e-5)
        self.solver.set_param("inner_maxeval", 30)
        self.solver.set_param("verbosity", 0)

    def _export_results(self):
        '''
        Internal method to export results after the end of the optimization loop
        '''
        #intermediate fields
        #intermediate_field[0] is on MS1
        #intermediate_field[1] is on MS2
        #intermediate_field[2] is the output

        w_masks = self._split_masks(self.w_norm)
        if self.use_unit_cells:
            #true diameters are saved; all unit cells of this simulation share the same parameter range
            if self.use_filtering:
                w_masks = [self._normParam_filter(w) for w in w_masks]
            masks = [self.unit_cells[0].get_param_from_norm(w) for w in w_masks]
        else:
            masks = [self._phase_given_w(w) for w in w_masks]
            if self.use_filtering:
                masks = [self._phase_filter(m) for m in masks]


        #make plots for every mask, and adjust whether it represents diameters or phase values
        #the plot should be zoomed over the maximum metasurface diameter that is allowed by the optimization
        plot_zoom = self.get_ms_max_avail_diameter()
        for i, mask in enumerate(masks, start=1):
            fig, ax = plt.subplots()
            if self.use_unit_cells:
                im = ax.pcolormesh(self.space_grid.X*1e6, self.space_grid.Y*1e6, mask.reshape(self.space_grid.get_grid_shape()),rasterized=True)
                fig.colorbar(im, ax=ax)
                ax.set_title(f"Mask {i} (Diameters)")
            else:
                im = ax.pcolormesh(self.space_grid.X*1e6, self.space_grid.Y*1e6, mask, cmap='hsv', rasterized = True)
                fig.colorbar(im, ax=ax)
                ax.set_title(f"Mask {i} (Phase)")
            ax.set_xlim(-plot_zoom/2*1e6, plot_zoom/2*1e6)
            ax.set_ylim(-plot_zoom/2*1e6, plot_zoom/2*1e6)
            plt.savefig(f"{self.results_folder}/mask_{i}.pdf")
            plt.close(fig)

        for i, mask in enumerate(masks, start=1):
            np.save(f"{self.results_folder}/mask_{i}.npy", mask.reshape(self.space_grid.get_grid_shape()))

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

    def set_space_grid(self, new_space_grid: SpaceGrid):
        '''
        Update the space grid and associate parameters used for propagation.
        '''
        for source in self.sources:
            if source.get_field_arr().shape != new_space_grid.get_grid_shape():
                raise ValueError("The provided space grid is not compatible with the source fields.")
        for target in self.targets:
            if target.get_field_arr().shape != new_space_grid.get_grid_shape():
                raise ValueError("The provided space grid is not compatible with the target fields.")

        self.space_grid = new_space_grid
        self.num_phase_masks = self.space_grid.get_num_phase_masks()

        self.space_grid_is_set = False
        self.is_optimized_flag = False
        self.best_opt_val = 0

        if self.space_grid.get_distances() is not None:
            try:
                if self.space_grid.get_materials_refr_index().shape != self.space_grid.get_distances().shape:
                    raise ValueError("Materials array must have the same shape as d array.")
            except AttributeError:
                raise ValueError("Materials must be an Numpy array.")
            self.space_grid_is_set = True
            self.d = self.space_grid.get_distances()
            print("Using distances: ", self.d)
            self.ks = 2 * np.pi / self.space_grid.get_sampling_period()
            self.kappas = np.arange(-self.space_grid.get_Nx()//2+1, self.space_grid.get_Nx()//2+1) * self.ks / self.space_grid.get_Nx()
            self.KX, self.KY = np.meshgrid(self.kappas, self.kappas)
            self.K_parallel = np.sqrt(self.KX**2 + self.KY**2)
            self.nu_parallel = self.K_parallel / (2 * np.pi)
            
            #phase masks initialization
            self.w_norm = np.zeros((self.num_phase_masks*self.space_grid.get_S()))
            self.w_norm_symm = np.zeros(self.space_grid.get_dof())
            self.phase_mask = self._phase_given_w(self.w_norm)
            self.filtered_phase_mask = np.zeros_like(self.w_norm)
            
            #initialize intermediate fields
            self.intermediate_electric_fields = [None] * (self.num_phase_masks + 1)
            self.intermediate_magnetic_fields = [None] * (self.num_phase_masks + 1)
            
            #pyfftw init
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
        else:
            self.space_grid_is_set = False

    def update_propagation_distances(self, new_distances: np.ndarray):
        '''
        Update the propagation distances of the simulation by creating a new space grid 
        with the specified distances while maintaining the other properties of the current space grid.
        '''
        self.opt_history = [] #updating distances resets the history
        new_space_grid = SpaceGrid(
                side_pixels=self.space_grid.get_Nx(), 
                unit_cell_pitch=self.space_grid.get_unit_cell_pitch(),
                prop_distances=new_distances, 
                materials_refr_index=self.space_grid.get_materials_refr_index(),
                xy_symmetry=True,
            )
        
        self.set_space_grid(new_space_grid)

    def inject_phase_masks(self, input_phase_masks: list, apply_filtering: bool = True):
        """
        Inject the specified phase masks into the simulation, optionally applying filtering based on the `apply_filtering` flag.
        Args:
            input_phase_masks (list): A list of phase masks, flattened, to be injected into the simulation. 
            apply_filtering (bool): Apply filtering to the injected phase masks. Defaults to True.
        Returns:
            None
        """
        if len(input_phase_masks) != self.num_phase_masks:
            raise ValueError(f"Expected {self.num_phase_masks} phase masks, but got {len(input_phase_masks)}.")
        
        for i, phase_mask in enumerate(input_phase_masks):
            #set the phase mask attribute with the input phase masks
            self.phase_mask[i*self.space_grid.get_S():(i+1)*self.space_grid.get_S()] = phase_mask
            #self.w_norm[i*self.space_grid.get_S():(i+1)*self.space_grid.get_S()] = self._w_given_phase(phase_mask)
        
        if apply_filtering: #filtering is applied regardless of the global optimization setting
            if self.use_unit_cells:
                selected_filter = lambda x: self._normParam_filter(x)
            else:
                selected_filter = lambda x: self._phase_filter(x)
            if self.num_phase_masks == 2:
                filtered_phase_mask_1 = selected_filter(self.phase_mask[:self.space_grid.get_S()])
                filtered_phase_mask_2 = selected_filter(self.phase_mask[self.space_grid.get_S():])
                self.filtered_phase_mask = np.concatenate((filtered_phase_mask_1, filtered_phase_mask_2))
            elif self.num_phase_masks == 3:
                filtered_phase_mask_1 = selected_filter(self.phase_mask[:self.space_grid.get_S()])
                filtered_phase_mask_2 = selected_filter(self.phase_mask[self.space_grid.get_S():2*self.space_grid.get_S()])
                filtered_phase_mask_3 = selected_filter(self.phase_mask[2*self.space_grid.get_S():])
                self.filtered_phase_mask = np.concatenate((filtered_phase_mask_1, filtered_phase_mask_2, filtered_phase_mask_3))
        else:
            self.filtered_phase_mask = self.phase_mask

        

    def _update_phase_masks_with_diameters(self, input_diameters_masks: list, apply_filtering: bool = False):
        """
        Constructs the phase masks based on the provided diameter masks and the currently active unit cell.
        Args:
            input_diameters_masks (list): A list of diameter masks, flattened, to be used for updating the phase masks.
        Returns:
            None
        """
        if len(input_diameters_masks) != self.num_phase_masks:
            raise ValueError(f"Expected {self.num_phase_masks} masks, but got {len(input_diameters_masks)}.")

        #check if unit current unit cell is set and not None
        if self.current_unit_cell is None:
            raise ValueError("Cannot generate phase masks from diameters if no unit cell is provided.")

        mask1 = input_diameters_masks[0]
        mask2 = input_diameters_masks[1] if self.num_phase_masks > 1 else None
        mask3 = input_diameters_masks[2] if self.num_phase_masks > 2 else None

        phase_mask_1 = self.current_unit_cell.get_phase_at_param(mask1)
        phase_mask_2 = self.current_unit_cell.get_phase_at_param(mask2) 
        phase_mask_3 = self.current_unit_cell.get_phase_at_param(mask3) if mask3 is not None else None

        phase_mask_list = [phase_mask_1, phase_mask_2, phase_mask_3][:self.num_phase_masks]
        self.inject_phase_masks(phase_mask_list, apply_filtering=apply_filtering)

    def source_sweep(self, distances: np.ndarray = None, input_phase_masks: list = None, input_diameter_masks: list = None, apply_filtering: bool = True, save_output: bool = False):
        """
        Runs the propagation algorithm for the specified sources and targets in the simulation object. 
        There are two operating modes of this method: if phase masks are provided via the `input_phase_masks` argument, 
        If 'input_diameter_masks' is provided, phase masks are
        updated based on the source wavelength and unit cell data. If none of the two are provided, the last phase masks
        One cannot provide both `input_phase_masks` and `input_diameter_masks` simultaneously.
        Example: if the simulation contains two targets T1 and T2, with associated sources S1, S2, at two distinct wavelengths L1, L2, passing input_diameters_masks
        will update the phase masks for each source by selecting the appropriate unit cell based on the source wavelength and the provided diameter mask.
        Unit cell data must be already be present in the simulation object.
        It is typically used for studying the wavelength-dependent performance of the system.
        Args:
            distances (np.ndarray, optional): Array containing the propagation distances. If None, the current distances in the simulation object are used.
            input_phase_masks (list, optional): A list of phase masks to be injected into the simulation. If None, the current phase masks are used.
            input_diameter_masks (list, optional): A list of diameter masks to be used for updating the phase masks based on the source wavelength and unit cell data.
            apply_filtering (bool, optional): Controls filtering with higher priority than Simulation settings. Defaults to True.
            save_output (bool, optional): Save the output results to a file. Default is False.

        Returns:
            dict: A dictionary mapping source wavelengths to the overlap between the output field and the associated target field.
        """

        #setup
        if input_diameter_masks is not None and input_phase_masks is not None:
            raise ValueError("Cannot choose: both input_phase_masks and input_diameter_masks were provided.")

        results_dict = {}

        if distances is not None:
            self.update_propagation_distances(distances)

        if input_phase_masks is not None:
            self.inject_phase_masks(input_phase_masks, apply_filtering=apply_filtering)

        if input_diameter_masks is not None:
            use_unit_cells = True
        else:
            use_unit_cells = False

        #simulate
        for target in self.targets:
            source = target.get_associated_source()
            wavelength = source.get_wavelength()
            self._update_wavelength(wavelength)
            #retrieve the appropriate phase mask from the simulation's active unit cell
            if use_unit_cells:
                self._update_phase_masks_with_diameters(input_diameter_masks, apply_filtering=apply_filtering)
            output_field = self._forward_propagate(source, returns_field = True)
            #compute the overlap between the output field and associated target field
            target_field = target.get_field(self.get_space_grid(), flatten = True)

            overlap = np.abs(np.sum(np.conj(target_field) * output_field))**2
            results_dict[source.get_wavelength()] = overlap
            if save_output:
                save_filename = f"{self.results_folder}/output_field_{str(np.round(wavelength*1e6, 2))}"
                save_filename = save_filename.replace('.', '_') + ".npy"
                np.save(save_filename, output_field)

        return results_dict
            
    def run(self):
        """
        Run the optimization process for the simulation.
        This method checks if the space grid and optimization settings are set,
        initializes the phase masks, and performs the optimization based on the chosen strategy.
        """
        if self.space_grid_is_set:
            if self.is_optimization_set:
                print(f"Using {self.num_phase_masks} phase masks")
                print("Initializing solver...")
                self._solver_init()
                print("Starting optimization...")
                self._initialize_phase_masks()
                #initialize nlopt solver

                if self.space_grid.xy_symmetry:
                    if self.strategy == 'epigraph':
                        epigraph_w_norm_symm = self.solver.optimize(np.insert(self.w_norm_symm, 0, 0))
                        self.w_norm_symm[:] = epigraph_w_norm_symm[1:]
                        self.w_norm[:] = self.optimal_w_norm
                    elif self.strategy == 'mean':
                        self.w_norm_symm[:] = self.solver.optimize(self.w_norm_symm)
                        self.best_opt_val = self.solver.last_optimum_value()
                        if self.num_phase_masks == 2:
                            self.w_norm[:self.space_grid.get_S()] = self._expand_symmetrize(self.w_norm_symm[:self.space_grid.get_S_symm()])
                            self.w_norm[self.space_grid.get_S():] = self._expand_symmetrize(self.w_norm_symm[self.space_grid.get_S_symm():])
                        elif self.num_phase_masks == 3:
                            self.w_norm[:self.space_grid.get_S()] = self._expand_symmetrize(self.w_norm_symm[:self.space_grid.get_S_symm()])
                            self.w_norm[self.space_grid.get_S():2*self.space_grid.get_S()] = self._expand_symmetrize(self.w_norm_symm[self.space_grid.get_S_symm():2*self.space_grid.get_S_symm()])
                            self.w_norm[2*self.space_grid.get_S():] = self._expand_symmetrize(self.w_norm_symm[2*self.space_grid.get_S_symm():])
                    else:
                        raise ValueError(f"Unknown strategy: {self.strategy}")

                    self.is_optimized_flag = True  
                else:
                    if self.strategy == 'epigraph':
                        self.epigraph_w_norm = np.insert(self.w_norm, 0, 0)
                        self.epigraph_w_norm[:] = self.solver.optimize(self.epigraph_w_norm)
                        #self.w_norm[:] = self.epigraph_w_norm[1:]
                        self.w_norm[:] = self.optimal_w_norm #21/10/2026
                        #self.best_opt_val = self.best_epigraph
                    elif self.strategy == 'mean':
                        self.w_norm[:] = self.solver.optimize(self.w_norm)
                        self.best_opt_val = self.solver.last_optimum_value()
                    else:
                        raise ValueError(f"Unknown strategy: {self.strategy}")
                    
                    self.is_optimized_flag = True
                    

                if self.save_results:
                    self._export_results()

                self.iter_num[:] = [0]
                #might need to take the absolute value to make both average and epigraph strategies
                #compatible with the ego script
                return self.best_opt_val
            else:
                raise RuntimeError("Optimization settings have not been specified.")
        else:
            raise RuntimeError("Space grid has not been specified.")