import mpid as pid
import numpy as np
from mpi4py.futures import MPIPoolExecutor

from smt.applications.ego import EGO, KRG
from smt.sampling_methods import LHS
from smt.design_space import (
    DesignSpace,
)
from sklearn.model_selection import KFold, LeaveOneOut
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import copy
import pickle
import os
import gc
if not os.path.exists("results"):
    os.makedirs("results")
#Space grid definition
N = 1025
unit_cell_pitch = 0.700 #i.e., 700 [nm]

n0 = 1.45
n1 = 1.0
n2 = 1.45
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
    wavelength = central_wavelength_um

    wavelengths_to_test_um = [1.530, 1.550, 1.570] #[um]

    source_files = ["fiber_modes/LP01_field_distribution_lda_1530_N_1025.npy",
                    "fiber_modes/LP01_field_distribution_lda_1550_N_1025.npy",
                    "fiber_modes/LP01_field_distribution_lda_1570_N_1025.npy"]

    # source = pid.Source(input_field='fiber_modes/LP01_field_distribution_lda_1550_N_1025.npy', 
    #                     wavelength=wavelength, 
    #                     field_norm=True)

    sources = [pid.Source(input_field=np.load(file), 
                          wavelength=wvl, 
                          field_norm=True) for file, wvl in zip(source_files, wavelengths_to_test_um)]

    target_files = ["fiber_modes/LP21_field_distribution_lda_1530_N_1025.npy",
                    "fiber_modes/LP21_field_distribution_lda_1550_N_1025.npy",
                    "fiber_modes/LP21_field_distribution_lda_1570_N_1025.npy"]

    targets = [pid.Target(input_field=np.load(file), 
                          associated_source=sources[i], 
                          wavelength=wvl, 
                          field_norm=True) for i, (file, wvl) in enumerate(zip(target_files, wavelengths_to_test_um))]
    
    # target = pid.Target(input_field='fiber_modes/LP21_field_distribution_lda_1550_N_1025.npy', 
    #                     associated_source=source, 
    #                     wavelength=wavelength, 
    #                     field_norm=True)
  
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
        results_while_solving = False,
        save_results = False,
    )

    return simulation

def EI(GP, points, f_min):
    pred = - GP.predict_values(points) #GP predicts overlaps (positive)
    var = GP.predict_variances(points)
    args0 = (f_min - pred) / np.sqrt(var)
    args1 = (f_min - pred) * norm.cdf(args0)
    args2 = np.sqrt(var) * norm.pdf(args0)

    if var.size == 1 and var == 0.0:  # can be use only if one point is computed
        return 0.0

    ei = args1 + args2
    return ei[0,0]

def leave_one_out(initial_dataset):
    full_x_doe = initial_dataset[:, :2]
    full_y_doe = initial_dataset[:, 2].reshape(-1, 1)
    loo = LeaveOneOut()

    #available kernels: ‘matern52’, ‘matern32’ ‘abs_exp’, ‘squar_exp’
    kernels_to_test = ['abs_exp', 'squar_exp', 'matern52', 'matern32']
    all_z_values = []

    for kernel in kernels_to_test:
        errs = []
        z_values = []
        true_values = []
        predicted_values = []
        predicted_confidence = []

        for i, (train_index, test_index) in enumerate(loo.split(full_x_doe)):

            print(f"LOO Iteration {i}")

            train_values = full_x_doe[train_index]
            test_values = full_x_doe[test_index]

            train_outputs = full_y_doe[train_index]
            test_outputs = full_y_doe[test_index]

            sm_loo = KRG(design_space=design_space, 
                        n_start=40, 
                        print_global=False,
                        eval_noise=True,
                        corr=kernel,
                        )
            #no_filt has a good fit using matern32
            #filt has a good fit using matern32
            sm_loo.set_training_values(train_values, train_outputs)
            sm_loo.train()

            #this is the predicted value from the surrogate model
            test_pred_values = sm_loo.predict_values(test_values) 
            #for each point in the surrogate model we can get an estimate of the prediction variance, in other
            #words, a measure of how much the model trusts the prediction at that point
            test_pred_variances = sm_loo.predict_variances(test_values) 
            #standardized error: this value should roughly behave as a normal
            #distribution (Gaussian) with mean 0 and variance 1
            z = (test_outputs - test_pred_values) / np.sqrt(test_pred_variances)
            #note that we divide for the standard deviation, thus the deviation at 
            #the numerator shall be within +/- 3 standard deviations, for most points
            z_values.append(z)
            true_values.append(test_outputs)
            predicted_values.append(test_pred_values)
            predicted_confidence.append(np.sqrt(test_pred_variances)) #standard deviation
            sq_err = np.mean((test_pred_values - test_outputs)**2)

            print(f" Squared error for LOO iteration {i}: {sq_err:.6e}")
            errs.append(sq_err)
        all_z_values.append(z_values)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(len(true_values)):
            #confidence interval is set to 3 standard deviations (99.7%)
            ax.errorbar(i, predicted_values[i], yerr=3 * predicted_confidence[i], fmt='o', label='Predicted with confidence' if i == 0 else "")
            ax.scatter(i, true_values[i], marker='x', color='red', label='True value' if i == 0 else "")
        ax.set_xticks(range(len(true_values)))
        ax.set_xlabel('LOO Iteration', fontsize=20)
        ax.set_ylabel('Value', fontsize=20)
        ax.set_title('LOO Predicted vs True Values', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.savefig(f'{results_folder}/loo_validation_kernel_{kernel}.pdf')
        plt.close(fig)

        #make a line plot to show on a diagonal how predicted and true values compare
        true_values_array = np.concatenate(true_values).ravel()
        predicted_values_array = np.concatenate(predicted_values).ravel()

        lower = min(true_values_array.min(), predicted_values_array.min())
        upper = max(true_values_array.max(), predicted_values_array.max())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(true_values_array, predicted_values_array, color="blue")
        ax.plot([lower, upper], [lower, upper], "r--", label="y=x")
        ax.set_xlabel('True Value', fontsize=20)
        ax.set_ylabel('Predicted Value', fontsize=20)
        ax.set_title('LOO Predicted vs True Values', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.savefig(f'{results_folder}/loo_predicted_vs_true_diagonal_kernel_{kernel}.pdf')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        z_values_array = np.concatenate(z_values)
        ax.scatter(range(len(z_values_array)), z_values_array, color='blue', label='Standardized residuals')
        ax.set_ylabel('Standardized Residual', fontsize=20)
        ax.set_title('LOO Standardized Residuals', fontsize=20)
        ax.set_xticks(range(len(z_values_array)))
        ax.set_ylim(min(z_values_array) - 0.5, max(z_values_array) + 0.5)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.savefig(f'{results_folder}/loo_standardized_residuals_kernel_{kernel}.pdf')
        plt.close(fig)

    #in the same plot, represent the normal distribution (0,1) and 
    #all the fitted standardized residuals from the different KRG kernels
    fig, ax = plt.subplots(figsize=(10, 6))
    all_std_values = []
    x = np.linspace(-4, 4, 100)
    for kernel_name, z_vals in zip(kernels_to_test, all_z_values):
        mu, std = norm.fit(np.array(z_vals))
        all_std_values.append(std)
        print(f"FIT VALUES for kernel {kernel_name}: mu =", mu, "std =", std)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, label=f'ker={kernel_name}')


    ax.plot(x, norm.pdf(x, 0, 1), 'g--', label='N(0,1) Gaussian')

    ax.set_xlabel('std. residual', fontsize=20)
    ax.set_ylabel('Standardized Residual', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(f'{results_folder}/loo_standardized_residuals_gaussian_fit.pdf')
    plt.close(fig)

    best_kernel = kernels_to_test[np.argmin(np.array(all_std_values))]

    #train the model using the whole dataset
    sm_final = KRG(design_space=design_space, 
                   n_start=25, 
                   print_global=False, 
                   eval_noise=True, 
                   corr=best_kernel)
    sm_final.set_training_values(full_x_doe, full_y_doe)
    sm_final.train()

    #save the train data
    with open(f'{surrogate_folder_name}/train_data.pkl', 'wb') as f:
        pickle.dump((full_x_doe, full_y_doe), f)

    #save the trained surrogate model
    with open(f'{surrogate_folder_name}/trained_surrogate_model_kernel_{best_kernel}.pkl', 'wb') as f:
        pickle.dump(sm_final, f)
    return sm_final

def run_ego(surrogate_model, n_ego_iter, xlimits, simulation):
    sm = surrogate_model #the surrogate is trained on overlap values (positive)
    #load the associated fold training data
    xdoe_train_loaded, ydoe_train_loaded = pickle.load(open(f'{surrogate_folder_name}/train_data.pkl', 'rb'))
    x_data = copy.deepcopy(xdoe_train_loaded)
    y_data = - copy.deepcopy(ydoe_train_loaded)
    n_start = 25 #number of initial random points to sample for EGO refinement
    x_et_k_tested_history = []
    EI_history = []
    for k in range(n_ego_iter):
        d0_start = np.atleast_2d(xlimits[0][0] + np.random.rand(n_start)*(xlimits[0][1]-xlimits[0][0]))
        d1_start = np.atleast_2d(xlimits[1][0] + np.random.rand(n_start)*(xlimits[1][1]-xlimits[1][0]))
        x_start = np.hstack((d0_start.T, d1_start.T))
        f_min_k = np.min(y_data) #the most negative data point is the best point
        opt_all = np.array(#maximizes the expected improvement in a region around x_st, an element of the start point
            [
                minimize(lambda x: float(-EI(sm, np.atleast_2d(x), f_min_k)), x_st, method="SLSQP", bounds=xlimits)
                for x_st in x_start
            ]
        )
        opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
        obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
        ind_min = np.argmin(obj_success) #in this case, the most negative -EI is the best (maximum EI)
        #save the minimum EI value for this iteration to the history
        EI_history.append(- obj_success[ind_min])
        opt = opt_success[ind_min]
        #select the non-optimal points for history purposes. Later, we will plot
        #all the non-optimal points with crosses to show the search process of the n_start points
        non_optimal_points = [opt_i["x"] for i, opt_i in enumerate(opt_all) if i != ind_min]
        x_et_k = opt["x"] #this is the coordinate of the most interesting point: the one that maximizes the EI (minimize -EI)
        d0_et_k = x_et_k[0]
        d1_et_k = x_et_k[1]
        d2_et_k = d0_et_k
        d_et_k = np.round(np.array([d0_et_k, d1_et_k, d2_et_k]),0)
        print(d_et_k)
        simulation.update_propagation_distances(d_et_k)
        simulation.run()
        y_et_k = - simulation.get_best_opt_val() #evaluate the objective function at the new point
        x_et_k_tested = np.atleast_2d(x_et_k)
        x_et_k_tested_history.append(x_et_k_tested)

        d0_EI = np.linspace(xlimits[0][0], xlimits[0][1], 100)
        d1_EI = np.linspace(xlimits[1][0], xlimits[1][1], 100)
        d0_EI_grid, d1_EI_grid = np.meshgrid(d0_EI, d1_EI)
        EI_grid = np.zeros_like(d0_EI_grid)
        for i in range(d0_EI_grid.shape[0]):
            for j in range(d0_EI_grid.shape[1]):
                EI_grid[i,j] = EI(sm, np.atleast_2d([d0_EI_grid[i,j], d1_EI_grid[i,j]]), f_min_k)

        fig, ax = plt.subplots(figsize=(8,6))
        contour = ax.contourf(d0_EI_grid, d1_EI_grid, EI_grid, levels=50, cmap='viridis')
        fig.colorbar(contour, ax=ax, label='Expected Improvement')
        #scatter the full history of tested points up to the current iteration k, the current point is red, the others are blue
        x_et_k_tested_history_array = np.vstack(x_et_k_tested_history)
        #scatter last point in red, the others in blue
        ax.scatter(x_et_k_tested_history_array[:-1,0], x_et_k_tested_history_array[:-1,1], color='blue', label='Previous best')
        ax.scatter(x_et_k_tested_history_array[-1,0], x_et_k_tested_history_array[-1,1], color='red', label='Best point')
        #also plot the non optimal points with crosses for this iteration
        if len(non_optimal_points) > 0:
            non_optimal_points_array = np.vstack(non_optimal_points)
            ax.scatter(non_optimal_points_array[:,0], non_optimal_points_array[:,1], color='orange', marker='x', label='Candidate')
        ax.set_title(f'EGO Iteration {k+1}/{n_ego_iter}')
        ax.set_xlabel('d0 [um]')
        ax.set_ylabel('d1 [um]')
        ax.set_xlim(xlimits[0])
        ax.set_ylim(xlimits[1])
        ax.legend(loc='upper left', framealpha=0.5)
        fig.tight_layout()
        fig.savefig(f'{surrogate_folder_name}/optimisation_{k}.png')
        plt.close(fig)

        y_data = np.atleast_2d(np.append(y_data, y_et_k)).reshape(-1,1)
        x_data = np.atleast_2d(np.append(x_data, x_et_k_tested, axis=0))
        sm.set_training_values(x_data, - y_data) #train the surrogate with positive values.
        sm.train() #train the model with the new data point added to the previous ones
        print(f"{k}-th EGO refinement complete")
        #for this k iteration out of the n_start, make a plot of the EI surface and append the poisition of the 
        #selected point. Save the figure to a plot list ims, which will be used to create an animation at the end of the EGO loop
        #To test the EI surface, use a custom grid of d0,d1 points, and compute the EI for each point in the grid. Then plot the EI surface as a contour plot, and mark the selected point with a red dot.
        
    ind_best = np.argmin(y_data)
    x_opt = x_data[ind_best]
    y_opt = - y_data[ind_best]
    np.savetxt(f'{surrogate_folder_name}/ego_opt_results.txt', np.hstack((x_opt,y_opt)))
    ims = []
    fig = plt.figure(figsize=[10, 10])

    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    for k in range(n_ego_iter):
        image_pt = mpimg.imread(f"{surrogate_folder_name}/optimisation_{k}.png")
        im = plt.imshow(image_pt)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500)
    #save the animation to a gif file
    ani.save(f'{surrogate_folder_name}/ego_optimization.gif', writer='pillow')

    #plot of the EI history

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(range(1, n_ego_iter+1), EI_history, marker='o')
    ax.set_title('Expected Improvement History')
    ax.set_xlabel('EGO Iteration')
    ax.set_ylabel('Expected Improvement')
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'{surrogate_folder_name}/EI_history.pdf')
    plt.close(fig)

    #surface plot of the refined surrogate model
    d0_grid = np.linspace(xlimits[0][0], xlimits[0][1], 100)
    d1_grid = np.linspace(xlimits[1][0], xlimits[1][1], 100)
    d0_grid_mesh, d1_grid_mesh = np.meshgrid(d0_grid, d1_grid)
    d_grid = np.column_stack((d0_grid_mesh.flatten(), d1_grid_mesh.flatten()))
    sm_predictions = sm.predict_values(d_grid).reshape(d0_grid_mesh.shape)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(d0_grid_mesh, d1_grid_mesh, sm_predictions, cmap='viridis', alpha=0.7, edgecolor='none')
    ax.scatter(xdoe_train_loaded[:,0], xdoe_train_loaded[:,1], ydoe_train_loaded.flatten(), color='blue', label='DOE Points')
    ax.scatter(x_data[len(xdoe_train_loaded):,0], x_data[len(xdoe_train_loaded):,1], - y_data[len(xdoe_train_loaded):].flatten(), color='red', label='EGO Evaluated Points')
    ax.scatter([x_opt[0]], [x_opt[1]], [y_opt], color='green', s=100, label='Best Point', edgecolor='black')
    ax.set_xlabel('d0 [um]')
    ax.set_ylabel('d1 [um]')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{surrogate_folder_name}/refined_surrogate_model_surface.pdf')
    plt.close(fig)
    #contour plot
    fig, ax = plt.subplots(figsize=(8,6))
    cp = ax.contourf(d0_grid_mesh, d1_grid_mesh, sm_predictions, levels=50, cmap='viridis')
    fig.colorbar(cp, ax=ax, label='Surrogate Model Prediction')
    ax.scatter(xdoe_train_loaded[:,0], xdoe_train_loaded[:,1], color='blue', label='DOE Points')
    ax.scatter(x_data[len(xdoe_train_loaded):,0], x_data[len(xdoe_train_loaded):,1], color='red', label='EGO Evaluated Points')
    ax.scatter([x_opt[0]], [x_opt[1]], color='green', s=100, label='Best Point', edgecolor='black')
    ax.set_xlabel('d0 [um]')
    ax.set_ylabel('d1 [um]')
    ax.set_xlim(xlimits[0])
    ax.set_ylim(xlimits[1])
    ax.legend(framealpha=0.5)
    fig.tight_layout()
    fig.savefig(f'{surrogate_folder_name}/refined_surrogate_model_contour.pdf')
    plt.close(fig)

    pickle.dump(sm, open(f'{surrogate_folder_name}/refined_surrogate_model.pkl', 'wb'))
    pickle.dump((x_data, y_data), open(f'{surrogate_folder_name}/refined_model_training_data.pkl', 'wb'))
    return x_opt, y_opt, ind_best, x_data, y_data


##############################
#EGO setup
##############################
if __name__ == "__main__":
    n_ego_iter = 20
    xlimits = np.array([[100,700], [100,700]]) # d0, d1 in [um]
    seed = 777
    np.random.seed(seed)
    design_space = DesignSpace(xlimits, seed=seed)
    criterion = 'EI'
    ndoe = 25 #all the samples will be used for leave-one-out cross-validation

    #first step: check if a DOE exists, if not not, build one
    surrogate_folder_name = 'single_converter_symm_surrogate_mf_filt'
    results_folder = 'single_converter_symm_results_ego_mf_filt'
    if not os.path.exists(surrogate_folder_name):
        os.makedirs(surrogate_folder_name)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    simulation = setup_optical_system()

    if not os.path.exists(f'{surrogate_folder_name}/ego_doe.npy'):
        #build DOE
        print("Building DOE...")
        sampling = LHS(xlimits=xlimits, seed=seed)
        xdoe = np.round(sampling(ndoe),0)
        ydoe = []

        ydoe = np.zeros((len(xdoe), 1), dtype=float)
        
        for i, point in enumerate(xdoe):
            print(f"DOE evaluation {i+1}/{len(xdoe)} for point {point}")
            d0 = point[0]
            d1 = point[1]
            d2 = d0
            d = np.round(np.array([d0, d1, d2]),0)
            simulation.update_propagation_distances(d)
            simulation.run()
            ydoe[i, 0] = simulation.get_best_opt_val()

        with open(f'{surrogate_folder_name}/ego_evaluations.txt', 'a') as f:
            for xd, yd in zip(xdoe, ydoe):
                f.write(f"{xd[0]:.6e} {xd[1]:.6e} {yd[0]:.6e}\n")
    else:
        print("Loading existing DOE...")
        initial_dataset = np.load(f'{surrogate_folder_name}/ego_doe.npy')
        xdoe = initial_dataset[:, :xlimits.shape[0]]
        ydoe = initial_dataset[:, xlimits.shape[0]:]

    initial_dataset = np.hstack((xdoe,ydoe))
    #save the doe
    np.save(f'{surrogate_folder_name}/ego_doe.npy', initial_dataset)

    trained_surrogate_model = leave_one_out(initial_dataset)

    print("Starting EGO optimization...")
    x_opt, y_opt, ind_best, x_data, y_data = run_ego(surrogate_model=trained_surrogate_model, 
                                                    n_ego_iter=n_ego_iter, 
                                                    xlimits=xlimits, 
                                                    simulation=simulation)
