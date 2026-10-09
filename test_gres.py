import numpy as np
import matplotlib.pyplot as plt
import pyfftw
import os
import nlopt
from scipy.sparse.linalg import gmres, LinearOperator

results_folder = "fold_gmres"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Physical parameters
lda0 = 1.55e-6            # Wavelength [m]
k0 = 2 * np.pi / lda0
n_sub = 1.444            # Substrate index
lambda_sub = lda0 / n_sub
k_sub = 2 * np.pi / lambda_sub
substrate_thickness = 500e-6

# Grid definition
Nx = 1025
Ny = Nx
dof = Nx * Ny
unit_cell_pitch = 700e-9
size_x = unit_cell_pitch * Nx
size_y = unit_cell_pitch * Ny

xs = (np.arange(Nx) - Nx // 2) * unit_cell_pitch
ys = (np.arange(Ny) - Ny // 2) * unit_cell_pitch
X, Y = np.meshgrid(xs, ys)

# Spatial frequencies (FFT order)
kx = np.fft.fftfreq(Nx, d=unit_cell_pitch) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=unit_cell_pitch) * 2 * np.pi
Kx, Ky = np.meshgrid(kx, ky)
K_parallel_sq = Kx**2 + Ky**2

# Propagation Kernel (FFT order)
kz_sq = k_sub**2 - K_parallel_sq
propagating_mask = kz_sq >= 0

# Evanescent wave filtering
kz = np.where(propagating_mask, np.sqrt(np.maximum(0, kz_sq)), 0.0)
P_nat = np.where(propagating_mask, np.exp(1j * kz * substrate_thickness), 0.0)
P_adj = np.conj(P_nat.T)
# Source definition
pixel_shift = 400
source_radius = 21e-6
output_radius = 21e-6
source_mask = np.where(X**2 + Y**2 <= source_radius**2, 1.0, 0.0)
output_mask = np.where(X**2 + Y**2 <= output_radius**2, 1.0, 0.0)
source_mask = np.roll(source_mask, shift=(0, -pixel_shift), axis=(0, 1))
output_mask_1 = np.roll(output_mask, shift=(0, pixel_shift), axis=(0, 1))
output_mask_2 = np.roll(output_mask, shift=(int(60e-6/unit_cell_pitch), pixel_shift), axis=(0, 1))
output_mask_3 = np.roll(output_mask, shift=(int(-60e-6/unit_cell_pitch), pixel_shift), axis=(0, 1))
#output_mask = output_mask_1 + output_mask_2 + output_mask_3
output_mask = output_mask_1 #temporary
I_output = output_mask.flatten()

#import a fiber mode profile and position it over the source mask
source_fiber_mode1 = "fiber_modes/FMF/FMF_LP01_field_distribution_lda_1550_N_1025.npy"
source_fiber_mode2 = "fiber_modes/FMF/FMF_LP11_field_distribution_lda_1550_N_1025.npy"
source_fiber_mode3 = "fiber_modes/FMF/FMF_LP21_field_distribution_lda_1550_N_1025.npy"
target_fiber_mode1 = "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy"
target_fiber_mode2 = "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy"
target_fiber_mode3 = "fiber_modes/SMF/SMF_LP01_field_distribution_lda_1550_N_1025.npy"

#import the file and shift it by an 
import_source_field_1 = np.load(source_fiber_mode1)
import_source_field_2 = np.load(source_fiber_mode2)
import_source_field_3 = np.load(source_fiber_mode3)
import_target_field_1 = np.load(target_fiber_mode1)
import_target_field_2 = np.load(target_fiber_mode2)
import_target_field_3 = np.load(target_fiber_mode3)
#pad the imported source fields 
import_source_field_1 = np.pad(import_source_field_1, (((Nx - import_source_field_1.shape[0])//2,(Nx - import_source_field_1.shape[0])//2 ), ( (Ny - import_source_field_1.shape[1])//2, (Ny - import_source_field_1.shape[1])//2 )), mode='constant')
import_source_field_2 = np.pad(import_source_field_2, (((Nx - import_source_field_2.shape[0])//2,(Nx - import_source_field_2.shape[0])//2 ), ( (Ny - import_source_field_2.shape[1])//2, (Ny - import_source_field_2.shape[1])//2 )), mode='constant')
import_source_field_3 = np.pad(import_source_field_3, (((Nx - import_source_field_3.shape[0])//2,(Nx - import_source_field_3.shape[0])//2 ), ( (Ny - import_source_field_3.shape[1])//2, (Ny - import_source_field_3.shape[1])//2 )), mode='constant')
import_target_field_1 = np.pad(import_target_field_1, (((Nx - import_target_field_1.shape[0])//2,(Nx - import_target_field_1.shape[0])//2 ), ( (Ny - import_target_field_1.shape[1])//2, (Ny - import_target_field_1.shape[1])//2 )), mode='constant')
import_target_field_2 = np.pad(import_target_field_2, (((Nx - import_target_field_2.shape[0])//2,(Nx - import_target_field_2.shape[0])//2 ), ( (Ny - import_target_field_2.shape[1])//2, (Ny - import_target_field_2.shape[1])//2 )), mode='constant')
import_target_field_3 = np.pad(import_target_field_3, (((Nx - import_target_field_3.shape[0])//2,(Nx - import_target_field_3.shape[0])//2 ), ( (Ny - import_target_field_3.shape[1])//2, (Ny - import_target_field_3.shape[1])//2 )), mode='constant')

#shift the source fields to align with the source mask
source_field_1 = np.roll(import_source_field_1, shift=(0, -pixel_shift), axis=(0, 1)) / np.sum(np.abs(import_source_field_1) ** 2)
source_field_2 = np.roll(import_source_field_2, shift=(0, -pixel_shift), axis=(0, 1)) / np.sum(np.abs(import_source_field_2) ** 2)
source_field_3 = np.roll(import_source_field_3, shift=(0, -pixel_shift), axis=(0, 1)) / np.sum(np.abs(import_source_field_3) ** 2)

# Angle of incidence
input_angle = 8.0 / 180 * np.pi  # radians
deflection_phase = np.exp(1j * k_sub * X * np.sin(input_angle))
tilted_field_1 = source_field_1 * deflection_phase
tilted_field_2 = source_field_2 * deflection_phase
tilted_field_3 = source_field_3 * deflection_phase

#target fields
target_field_1 = np.roll(import_target_field_1, shift=(0, -pixel_shift), axis=(0, 1)) / np.sum(np.abs(import_target_field_1) ** 2)
target_field_2 = np.roll(import_target_field_2, shift=(0, -pixel_shift), axis=(0, 1)) / np.sum(np.abs(import_target_field_2) ** 2)
target_field_3 = np.roll(import_target_field_3, shift=(0, -pixel_shift), axis=(0, 1)) / np.sum(np.abs(import_target_field_3) ** 2)

#phase mask boundary region
phase_mask_region = np.ones((Nx, Ny)) - output_mask - source_mask
boundary_X_width = 150  # number of pixels to exclude from the edges
boundary_Y_width = 150
phase_mask_region[:boundary_Y_width, :] = 0
phase_mask_region[-boundary_Y_width:, :] = 0
phase_mask_region[:, :boundary_X_width] = 0
phase_mask_region[:, -boundary_X_width:] = 0

# w_norm boundaries for nlopt
w_norm_upper_boundary = np.where(phase_mask_region,1,0)
w_norm_lower_boundary = np.zeros_like(w_norm_upper_boundary)

gmres_data_folder = 'gmres_data'
def initialize_phase_mask():
    '''Utility function used to import phase masks from a previous
    stacked simulation.'''
    phase_mask_1 = np.load(f"{gmres_data_folder}/phase_mask_1.npy")
    phase_mask_2 = np.load(f"{gmres_data_folder}/phase_mask_2.npy")
    phase_mask_3 = np.load(f"{gmres_data_folder}/phase_mask_3.npy")

    center_mask_1_x = 2 * substrate_thickness * np.atan(input_angle)
    center_mask_1_y = 0

    center_mask_2_x = 4 * substrate_thickness * np.atan(input_angle)
    center_mask_2_y = 0

    center_mask_3_x = 6 * substrate_thickness * np.atan(input_angle)
    center_mask_3_y = 0

    #pad the phase masks so that they have the same size as the spatial grid. By padding I mean adding 
    # zero-padding around the phase masks

    phase_mask_1_padded = np.pad(phase_mask_1, ((Ny//2 - phase_mask_1.shape[0]//2, Ny//2 - phase_mask_1.shape[0]//2),
                                                 (Nx//2 - phase_mask_1.shape[1]//2, Nx//2 - phase_mask_1.shape[1]//2)),
                                 mode='constant', constant_values=0)
    phase_mask_2_padded = np.pad(phase_mask_2, ((Ny//2 - phase_mask_2.shape[0]//2, Ny//2 - phase_mask_2.shape[0]//2),
                                                 (Nx//2 - phase_mask_2.shape[1]//2, Nx//2 - phase_mask_2.shape[1]//2)),
                                 mode='constant', constant_values=0)
    phase_mask_3_padded = np.pad(phase_mask_3, ((Ny//2 - phase_mask_3.shape[0]//2, Ny//2 - phase_mask_3.shape[0]//2),
                                                 (Nx//2 - phase_mask_3.shape[1]//2, Nx//2 - phase_mask_3.shape[1]//2)),
                                 mode='constant', constant_values=0)

    overall_mask = np.roll(phase_mask_1_padded, shift=(center_mask_1_y//unit_cell_pitch, center_mask_1_x//unit_cell_pitch - pixel_shift), axis=(0, 1)) + \
                   np.roll(phase_mask_2_padded, shift=(center_mask_2_y//unit_cell_pitch, center_mask_2_x//unit_cell_pitch - pixel_shift), axis=(0, 1)) + \
                   np.roll(phase_mask_3_padded, shift=(center_mask_3_y//unit_cell_pitch, center_mask_3_x//unit_cell_pitch - pixel_shift), axis=(0, 1))

    overall_mask = (overall_mask + 2*np.pi) % (2*np.pi)
    
    plt.figure(figsize=(6, 5))
    im=plt.pcolormesh(
        X * 1e6, Y * 1e6,
        overall_mask + source_mask + output_mask,
        cmap = 'hsv',
        shading="auto", 
        rasterized=True
    )
    plt.colorbar(im, label="Mask amplitude")
    plt.scatter((center_mask_1_x - pixel_shift*unit_cell_pitch) * 1e6, center_mask_1_y * 1e6, color='white', marker='*', s=70)
    plt.scatter((center_mask_2_x - pixel_shift*unit_cell_pitch) * 1e6, center_mask_2_y * 1e6, color='white', marker='*', s=70)
    plt.scatter((center_mask_3_x - pixel_shift*unit_cell_pitch) * 1e6, center_mask_3_y * 1e6, color='white', marker='*', s=70)
    plt.title("Overall Phase Mask")
    plt.xlabel("X (µm)")
    plt.ylabel("Y (µm)")
    plt.gca().set_aspect("equal")
    plt.savefig(f"{results_folder}/overall_phase_mask.pdf")
    return (overall_mask*phase_mask_region).flatten()

#phase mask setup
w_norm = initialize_phase_mask() / (2 * np.pi)
phase_mask = np.zeros_like(w_norm)

u_k = np.zeros(dof, dtype=np.complex128) #initial guess for the intermediate field
Phi_2D = np.zeros((Ny, Nx), dtype=np.complex128)

#plot the boundary mask
plt.figure(figsize=(6, 5))
plt.pcolormesh(
    X * 1e6, Y * 1e6,
    phase_mask_region,
    shading="auto", rasterized=True
)
plt.title("Boundary Mask")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.gca().set_aspect("equal")
plt.colorbar(label="Mask amplitude")
plt.savefig(f"{results_folder}/boundary_mask.pdf")

#plot both input and output masks
plt.figure(figsize=(6, 5))
plt.pcolormesh(
    X * 1e6, Y * 1e6,
    source_mask,
    shading="auto", rasterized=True
)
plt.title("Input Mask")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.gca().set_aspect("equal")
plt.colorbar(label="Mask amplitude")
plt.savefig(f"{results_folder}/input_mask.pdf")

plt.figure(figsize=(6, 5))
plt.pcolormesh(
    X * 1e6, Y * 1e6,
    output_mask,
    shading="auto", rasterized=True
)
plt.title("Output Mask")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.gca().set_aspect("equal")
plt.colorbar(label="Mask amplitude")
plt.savefig(f"{results_folder}/output_mask.pdf")

#plot the source field
plt.figure(figsize=(6, 5))
plt.pcolormesh(
    X * 1e6, Y * 1e6,
    np.abs(source_field_1) / np.max(np.abs(source_field_1)),
    shading="auto", rasterized=True
)
plt.title("Source Field Amplitude")
plt.xlabel("X (µm)")
plt.ylabel("Y (µm)")
plt.gca().set_aspect("equal")
plt.colorbar(label="Normalized amplitude")
plt.savefig(f"{results_folder}/source_field_amplitude.pdf")

def forward_propagate(field):
    """
    Forward propagate a field using the angular spectrum method.
    Args:
        field : ndarray input field in the spatial domain (flattened).
    Returns:
        prop_field : ndarray Propagated field in the spatial domain (flattened).
    """
    #make sure the input field is 2D
    field_2D = np.reshape(field, (Nx, Ny))  # reshape to 2D array
    field_k = np.fft.fft2(field_2D, norm='ortho')
    field_k_prop = field_k * P_nat
    prop_field = np.fft.ifft2(field_k_prop, norm='ortho').flatten()

    return prop_field

def adjoint_propagate(field):
    """
    Adjoint propagate a field using the angular spectrum method.
    Args:
        field : ndarray input field in the spatial domain (flattened).
    Returns:
        adj_field_prop : ndarray Propagated field in the spatial domain.
    """
    field_2D = np.reshape(field, (Nx, Ny))  # reshape to 2D array
    field_k = np.fft.fft2(field_2D, norm='ortho')
    field_k_prop = field_k * P_adj
    adj_prop_field = np.fft.ifft2(field_k_prop, norm='ortho').flatten()
    return adj_prop_field

def phase_given_w(w_norm):
    return 2 * np.pi * w_norm

def apply_m(u_k):
    '''Applies the M matrix to a given uk vector, M=(Phi_1 U Phi_0 U)u_k, 
    where U=F^-1PF is the ASM propagation matrix
    
    Args:
        u_k : ndarray Input field vector in the spatial domain.
    Returns:
        res4.flatten(): the result after applying the M matrix, flattened to a 1D array.
    '''
    phi_0 = (1 - output_mask).flatten() #2D array
    phi_1 = ((1 - source_mask) * Phi_2D).flatten() #2D array
    res1 = forward_propagate(u_k)
    res2 = res1 * phi_0
    res3 = forward_propagate(res2)
    res4 = res3 * phi_1

    return res4.flatten()

def apply_m_adjoint(u_k):
    '''Applies the adjoint of the M matrix to a given uk vector, M=(Phi_1 U Phi_0 U)^(dagger)u_k, 
    where U=F^-1PF is the ASM propagation matrix
    
    Args:
        u_k : ndarray Input field vector in the spatial domain.
    Returns:
        M_u_k.flatten(): the result after applying the adjoint of the M matrix to vector u_k, flattened to a 1D array.
    '''
    #phi_0 is the phase mask on the output plane
    phi_0_adj = (1 - output_mask).flatten() #2D array
    #phi_1 is the phase mask on the input plane. 
    # It is the top layer that stores the phase mask.
    phi_1_adj = np.conj(((1 - source_mask) * Phi_2D).flatten()) #2D array

    res0 = u_k * phi_1_adj
    res1 = adjoint_propagate(res0)
    res2 = res1 * phi_0_adj
    res3 = adjoint_propagate(res2)
    M_adj_u_k = res3.flatten()
    return M_adj_u_k

def auk(u_k):
    """
    Applies the A matrix to a given uk vector.
    Args:
        u_k : ndarray Input field vector in the spatial domain.
    Returns:
        res : ndarray Result after applying the A matrix.
    """

    res = u_k - apply_m(u_k) 
    return res

def auk_adjoint(u_k):
    """
    Applies the adjoint of the A matrix to a given uk vector.
    Args:
        u_k : ndarray Input field vector in the spatial domain.
    Returns:
        res : ndarray Result after applying the adjoint of the A matrix.
    """
    res = u_k - apply_m_adjoint(u_k)
    return res
    
def gmres_solver():
    A_uk = LinearOperator((dof,dof), matvec=auk, dtype=np.complex128)

    res_u_k, info = gmres(A_uk, tilted_field_1.flatten(), u_k, restart=50, maxiter=30)

    if info != 0:
        print(f"WARNING: GMRES did not converge, info: {info}")

    return res_u_k

def adjoint_gmres_solver(s_adj):
    adj_A_uk = LinearOperator((dof,dof), matvec=auk_adjoint, dtype=np.complex128)
    res_u_k, info = gmres(adj_A_uk, s_adj, restart=50, maxiter=30)

    if info != 0:
        print(f"WARNING: adjoint GMRES did not converge, info: {info}")

    return res_u_k

def cost_fun(x,grad):
    """
    Cost function for the optimization algorithm.
    Args:
        x : ndarray Current guess for the optimization variables.
        grad : ndarray Gradient of the cost function with respect to x.

    Returns:
        overlap: value of the cost function (float)
    """
    w_norm[:] = x  # update the current guess for the optimization variables, in place
    phase_mask[:] = phase_given_w(w_norm)
    Phi_2D[:,:] = np.exp(1j * phase_mask.reshape((Ny, Nx)))  # construct the complex phase mask

    #initiate the GMRES computation to find uk
    print("Starting GMRES solver...")
    u_k[:] = gmres_solver()  # update the intermediate field using GMRES solver
    print("Starting forward propagation...")
    propagated_u_k = forward_propagate(u_k)
    output_field = I_output * propagated_u_k  # compute the output field based on the updated intermediate field

    fig,ax = plt.subplots()
    ax.pcolormesh(X*1e6, Y*1e6, np.abs(propagated_u_k.reshape((Ny, Nx))), shading='auto', rasterized=True)
    ax.set_xlabel("X [um]")
    ax.set_ylabel("Y [um]")
    plt.savefig(f"{results_folder}/output_field.pdf")


    tilde_S = np.sum(output_field * np.conj(target_field_1.flatten()))
    overlap = (np.abs(tilde_S) ** 2)   # compute the overlap with the target field
    print("Overlap = ", overlap)
    
    print("Starting adjoint GMRES solver...")
    s_adj = tilde_S * adjoint_propagate(I_output * target_field_1.flatten())
    
    v = adjoint_gmres_solver(s_adj)
    w = apply_m(u_k)
    #update gradients in place
    if grad.size > 0:
        grad[:] = -2 * np.imag(np.conj(v) * w) * 2 * np.pi

    print("Finished iteration")
    return overlap

if __name__ == "__main__":
    opt_max_eval = 140  #max number of optimization evaluations
    # #setup the optimization algorithm
    solver = nlopt.opt(nlopt.LD_CCSAQ, dof)
    solver.set_lower_bounds(w_norm_lower_boundary.flatten())
    solver.set_upper_bounds(w_norm_upper_boundary.flatten())
    print("shape of boundaries",w_norm_upper_boundary.flatten().shape)
    solver.set_max_objective(cost_fun)
    solver.set_maxeval(opt_max_eval)
    print("w_norm shape", w_norm.shape)
    solver.optimize(w_norm)