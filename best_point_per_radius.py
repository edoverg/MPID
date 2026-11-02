'''
For each conic filter radius, takes the best propagation distances found by the EGO run
(matern32_surrogate_filtered_<radius>/ego_opt_results.txt), runs the mask optimization
at 1.50/1.55/1.60 um (as in single_converter_symm.py) and then evaluates the resulting
masks with a source sweep at 5 wavelengths (as in single_converter_symm_source_sweep.py).
Results of each (radius, strategy) pair go to results_radius_<radius>_<strategy>/.
'''
import os
import mpid as pid
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.rcParams['mathtext.default'] = 'regular'
from scipy.interpolate import CubicSpline

N = 1025
unit_cell_pitch = 0.650  # [um]
mat_refr_index = np.array([1.444, 1.0, 1.444])
ms_max_available_diameter = 122  # [um]
output_fiber_core_diameter = 19  # [um]
opt_max_eval = 250

opt_wavelengths_um = [1.5, 1.55, 1.6]
sweep_wavelengths_um = [1.5, 1.53, 1.55, 1.57, 1.6]


def radius_tag(radius: float) -> str:
    if radius == 0:
        return "0"
    else:
        return f"{radius:.1f}".replace('.', '_')


def results_folder_name(radius: float, strategy: str) -> str:
    return f"results_radius_{radius_tag(radius)}_{strategy}"


def load_best_distances(radius: float, strategy: str) -> np.ndarray:
    '''
    Reads d0, d1 from the EGO results (rows: d0, d1, best value) and returns [d0, d1, d2=d0].
    EGO simulates distances rounded to 0 decimals, so the same rounding is applied here.
    '''
    d0, d1, _ = np.loadtxt(f"matern32_surrogate_filtered_{radius_tag(radius)}_{strategy}/ego_opt_results.txt")
    d0, d1 = np.round(d0, 0), np.round(d1, 0)
    return np.array([d0, d1, d0])


def build_simulation(wavelengths_um: list, distances: np.ndarray, radius: float, strategy: str,
                     results_folder: str, optimize: bool) -> pid.Simulation:
    '''Builds the mode converter simulation for the given wavelengths, distances [um] and filter radius.'''
    lda_nm = [int(round(w * 1000)) for w in wavelengths_um]
    sources = [pid.Source(input_field=np.load(f"fiber_modes/SMF/SMF_LP01_field_distribution_lda_{l}_N_1025.npy"),
                          wavelength=w, field_norm=True) for l, w in zip(lda_nm, wavelengths_um)]
    targets = [pid.Target(input_field=np.load(f"fiber_modes/FMF/FMF_LP21_field_distribution_lda_{l}_N_1025.npy"),
                          associated_source=s, wavelength=w, field_norm=True)
               for l, w, s in zip(lda_nm, wavelengths_um, sources)]

    unit_cells = []
    for w in wavelengths_um:
        tag = str(w).replace('.', '_')
        unit_cells.append(pid.UnitCell(
            wavelength=w,
            tune_param=np.load(f"unit_cell/final/export/unit_cell_diameter_P_0_65_h_0_9_{tag}.npy"),
            sim_phase=np.load(f"unit_cell/final/export/unit_cell_phase_P_0_65_h_0_9_{tag}.npy")))

    space_grid = pid.SpaceGrid(
        side_pixels=N,
        unit_cell_pitch=unit_cell_pitch,
        prop_distances=distances,
        materials_refr_index=mat_refr_index,
        xy_symmetry=True,
    )
    simulation = pid.Simulation(
        sources=sources,
        targets=targets,
        central_wavelength=wavelengths_um[0],
        space_grid=space_grid,
        n_cpu=1
    )
    os.makedirs(results_folder, exist_ok=True)
    simulation.results_folder = results_folder
    if radius > 0:
        simulation.set_optimization(
            opt_param_min=0,
            opt_param_max=1,
            ms_max_avail_diameter=ms_max_available_diameter,
            phase_mask_init_size=output_fiber_core_diameter / 3,
            unit_cells=unit_cells,
            opt_max_eval=opt_max_eval,
            conic_filter_radius=radius,
            strategy=strategy,
            results_while_solving=optimize,
            save_results=optimize,
        )
    else:
        simulation.set_optimization(
            opt_param_min=0,
            opt_param_max=1,
            ms_max_avail_diameter=ms_max_available_diameter,
            phase_mask_init_size=output_fiber_core_diameter / 3,
            unit_cells=unit_cells,
            opt_max_eval=opt_max_eval,
            strategy=strategy,
            results_while_solving=optimize,
            save_results=optimize,
        )
    return simulation


def run_optimization(distances: np.ndarray, radius: float, strategy: str, results_folder: str = None):
    '''
    Optimizes the masks for fixed propagation distances [um] (d0, d1, d2), filter radius and strategy.
    Masks are saved in `results_folder`. Returns the best optimization value.
    '''
    results_folder = results_folder or results_folder_name(radius, strategy)
    simulation = build_simulation(opt_wavelengths_um, np.asarray(distances, dtype=float), radius,
                                  strategy, results_folder, optimize=True)
    simulation.run()
    best_opt_val, d = simulation.get_best_opt_and_distances()
    print(f"[radius {radius}, {strategy}] best optimization value: {best_opt_val}, distances: {d}")
    return best_opt_val


def _zoom(ax):
    lim = output_fiber_core_diameter / 2 + 10
    ax.set_xlim(-lim, lim)


def run_sweep(distances: np.ndarray, radius: float, strategy: str, results_folder: str = None):
    '''
    Loads the masks saved by run_optimization, evaluates them at the sweep wavelengths and
    saves field plots, comparison plots and the mode purity curve in `results_folder`.
    '''
    results_folder = results_folder or results_folder_name(radius, strategy)
    simulation = build_simulation(sweep_wavelengths_um, np.asarray(distances, dtype=float), radius,
                                  strategy, results_folder, optimize=False)
    mask1 = np.load(f"{results_folder}/mask_1.npy").flatten()
    mask2 = np.load(f"{results_folder}/mask_2.npy").flatten()
    # saved masks are already filtered
    results = simulation.source_sweep(distances=np.asarray(distances, dtype=float),
                                      input_diameter_masks=[mask1, mask2],
                                      save_output=True, apply_filtering=False)

    X = simulation.space_grid.get_X() * 1e6
    Y = simulation.space_grid.get_Y() * 1e6
    lim = output_fiber_core_diameter / 2 + 10
    core = lambda: plt.Circle((0, 0), output_fiber_core_diameter / 2, color='red', fill=False, linewidth=2)

    def style(ax):
        ax.tick_params(axis='both', which='major', labelsize=20)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.array(list(results.keys())) * 1e9, np.array(list(results.values())), marker='+', linewidth=2)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.set_xlabel('wavelength [nm]', fontsize=20)
    ax.set_ylabel('ob. fun.', fontsize=20)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{results_folder}/source_sweep_results.pdf")
    plt.close(fig)

    modal_purity_list = []
    for wvl, target_file_nm in zip(sweep_wavelengths_um, [int(round(w * 1000)) for w in sweep_wavelengths_um]):
        save_filename = f"{results_folder}/output_field_{np.round(wvl, 2)}".replace('.', '_')
        output_field = np.load(save_filename + ".npy").reshape(simulation.space_grid.get_Nx(),
                                                                simulation.space_grid.get_Ny())
        target_field = np.load(f"fiber_modes/FMF/FMF_LP21_field_distribution_lda_{target_file_nm}_N_1025.npy")
        amp = np.abs(output_field) / np.max(np.abs(output_field))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pcolormesh(X, Y, amp, cmap='viridis', rasterized=True)
        fig.colorbar(ax.collections[0], ax=ax, label='Magnitude [a.u.]')
        ax.set_aspect('equal'); _zoom(ax); ax.set_ylim(-lim, lim); ax.add_artist(core())
        ax.set_xlabel('X [um]', fontsize=20); ax.set_ylabel('Y [um]', fontsize=20); style(ax)
        ax.set_title(r'Output Field at $\lambda=$' + str(np.round(wvl, 2)) + r' $\mu$m')
        fig.tight_layout(); fig.savefig(save_filename + ".pdf"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pcolormesh(X, Y, (np.angle(output_field) + 2 * np.pi) % (2 * np.pi), cmap='hsv',
                      vmin=0, vmax=2 * np.pi, rasterized=True)
        fig.colorbar(ax.collections[0], ax=ax, label='Phase [rad]')
        ax.set_aspect('equal'); _zoom(ax); ax.set_ylim(-lim, lim); ax.add_artist(core())
        ax.set_xlabel('X [um]', fontsize=20); ax.set_ylabel('Y [um]', fontsize=20); style(ax)
        ax.set_title(r'Output Field Phase at $\lambda=$' + str(wvl) + r' $\mu$m')
        fig.tight_layout(); fig.savefig(save_filename + "_phase.pdf"); plt.close(fig)

        overlap = np.abs(np.sum(output_field * np.conj(target_field))) ** 2
        modal_purity = overlap / (np.sum(np.abs(output_field) ** 2) * np.sum(np.abs(target_field) ** 2))
        modal_purity_list.append(modal_purity)
        print(f"[radius {radius}, {strategy}] lambda={wvl} um: modal purity={modal_purity}, "
              f"original cost function={overlap / np.sum(np.abs(target_field) ** 2)}")

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        im = axs[0].pcolormesh(X, Y, amp, cmap='viridis', shading='auto', rasterized=True)
        fig.colorbar(im, ax=axs[0], label='Magnitude [a.u.]')
        axs[0].set_aspect('equal'); _zoom(axs[0]); axs[0].set_ylim(-lim, lim); axs[0].add_artist(core())
        axs[0].set_xlabel('X [um]', fontsize=20); axs[0].set_ylabel('Y [um]', fontsize=20)
        axs[0].set_title('Output Field Amplitude'); style(axs[0])
        for k, (ax, coord, out_s, tgt_s, lab) in enumerate([
                (axs[1], Y[:, 0], output_field[:, output_field.shape[1] // 2],
                 target_field[:, target_field.shape[1] // 2], 'Y'),
                (axs[2], X[0, :], output_field[output_field.shape[0] // 2, :],
                 target_field[target_field.shape[0] // 2, :], 'X')]):
            ax.plot(coord, np.abs(out_s) / np.max(np.abs(out_s)), label='Output', c='blue', linewidth=2)
            ax.plot(coord, np.abs(tgt_s) / np.max(np.abs(tgt_s)), label='Target', c='red', linewidth=2)
            ax.set_xlabel(f'{lab} [um]', fontsize=20); ax.set_ylabel('Magnitude [a.u.]', fontsize=20)
            ax.set_title(f'{lab} Cross Section'); ax.set_ylim(0, 1); _zoom(ax)
            ax.legend(fontsize=20); style(ax)
        fig.delaxes(axs[3])
        fig.tight_layout(); fig.savefig(save_filename + "_comparison.pdf"); plt.close(fig)

    wavelengths_nm = np.array(list(results.keys())) * 1e9
    #modal_purity_dB = 10 * np.log10(np.array(modal_purity_list))
    np.save(f"{results_folder}/modal_purity.npy", np.column_stack([wavelengths_nm, modal_purity_list]))
    cs = CubicSpline(wavelengths_nm, modal_purity_list)
    fine = np.linspace(wavelengths_nm.min(), wavelengths_nm.max(), 500)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(wavelengths_nm, modal_purity_list, c='r', marker='o', s=50)
    ax.plot(fine, cs(fine), c='b', linewidth=2.5)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.set_xlabel('wavelength [nm]', fontsize=20)
    ax.set_ylabel('mode purity [a.u.]', fontsize=20)
    ax.set_ylim(np.min(modal_purity_list) - 0.1, np.max(modal_purity_list) + 0.1)
    style(ax)
    fig.tight_layout(); fig.savefig(f"{results_folder}/mode_purity.pdf"); plt.close(fig)
    return results


def plot_purity_vs_radius(radii: list, save_name: str = None):
    '''
    Plots the mode purity [a.u.] against wavelength [nm], one trace per filter radius,
    reading the modal_purity.npy files written by run_sweep.
    '''
    fig, ax = plt.subplots()
    strategy = ['epigraph']
    radius_colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']
    for s in strategy:
        for radius in radii:
            #in the same plot, report the mode purity for each radius and strategy. Use same line color for equal radius, continuous lines for epigraph and dashed for mean
            data = np.load(f"{results_folder_name(radius, s)}/modal_purity.npy")
            wl, purity = data[:, 0], data[:, 1]
            fine = np.linspace(wl.min(), wl.max(), 500)
            line, = ax.plot(fine, CubicSpline(wl, purity)(fine), linewidth=2.5, label=f'{radius:g}', linestyle='-' if s == 'epigraph' else '--', color=radius_colors[radii.index(radius)])
            ax.scatter(wl, purity, color=line.get_color(), s=50, zorder=3)

    #add a text box with the radii below the maximum value of the curve and aligned on the left of the point, except for the 1.5 radii which will be above
    #and of the same color as the corresponding curve and with text aligned on left of the point
    for radius in radii:
        data = np.load(f"{results_folder_name(radius, 'epigraph')}/modal_purity.npy")
        wl, purity = data[:, 0], data[:, 1]
        if radius == 1.5:
            ax.text(wl[np.argmax(purity)]-5, np.max(purity) + 0.01, f'R = {radius:g}', fontsize=20, verticalalignment='bottom', horizontalalignment='left', color=radius_colors[radii.index(radius)])
        else:
            ax.text(wl[np.argmax(purity)]-5, np.max(purity) - 0.01, f'R = {radius:g}', fontsize=20, verticalalignment='top', horizontalalignment='left', color=radius_colors[radii.index(radius)])
    
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.set_xlabel('wavelength [nm]', fontsize=20)
    ax.set_ylabel('mode purity [a.u.]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(save_name or f"mode_purity_vs_radius.pdf")
    plt.close(fig)


if __name__ == "__main__":
    radii = [3.0, 7.0]
    strategy = 'mean'
    for radius in radii:
        d = load_best_distances(radius=radius, strategy=strategy)
        print(f"=== radius {radius}: distances {d} ===")
        run_optimization(d, radius, strategy)
        run_sweep(d, radius, strategy)
    plot_purity_vs_radius(radii)
