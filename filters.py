import autograd.numpy as np
from numpy.typing import ArrayLike as ArrayLikeType
import matplotlib.pyplot as plt

filter_radius_um = 1
def _centered(arr: np.ndarray, newshape: ArrayLikeType) -> np.ndarray:
    """Formats the output of an FFT to center the zero-frequency component.

    A helper function borrowed from SciPy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270

    Args:
        arr: output array from an FFT operation.
        newshape: 1d array with two elements (integers) specifying the dimensions
            of the array to be returned.

    Returns:
        The input array with the zero-frequency component as the central element.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]

def _edge_pad(arr: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Border-pads the edges of an array.

    Used to preprocess the design weights prior to convolution with the filter.
    Border padding an image will set the value of each padded pixel equal to
    the value of the nearest pixel in the image. Used to implement feature-
    preserving convolution filters that prevent unwanted edge effects.

    Args:
        arr: 2d array whose borders contain the values to use for padding
        pad: 2x2 array of integers indicating the size
            of the borders to pad the array with.

    Returns:
        A 2d array with border padding.
    """
    # fill sides
    left = np.tile(arr[0, :], (pad[0][0], 1))
    right = np.tile(arr[-1, :], (pad[0][1], 1))
    top = np.tile(arr[:, 0], (pad[1][0], 1)).transpose()
    bottom = np.tile(arr[:, -1], (pad[1][1], 1)).transpose()

    # fill corners
    top_left = np.tile(arr[0, 0], (pad[0][0], pad[1][0]))
    top_right = np.tile(arr[-1, 0], (pad[0][1], pad[1][0]))
    bottom_left = np.tile(arr[0, -1], (pad[0][0], pad[1][1]))
    bottom_right = np.tile(arr[-1, -1], (pad[0][1], pad[1][1]))

    if pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return np.concatenate(
            (
                np.concatenate((top_left, top, top_right)),
                np.concatenate((left, arr, right)),
                np.concatenate((bottom_left, bottom, bottom_right)),
            ),
            axis=1,
        )
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return np.concatenate((top, arr, bottom), axis=1)
    elif pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return np.concatenate((left, arr, right), axis=0)
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return arr
    else:
        raise ValueError("At least one of the padding numbers is invalid.")

def _quarter_to_full_kernel(arr: np.ndarray, pad_to: np.ndarray) -> np.ndarray:
    """Constructs the full kernel from its nonnegative quadrant.

    Args:
        arr: 2d input array representing the nonnegative quadrant of a
            filter kernel with C4v symmetry.
        pad_to: 1d array with two elements (integers) specifying the size
            of the zero padding.

    Returns:
        The complete kernel.
    """
    pad_size = pad_to - 2 * np.array(arr.shape) + 1

    top = np.zeros((pad_size[0], arr.shape[1]))
    bottom = np.zeros((pad_size[0], arr.shape[1] - 1))
    middle = np.zeros((pad_to[0], pad_size[1]))

    top_left = arr[:, :]
    top_right = np.flipud(arr[1:, :])
    bottom_left = np.fliplr(arr[:, 1:])
    bottom_right = np.flipud(
        np.fliplr(arr[1:, 1:])
    )  # equivalent to flip, but flip is incompatible with autograd

    return np.concatenate(
        (
            np.concatenate((top_left, top, top_right)),
            middle,
            np.concatenate((bottom_left, bottom, bottom_right)),
        ),
        axis=1,
    )

def conic_filter(phase_mask,radius,space_grid):
    '''Applies a conic filter to the input phase mask. The filter is defined as a 
    conic region of a specified radius, where the values are averaged. 

    Args:
        phase_mask: the input phase mask to be filtered (2D)
        radius: the radius of the conic filter in micrometers
        space_grid: an object containing the size and resolution of the phase mask
    Returns:
        The filtered phase mask (2D)
    '''

    Nx = space_grid.get_Nx()
    Ny = space_grid.get_Ny()

    xv = np.arange(0, Nx//2-1) * (space_grid.get_resolution())
    yv = np.arange(0, Ny//2-1) * (space_grid.get_resolution())
    Xv, Yv = np.meshgrid(xv, yv)
    
    h = np.where(Xv**2 + Yv**2 < radius**2, (1 - np.sqrt(abs(Xv**2 + Yv**2)) / radius),0)

    h = _quarter_to_full_kernel(h, 3 * np.array([Nx, Ny]))

    h = h / np.sum(h)
    #make a plot of the filter kernel
    H = np.fft.fft2(h)

    phase_mask_2d = phase_mask.reshape(Nx,Ny)
    phase_mask_padded = _edge_pad(phase_mask_2d, ((Nx, Nx), (Ny, Ny)))
    
    phase_mask_fft = np.fft.fft2(phase_mask_padded)
    
    Y = phase_mask_fft * H
    
    y = _centered(
        np.real(np.fft.ifft2(Y)),
        (Nx, Ny)
    )

    return y.flatten()