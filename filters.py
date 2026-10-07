import autograd.numpy as npa
from numpy.typing import ArrayLike as ArrayLikeType
import matplotlib.pyplot as plt

def _centered(arr: npa.ndarray, newshape: ArrayLikeType) -> npa.ndarray:
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
    newshape = npa.asarray(newshape)
    currshape = npa.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]

def _edge_pad(arr: npa.ndarray, pad: npa.ndarray) -> npa.ndarray:
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
    left = npa.tile(arr[0, :], (pad[0][0], 1))
    right = npa.tile(arr[-1, :], (pad[0][1], 1))
    top = npa.tile(arr[:, 0], (pad[1][0], 1)).transpose()
    bottom = npa.tile(arr[:, -1], (pad[1][1], 1)).transpose()

    # fill corners
    top_left = npa.tile(arr[0, 0], (pad[0][0], pad[1][0]))
    top_right = npa.tile(arr[-1, 0], (pad[0][1], pad[1][0]))
    bottom_left = npa.tile(arr[0, -1], (pad[0][0], pad[1][1]))
    bottom_right = npa.tile(arr[-1, -1], (pad[0][1], pad[1][1]))

    if pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return npa.concatenate(
            (
                npa.concatenate((top_left, top, top_right)),
                npa.concatenate((left, arr, right)),
                npa.concatenate((bottom_left, bottom, bottom_right)),
            ),
            axis=1,
        )
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return npa.concatenate((top, arr, bottom), axis=1)
    elif pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return npa.concatenate((left, arr, right), axis=0)
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return arr
    else:
        raise ValueError("At least one of the padding numbers is invalid.")

def _quarter_to_full_kernel(arr: npa.ndarray, pad_to: npa.ndarray) -> npa.ndarray:
    """Constructs the full kernel from its nonnegative quadrant.

    Args:
        arr: 2d input array representing the nonnegative quadrant of a
            filter kernel with C4v symmetry.
        pad_to: 1d array with two elements (integers) specifying the size
            of the zero padding.

    Returns:
        The complete kernel.
    """
    pad_size = pad_to - 2 * npa.array(arr.shape) + 1

    top = npa.zeros((pad_size[0], arr.shape[1]))
    bottom = npa.zeros((pad_size[0], arr.shape[1] - 1))
    middle = npa.zeros((pad_to[0], pad_size[1]))

    top_left = arr[:, :]
    top_right = npa.flipud(arr[1:, :])
    bottom_left = npa.fliplr(arr[:, 1:])
    bottom_right = npa.flipud(
        npa.fliplr(arr[1:, 1:])
    )  # equivalent to flip, but flip is incompatible with autograd

    return npa.concatenate(
        (
            npa.concatenate((top_left, top, top_right)),
            middle,
            npa.concatenate((bottom_left, bottom, bottom_right)),
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

    #positive quadrant of the filter
    xv = npa.arange(0, Nx//2+1) * (space_grid.get_resolution())
    yv = npa.arange(0, Ny//2+1) * (space_grid.get_resolution())
    #note that the filter's center is at (0,0)
    Xv, Yv = npa.meshgrid(xv, yv)

    h = npa.where(Xv**2 + Yv**2 < radius**2, (1 - npa.sqrt(abs(Xv**2 + Yv**2)) / radius),0)

    h = _quarter_to_full_kernel(h, 3 * npa.array([Nx, Ny]))

    h = h / npa.sum(h)
    #make a plot of the filter kernel
    H = npa.fft.fft2(h)

    phase_mask_2d = phase_mask.reshape(Nx,Ny)
    phase_mask_padded = _edge_pad(phase_mask_2d, ((Nx, Nx), (Ny, Ny)))
    
    phase_mask_fft = npa.fft.fft2(phase_mask_padded)
    
    Y = phase_mask_fft * H
    
    y = _centered(
        npa.real(npa.fft.ifft2(Y)),
        (Nx, Ny)
    )

    return y.flatten()