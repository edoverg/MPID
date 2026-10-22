import autograd.numpy as npa
import numpy as np
from scipy.fft import next_fast_len
from numpy.typing import ArrayLike as ArrayLikeType
import matplotlib.pyplot as plt

class Filter:
    def __init__(self, radius, space_grid):
        self.radius = radius
        self.space_grid = space_grid
        self.p = int(np.ceil(radius / self.space_grid.get_resolution()))
        self.Fx = next_fast_len(self.space_grid.get_Nx() + self.p)
        self.Fy = next_fast_len(self.space_grid.get_Ny() + self.p)
        self.H = self._construct_kernel(radius, space_grid)
        
    def apply_filter(self, input_parameters):
        input_parameters_2d = input_parameters.reshape(self.space_grid.get_Nx(), self.space_grid.get_Ny())
        #input_parameters_padded = self._edge_pad(input_parameters_2d, (((self.Fx-1)//2, (self.Fx)//2), ((self.Fy-1)//2, (self.Fy)//2)))
        input_parameters_padded = self._edge_pad(input_parameters_2d, ((0,0), (0,0)))
        #print("input_parameters_padded shape:",input_parameters_padded.shape)
        input_parameters_fft = npa.fft.fft2(input_parameters_padded)
        
        Y = input_parameters_fft * self.H
        y = self._centered(
            npa.real(npa.fft.ifft2(Y)),
            (self.space_grid.get_Nx(), self.space_grid.get_Ny())
        )
        return y.flatten()

    def _centered(self, arr: npa.ndarray, newshape: ArrayLikeType) -> npa.ndarray:
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

    def _edge_pad(self, arr: npa.ndarray, pad: npa.ndarray) -> npa.ndarray:
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

    def _quarter_to_full_kernel(self, arr: npa.ndarray, pad_to: npa.ndarray) -> npa.ndarray:
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

    def _construct_kernel(self,radius,space_grid):
        '''Applies a conic filter to the input parameters. The filter is defined as a 
        conic region of a specified radius, where the values undergo a weighted average. 

        Args:
            input_parameters: the input parameters to be filtered (2D)
            radius: the radius of the conic filter in micrometers
            space_grid: a SpaceGrid object containing the size and resolution of the parameters

        Returns:
            The filtered parameters (2D)
        '''
        Nx = space_grid.get_Nx()
        Ny = space_grid.get_Ny()

        #positive quadrant of the filter
        xv = npa.arange(0, Nx//2+1) * (space_grid.get_resolution())
        yv = npa.arange(0, Ny//2+1) * (space_grid.get_resolution())
        #note that the filter's center is at (0,0)
        Xv, Yv = npa.meshgrid(xv, yv)

        h = npa.where(Xv**2 + Yv**2 < radius**2, (1 - npa.sqrt(abs(Xv**2 + Yv**2)) / radius),0)

        #22/10/2026:
        # Necessity: computational efficiency of the filter; reduction of computational time
        # Issues: applying edge-padding/zero-padding increases the computational load.
        # This is associated to the FFT calculation. 
        # Solution: edge-padding is necessary is the signal extends up to the border. In that 
        # condition, given that FFT treats the signal as periodic, edge-padding ensures that the filter 
        # does not introduce artifacts at the boundaries. In our case, the signal is represented by the 
        # masks, which inherently have finite support and are zero padded by default. 
        # Therefore, edge-padding is not strictly necessary in this particular case.
        # This might need to be reconsidered if the signal extends close to the boundaries or if 
        # the filter radius is large compared to the signal support.
        h = self._quarter_to_full_kernel(h, npa.array([Nx, Ny]))
        
        print("Fx, Fy:", self.Fx, self.Fy)

        h = h / npa.sum(h)
        H = npa.fft.fft2(h)
        
        return H