#use this file to import some test phase masks and observe 
#the effect of changing the filtering radius
import numpy as np
import matplotlib.pyplot as plt
from autograd import tensor_jacobian_product

class TempGrid:
    def __init__(self, Nx, Ny, res): self.Nx, self.Ny, self.res = Nx, Ny, res
    def get_Nx(self): return self.Nx
    def get_Ny(self): return self.Ny
    def get_resolution(self): return self.res


grid = TempGrid(Nx=1025, Ny=1025, res=0.7)
radius_to_test = [2.0, 5.0, 11.0]

input_mask = np.load('results/mask_2.npy')
#make a plot of the input mask
# fig, ax = plt.subplots()
# ax.imshow(input_mask, cmap='inferno')
# #zoom in the center region
# center_x, center_y = grid.get_Nx() // 2, grid.get_Ny() // 2
# zoom_size = 100  # adjust as needed
# ax.set_xlim(center_x - zoom_size, center_x + zoom_size)
# ax.set_ylim(center_y - zoom_size, center_y + zoom_size)
# plt.savefig('input_mask.pdf')

#apply the filter
from filters import Filter

center_x, center_y = grid.get_Nx() // 2, grid.get_Ny() // 2
zoom_size = 80

colormap = 'gray_r'
fig, ax = plt.subplots(1,4, figsize=(12, 3))
ax[0].imshow(input_mask, cmap=colormap) 
ax[0].set_title('Input', fontsize=20)
ax[0].set_xlim(center_x - zoom_size, center_x + zoom_size)
ax[0].set_ylim(center_y - zoom_size, center_y + zoom_size)
ax[0].set_xticks([])
ax[0].set_yticks([])
for i, radius in enumerate(radius_to_test, start=1):
    filt = Filter(radius, grid)
    output_mask = filt.apply_filter(input_mask)
    ax[i].imshow(output_mask.reshape(grid.get_Nx(), grid.get_Ny()), cmap=colormap)    
    ax[i].set_title(f'r={radius} \u03BCm', fontsize=20)
    ax[i].set_xlim(center_x - zoom_size, center_x + zoom_size)
    ax[i].set_ylim(center_y - zoom_size, center_y + zoom_size)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
plt.tight_layout()
plt.savefig('filtered_masks.pdf')



