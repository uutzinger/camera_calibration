import numpy as np
from numba import cuda, vectorize, jit
from timeit import default_timer as timer

# Numba CUDA compiled
@cuda.jit(device=True)
def calibrate_gpu(img, ff, bg):
    return (img - bg) * ff

# Custom CUDA kernel
# The hard way: Check mandelbrot example:
# https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb
#
# blockDim.x,y,z gives number of threads in a block, in the particular direction
# gridDim.x,y.z  gives the number blocks in a grid, in the particular direction
# blockDim.x * gridDim.x gives the number of trheads in a grid in the x direction
# blockIdx  is the block  index in the grid
# threadIdx is the thread index within the block
@cuda.jit
def calibrate_kernel(data_cube,background_indx,flatfield, result):
    depth  = data_cube.shape[0] 
    height = data_cube.shape[1]
    width  = data_cube.shape[2]

    startZ, startX, startY = cuda.grid(3)
    #startZ = cuda.blockDim.z * cuda.blockIdx.z + cuda.threadIdx.z
    #startX = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    #startY = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    for z in range (startZ,depth,gridZ):
        for x in range(startX, height, gridX):
            for y in range(startY, width, gridY):
                I  = data_cube[z,x,y]
                bg = data_cube[background_indx, x, y]
                ff = flatfield[x,y]
                result[z, x, y] = calibrate_gpu(I, ff, bg)


####################################################################################################################
# Variables
####################################################################################################################

# Simulated image data, we have 14 images in one cube, we have a flatfield,
# we search in image cube for image with lowest intensity as that is the background
data_cube      = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
background     = (np.random.randint(0, 255,     (540, 720), 'uint8'))  # where we keep bg on CPU
flatfield      = np.cast['uint16'](2**8.*np.random.random((540, 720))) # flatfield correction scaled so that 255=100%
inten          = np.zeros(14, 'uint16')                                # helper to find background image
data_cube_corr = np.zeros((14, 540, 720), 'uint16')                    # resulting data cube on CPU

flatfield_cuda             = cuda.to_device(flatfield)                 # flatfield on GPU
data_cube_cuda             = cuda.to_device(data_cube)                 # data on GPU
data_cube_corr_cuda        = cuda.device_array(shape=data_cube.shape, dtype='uint16') # where results are stored on GPU
data_cube_corr_kernel_cuda = cuda.device_array(shape=data_cube.shape, dtype='uint16') # where results are stored on GPU with custom kernel

####################################################################################################################
# CUDA Custom Cuda Kernel
####################################################################################################################

# GPU block and thread allocation
blockdim = (2, 4, 4) # how to figure these out?
griddim  = (14, 8, 8) # how to figure these out?


# Send constant data to GPU
flatfield_cuda              = cuda.to_device(flatfield)                # remains the same

# Run In
background_indx = 3
data_cube_cuda       = cuda.to_device(data_cube)                       # send new data
background_indx_cuda = cuda.to_device(background_indx)                 # send background location
calibrate_kernel[griddim, blockdim](data_cube_cuda, background_indx_cuda, flatfield_cuda, data_cube_corr_kernel_cuda) 
#cuda.synchronize()                                                    # 
data_cube_corr = data_cube_corr_kernel_cuda.copy_to_host()             # obtain results back

# Measure
n_time    = 0.
i_time    = 0.
to_time   = 0.
from_time = 0.

for i in range(1000):
    # Find background
    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten)                                 # search for minimum intensity 
    end_time = timer()
    i_time = i_time + (end_time - start)

    # Send to GPU
    start = timer()
    data_cube_cuda       = cuda.to_device(data_cube)                   # send new data
    background_indx_cuda = cuda.to_device(background_indx)             # send background location
    end_time = timer()
    to_time = to_time + (end_time - start)

    # Execute Calibration
    start = timer()
    calibrate_kernel[griddim, blockdim](data_cube_cuda, background_indx_cuda, flatfield_cuda, data_cube_corr_kernel_cuda) 
    cuda.synchronize()                                                 # 
    end_time = timer()
    n_time = n_time + (end_time - start)

    # Retrieve from GPU
    start = timer()
    #cuda.synchronize()                                                #  
    data_cube_corr = data_cube_corr_kernel_cuda.copy_to_host()         # obtain results back
    from_time = from_time + (timer() - start)

print('=================================================================')
print('Find bg execution time is                 : {:8.2f}ms'.format(i_time   ))  
print('Sending data to GPU with Numba takes      : {:8.2f}ms'.format(to_time  )) 
print('CUDA Kernel execution time is             : {:8.2f}ms'.format(n_time   ))  
print('Retrieving data from GPU with Numba takes : {:8.2f}ms'.format(from_time)) 

# blockdim 1,  4,  4 griddim  4, 32, 32:  5.01 ms
# blockdim 1, 32,  8 griddim  1, 32, 16:  4.66 ms
# blockdim 2, 32,  8 griddim  2, 32, 16:  4.00 ms
# blockdim 2,  4,  4 griddim  4, 16, 16:  2.48 ms
# blockdim 1,  2,  2 griddim  1,  4,  4: 32.64 ms
# blockdim 1,  4,  4 griddim  1,  8,  8:  7.39 ms
# blockdim 1,  2,  2 griddim  1,  8,  8: 11.70ms
# blockdim 1,  2,  2 griddim  1, 16, 16:  7.71ms
# blockdim 1,  4,  4 griddim  1, 16, 16:  4.15ms
# blockdim 1,  4,  4 griddim  2, 16, 16:  4.35ms
# blockdim 1,  4,  4 griddim  4, 16, 16:  3.60ms
# blockdim 2,  4,  4 griddim  4, 16, 16:  2.49ms
# blockdim 4,  4,  4 griddim  4, 16, 16:  3.36ms
# blockdim 2,  4,  4 griddim  8, 16, 16:  2.18ms
# blockdim 4,  4,  4 griddim  8, 16, 16:  2.14ms
# blockdim 4,  4,  4 griddim  8, 32, 32:  2.96ms
# blockdim 4,  4,  4 griddim  14, 16, 16: 1.67ms
# blockdim 1,  4,  4 griddim  14, 16, 16: 3.14ms
# blockdim 2,  4,  4 griddim  14, 16, 16: 1.60ms
# blockdim 2,  8,  8 griddim  14, 16, 16: 2.99ms
# blockdim 2,  4,  4 griddim  14, 32, 32: 2.78ms
# blockdim 2,  4,  4 griddim  14, 16, 32: 2.64ms
# blockdim 2,  4,  4 griddim  14, 32, 16: 1.93ms
# blockdim 2,  4,  4 griddim  14, 16, 16: 1.70ms
# blockdim 2,  4,  4 griddim  14, 8, 16:  1.65ms
# blockdim 2,  4,  4 griddim  14, 8, 8:   1.29ms
# blockdim 2,  4,  4 griddim  14, 4, 4:   1.44ms
# blockdim 2,  8,  8 griddim  14, 8, 8:   2.01ms
# blockdim 1,  4,  4 griddim  14, 8, 8:   1.98ms
# blockdim 2,  4,  4 griddim  14, 8, 8:   1.25ms

