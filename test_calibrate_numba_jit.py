import numpy as np
import cupy  as cp
from numba import cuda, vectorize, jit
from timeit import default_timer as timer
from cupyx.time import repeat

# basic function
@cuda.jit(device=True)
def calibrate_gpu(img, ff, bg):
    return (img - bg) * ff

@jit
def calibrate_cpu(img, ff, bg):
    return (img - bg) * ff

@vectorize(['uint16(uint8, uint8, uint16)'], target='cpu')
def vector_cpu(data_cube,background,flatfield):
    return calibrate_cpu(data_cube, background, flatfield) 

@vectorize(['uint16(uint8, uint8, uint16)'], target='parallel')
def vector_parallel(data_cube,background,flatfield):
    return calibrate_cpu(data_cube, background, flatfield) 

@vectorize(['uint16(uint8, uint8, uint16)'], target='cuda')
def vector_gpu(data_cube,background,flatfield):
    return calibrate_gpu(data_cube, background, flatfield)

@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def vetor_np(data_cube, flatfield, bg):
    return np.multiply(np.subtract(data_cube, bg), flatfield) # 16bit multiplication

# The CUDA kernel
# The hard way
#
# Check mandelbrot example:
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
        for x in range(startX, width, gridX):
            for y in range(startY, height, gridY):
                I  = data_cube[z,x,y]
                bg = data_cube[background_indx, x, y]
                ff = flatfield[x,y]
                result[z, y, x] = calibrate_gpu(I, ff, bg)


def calibrate_cp(data_cube, flatfield, bg, result):
    cp.multiply(cp.subtract(data_cube, bg), flatfield, out=result)

@vectorize(['uint16(uint8, uint8, uint16, uint16)'], target='cuda')
def vector_cp(data_cube, background, flatfield, result):
    calibrate_cp(data_cube, background, flatfield, result)
    return result

####################################################################################################################
# Variable
####################################################################################################################

# Simulated image data, we have 14 images
data_cube      = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
background     = (np.random.randint(0, 255,     (540, 720), 'uint8'))  # where we keep bg
flatfield      = np.cast['uint16'](2**8.*np.random.random((540, 720))) # we can scale flatfield so that 255=100%
inten          = np.zeros(14, 'uint16')                                # help to find background image
data_cube_corr = np.zeros((14, 540, 720), 'uint16')                    # result

flatfield_cuda             = cuda.to_device(flatfield)                 # remains the same
data_cube_cuda             = cuda.to_device(data_cube)                 # send new data
data_cube_corr_cuda        = cuda.device_array(shape=data_cube.shape, dtype='uint16') # where results are stored on GPU
data_cube_corr_kernel_cuda = cuda.device_array(shape=data_cube.shape, dtype='uint16') # where results are stored on GPU

data_cube_cp               = cp.array(data_cube, copy=True)
flatfield_cp               = cp.array(flatfield, copy=True)
data_cube_corr_cp          = cp.empty_like(data_cube_cp)
background_cp              = cp.array(data_cube[0,:,:])

####################################################################################################################
# CPU Numpy, not accelerated
####################################################################################################################

# Measure
n_time    = 0.
i_time    = 0.

for i in range(1000):

    # Find background
    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten) # search for minimum intensity 
    background = data_cube[background_indx, :, :]
    end_time = timer()
    i_time = i_time + (end_time - start)

    start = timer()
    data_cube_corr = vector_cpu(data_cube, background, flatfield) 
    # Subtract background from images and mutiply flatfield
    for frame_idx in range(0,14):
        if frame_idx != frame_idx_bg:
            _ = np.multiply(np.subtract(data_cube[frame_idx, :, :], data_cube[background_indx, :, :]), flatfield, out = data_cube_corr[frame_idx, :, :]) 
    end_time = timer()
    n_time = n_time + (end_time - start)

print('Find bg execution time is        : {}'.format(i_time / 1000.))  
print('Numpy execution time is          : {}'.format(n_time / 1000.))  

####################################################################################################################
# CPU Numba Numpy Vectorized
####################################################################################################################

n_time    = 0.
i_time    = 0.

# Run in
data_cube_corr = vector_np(data_cube, flatfield, background)

for i in range(1000):

    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    # minimum intensity is at which frame  index
    frame_idx_bg = np.argmin(inten)
    background = data_cube[frame_idx_bg, :, :]
    end_time = timer()
    i_time = i_time + (end_time - start)

    start = timer()
    vector_np(data_cube, flatfield, background, out = data_cube_corr)
    end_time = timer()
    n_time  = n_time + (end_time - start)
    
print('Find bg execution time is        : {}'.format(i_time / 1000.))  
print('Numba Numpy execution time is    : {}'.format(n_time / 1000.))  

####################################################################################################################
# CPU Numba Vectorized
####################################################################################################################

# Measure
n_time    = 0.
i_time    = 0.

# run in
data_cube_corr = vector_cpu(data_cube, background, flatfield) 

for i in range(1000):
    # Find background
    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten) # search for minimum intensity 
    background = data_cube[background_indx, :, :]
    end_time = timer()
    i_time = i_time + (end_time - start)
    start = timer()
    data_cube_corr = vector_cpu(data_cube, background, flatfield) 
    end_time = timer()
    n_time = n_time + (end_time - start)

print('Find bg execution time is        : {}'.format(i_time / 1000.))  
print('Numba CPU execution time is      : {}'.format(n_time / 1000.))  

####################################################################################################################
# CPU Parallel Vectorized
####################################################################################################################

# Measure
n_time    = 0.
i_time    = 0.

# run in
data_cube_corr = vector_parallel(data_cube, background, flatfield) 

for i in range(1000):
    # Find background
    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten)                                 # search for minimum intensity 
    background = data_cube[background_indx, :, :]
    end_time = timer()
    i_time = i_time + (end_time - start)
    start = timer()
    data_cube_corr = vector_parallel(data_cube, background, flatfield) 
    end_time = timer()
    n_time = n_time + (end_time - start)

print('Find bg execution time is        : {}'.format(i_time / 1000.))    # 
print('Numba Parallel execution time is : {}'.format(n_time / 1000.))    # 

####################################################################################################################
# GPU Vectorized
####################################################################################################################

# Measure
n_time    = 0.
i_time    = 0.
to_time   = 0.
from_time = 0.

# run in
data_cube_cuda  = cuda.to_device(data_cube)                            # send new data
background_cuda = cuda.to_device(background)                           # send background location
vector_gpu(data_cube_cuda, background_cuda, flatfield_cuda, out=data_cube_corr_cuda)
cuda.synchronize()                                                     # 
data_cube_corr = data_cube_corr_cuda.copy_to_host()                    # retrieve results

for i in range(1000):

    # Find background
    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten)                                 # search for minimum intensity 
    background = data_cube[background_indx, :, :]
    end_time = timer()
    i_time = i_time + (end_time - start)

    # Send to GPU
    start = timer()
    data_cube_cuda  = cuda.to_device(data_cube)                        # send new data
    background_cuda = cuda.to_device(background)                       # send background location
    end_time = timer()
    to_time = to_time + (end_time - start)

    # Excute calibration
    start = timer()
    data_cube_corr = vector_gpu(data_cube_cuda, background_cuda, flatfield_cuda) 
    cuda.synchronize()                                                 # 
    end_time = timer()
    n_time = n_time + (end_time - start)

    # Retrieve from GPU
    start =  timer();
    data_cube_corr = data_cube_corr_cuda.copy_to_host()                # retrieve results
    end_time = timer()
    from_time = from_time + (end_time - start)

print('Find bg execution time is        : {}'.format(i_time    / 1000.))  
print('Data to GPU is                   : {}'.format(to_time   / 1000.)) 
print('Numba GPU execution time is      : {}'.format(n_time    / 1000.))  
print('Data from GPU is                 : {}'.format(from_time / 1000.))  

####################################################################################################################
# CUDA Custom Cuda Kernel
####################################################################################################################

# GPU block and thread allocation
blockdim = (1,  4,  4) # how to figure these out?
griddim  = (4, 32, 32) # how to figure these out?

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
    end_time = timer()
    n_time = n_time + (end_time - start)

    # Retrieve from GPU
    start = timer()
    #cuda.synchronize()                                                #  
    data_cube_corr = data_cube_corr_kernel_cuda.copy_to_host()         # obtain results back
    from_time = from_time + (timer() - start)

print('Find bg execution time is        : {}'.format(i_time   /1000.))  
print('Data to GPU is                   : {}'.format(to_time  /1000.)) 
print('CUDA Kernel execution time is    : {}'.format(n_time   /1000.))  
print('Data from GPU is                 : {}'.format(from_time/1000.)) 

# blockdim 1, 32,  8 griddim  1, 32, 16:  0.74 ms
# blockdim 2, 32,  8 griddim  2, 32, 16:  0.74 ms
# blockdim 2,  4,  4 griddim  4, 16, 16:  0.73 ms
# blockdim 1,  2,  2 griddim  1,  2,  2:  1.2  ms
# blockdim 1,  4,  4 griddim  1,  4,  4:  1.3  ms
# blockdim 1,  4,  4 griddim  1,  8,  8:  0.76 ms
# blockdim 1,  2,  2 griddim  1,  8,  8:  0.74 ms
# blockdim 1,  2,  2 griddim  1, 16, 16:  0.78 ms
# blockdim 1,  4,  4 griddim  1, 16, 16:  0.73 ms
# blockdim 1,  4,  4 griddim  2, 16, 16:  0.73 ms
# blockdim 1,  4,  4 griddim  4, 16, 16:  0.72 ms
# blockdim 1,  4,  4 griddim  4, 32, 32:  0.72 ms

####################################################################################################################
# CuPy 
####################################################################################################################

# Subtract background from images and multiply with flatfield
calibrate_cp(data_cube_cp, flatfield_cp, background_cp, data_cube_corr_cp)
# cp.multiply(cp.subtract(data_cube_gpu, data_cube_gpu[frame_idx_bg, :, :]), flatfield_gpu, out=data_cube_corr_cp) # 16bit multiplication

data_cube_corr = cp.asnumpy(data_cube_corr_cp)

print(repeat(calibrate_cp, (data_cube_cp, flatfield_cp, background_cp, data_cube_corr_cp), n_repeat=1000, n_warmup=10))

# Real run

for i in range(1000):

    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten) # just take a few points in an image
    # which frame has minimum intensity?
    frame_idx_bg = np.argmin(inten)
    background_cp = cp.array(data_cube[frame_idx_bg,:,:])
    end_time = timer()
    i_time = i_time + (end_time - start)
    
    start = timer()
    # copy image data to GPU
    data_cube_cp  = cp.array(data_cube, copy=True)
    end_time = timer()
    to_time = to_time + (end_time - start)

    start = timer()
    # Subtract background from images and multiply with flatfield
    calibrate_cp(data_cube_cp, flatfield_cp, background_cp, data_cube_corr_cp)
    # cp.multiply(cp.subtract(data_cube_gpu, data_cube_gpu[frame_idx_bg, :, :]), flatfield_gpu, out=data_cube_corr_cp) # 16bit multiplication
    end_time = timer()
    s_time = s_time + (end_time - start)

    start = timer()
    # copy image data from GPU to CPU
    data_cube_corr = cp.asnumpy(data_cube_corr_cp)
    end_time = timer()
    from_time = from_time + (end_time - start)

print('Find BG           execution time is : {}'.format(i_time    / 1000.0)) # 0.2ms
print('Transfer to GPU   execution time is : {}'.format(to_time   / 1000.0)) # 0.61ms
print('CuPy sub/mult     execution time is : {}'.format(s_time    / 1000.0)) # 0.11ms
print('Transfer from GPU execution time is : {}'.format(from_time / 1000.0)) # 2.99ms

####################################################################################################################
# CuPy Vectorized
####################################################################################################################

# Measure
n_time    = 0.
i_time    = 0.
to_time   = 0.
from_time = 0.

# run in
data_cube_cp    = cp.array(data_cube, copy=True)
background_cp   = cp.array(data_cube[3,:,:])
vector_cp(data_cube_cp, background_cp, flatfield_cp, out=data_cube_corr_cp)
cuda.synchronize()                                                     # 
data_cube_corr = cp.asnumpy(data_cube_corr_cp)                    # retrieve results

for i in range(1000):

    # Find background
    start = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten)                                 # search for minimum intensity 
    background = data_cube[background_indx, :, :]
    end_time = timer()
    i_time = i_time + (end_time - start)

    # Send to GPU
    start = timer()
    data_cube_cp  = cp.array(data_cube, copy=True)
    background_cp = cp.array(background, copy=True)
    end_time = timer()
    to_time = to_time + (end_time - start)

    # Excute calibration
    start = timer()
    vector_cp(data_cube_cp, background_cp, flatfield_cp, out = data_cube_corr_cp) 
    cuda.synchronize()
    end_time = timer()
    n_time = n_time + (end_time - start)

    # Retrieve from GPU
    start =  timer();
    data_cube_corr = cp.asnumpy(data_cube_corr_cp)
    end_time = timer()
    from_time = from_time + (end_time - start)

print('Find bg execution time is        : {}'.format(i_time    / 1000.))  
print('Data to GPU is                   : {}'.format(to_time   / 1000.)) 
print('Numba CuPy execution time is     : {}'.format(n_time    / 1000.))  
print('Data from GPU is                 : {}'.format(from_time / 1000.))  
