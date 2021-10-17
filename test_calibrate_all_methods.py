import numpy as np
import cupy  as cp
from numba import cuda, vectorize, jit
from timeit import default_timer as timer
from cupyx.time import repeat

# Numba CUDA compiled
@cuda.jit(device=True)
def calibrate_gpu(img, ff, bg):
    return (img - bg) * ff

# Numba CPU compiled
@jit
def calibrate_cpu(img, ff, bg):
    return (img - bg) * ff

# CuPy
def calibrate_cp(dc, ff, bg, result):
    cp.multiply(cp.subtract(dc, bg), ff, out=result)

# CPU vectorized
@vectorize(['uint16(uint8, uint8, uint16)'], target='cpu')
def vector_cpu(data_cube,background,flatfield):
    return calibrate_cpu(data_cube, background, flatfield) 

# CPU parallelized
@vectorize(['uint16(uint8, uint8, uint16)'], target='parallel')
def vector_parallel(data_cube,background,flatfield):
    return calibrate_cpu(data_cube, background, flatfield) 

# CUDA vectorized
@vectorize(['uint16(uint8, uint8, uint16)'], target='cuda')
def vector_gpu(data_cube,background,flatfield):
    return calibrate_gpu(data_cube, background, flatfield)

# CUDA vectorized
@vectorize(['uint16(uint8, uint8, uint16)'], target='cuda')
def vector_gpu_direct(data_cube,background,flatfield):
    return (data_cube - background) * flatfield

# Numpy Vectorized
@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def vector_np(data_cube, background, flatfield):
    return np.multiply(np.subtract(data_cube, background), flatfield) # 16bit multiplication

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

data_cube_cp               = cp.array(data_cube, copy=True)           # send some data to GPU
flatfield_cp               = cp.array(flatfield, copy=True)           # send flatfield to GPU
data_cube_corr_cp          = cp.empty_like(data_cube_cp)              # resreve space for results on GPU
background_cp              = cp.array(data_cube[0,:,:])               # background on GPU


####################################################################################################################
# CPU, just python, Ohhh my....
####################################################################################################################
depth  = data_cube.shape[0] 
height = data_cube.shape[1]
width  = data_cube.shape[2]

# Measure
n_time    = 0.
background_indx = 3 # dont want to measure background search

if False:
    start = timer()
    for z in range (depth):
        for x in range(height):
            for y in range(width):
                I  = data_cube[z,x,y]
                bg = data_cube[background_indx, x, y]
                ff = flatfield[x,y]
                data_cube_corr[z, x, y] = (I - bg) * ff
    end_time = timer()
    n_time = n_time + (end_time - start)

    print('=================================================================')
    print('Python execution time is              : {%.2f}s'.format(n_time))  

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

    # Subtract background from images and mutiply flatfield
    start = timer()
    for frame_idx in range(0,14):
        if frame_idx != background_indx:
            _ = np.multiply(np.subtract(data_cube[frame_idx, :, :], data_cube[background_indx, :, :]), flatfield, out = data_cube_corr[frame_idx, :, :]) 
    end_time = timer()
    n_time = n_time + (end_time - start)

print('=================================================================')
print('Find bg execution time is                 : {0:8.2f}ms'.format(i_time ))  
print('Numpy execution time is                   : {0:8.2f}ms'.format(n_time ))  

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
    
print('=================================================================')
print('Find bg execution time is                 : {:8.2f}ms'.format(i_time ))  
print('Numba Numpy execution time is             : {:8.2f}ms'.format(n_time ))  

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

print('=================================================================')
print('Find bg execution time is                 : {:8.2f}ms'.format(i_time ))  
print('Numba CPU execution time is               : {:8.2f}ms'.format(n_time ))  

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

print('=================================================================')
print('Find bg execution time is                 : {:8.2f}ms'.format(i_time ))    # 
print('Numba Parallel execution time is          : {:8.2f}ms'.format(n_time ))    # 

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
    vector_gpu(data_cube_cuda, background_cuda, flatfield_cuda, out = data_cube_corr_cuda) 
    cuda.synchronize()                                                 # 
    end_time = timer()
    n_time = n_time + (end_time - start)

    # Retrieve from GPU
    start =  timer();
    data_cube_corr = data_cube_corr_cuda.copy_to_host()                # retrieve results
    end_time = timer()
    from_time = from_time + (end_time - start)

print('=================================================================')
print('Find bg execution time is                 : {:8.2f}ms'.format(i_time    ))  
print('Sending data to GPU with numba takes      : {:8.2f}ms'.format(to_time   )) 
print('Numba GPU execution time is               : {:8.2f}ms'.format(n_time    ))  
print('Retrieving data from GPU with Numba takes : {:8.2f}ms'.format(from_time ))  

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
vector_gpu_direct(data_cube_cuda, background_cuda, flatfield_cuda, out=data_cube_corr_cuda)
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
    vector_gpu_direct(data_cube_cuda, background_cuda, flatfield_cuda, out = data_cube_corr_cuda) 
    cuda.synchronize()                                                 # 
    end_time = timer()
    n_time = n_time + (end_time - start)

    # Retrieve from GPU
    start =  timer();
    data_cube_corr = data_cube_corr_cuda.copy_to_host()                # retrieve results
    end_time = timer()
    from_time = from_time + (end_time - start)

print('=================================================================')
print('Find bg execution time is                 : {:8.2f}ms'.format(i_time    ))  
print('Sending data to GPU with numba takes      : {:8.2f}ms'.format(to_time   )) 
print('Numba GPU direct execution time is        : {:8.2f}ms'.format(n_time    ))  
print('Retrieving data from GPU with Numba takes : {:8.2f}ms'.format(from_time ))  

####################################################################################################################
# CUDA Custom Cuda Kernel
####################################################################################################################

# GPU block and thread allocation
blockdim = (2,  4, 4) # how to figure these out?
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

# Measure
n_time    = 0.
i_time    = 0.
to_time   = 0.
from_time = 0.

# Subtract background from images and multiply with flatfield
calibrate_cp(data_cube_cp, flatfield_cp, background_cp, data_cube_corr_cp)
# cp.multiply(cp.subtract(data_cube_gpu, data_cube_gpu[frame_idx_bg, :, :]), flatfield_gpu, out=data_cube_corr_cp) # 16bit multiplication

data_cube_corr = cp.asnumpy(data_cube_corr_cp)

print('=================================================================')
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
    cp.cuda.stream.get_current_stream().synchronize()
    # cp.multiply(cp.subtract(data_cube_gpu, data_cube_gpu[frame_idx_bg, :, :]), flatfield_gpu, out=data_cube_corr_cp) # 16bit multiplication
    end_time = timer()
    n_time = n_time + (end_time - start)

    start = timer()
    # copy image data from GPU to CPU
    data_cube_corr = cp.asnumpy(data_cube_corr_cp)
    end_time = timer()
    from_time = from_time + (end_time - start)

print('=================================================================')
print('Find BG execution time is                 : {:8.2f}ms'.format(i_time    )) # 0.2ms
print('Transfer to GPU with CuPy takes           : {:8.2f}ms'.format(to_time   )) # 0.61ms
print('CuPy sub/mult execution time is           : {:8.2f}ms'.format(n_time    )) # 0.11ms
print('Transfer from GPU with CuPy takes         : {:8.2f}ms'.format(from_time )) # 2.99ms
