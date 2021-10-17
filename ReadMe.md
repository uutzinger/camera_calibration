# Camera Calibration

## Summary

The task is to calibrate a stack of 14 images by subtracting the background image and by adjusting them for a flatfield.
```
(image - background) * flatfield
```
Image data and background data is either ```uint8``` or ```uint16```.  
The flatfield is ```0...1``` and should scaled by ```255```.  Then the results will be ```uint16``` and range from ```0...65535```

From speed results below, one concludes that GPU calculation does not accelerate this problem. However if other more complex computations are needed, this might change.

### Speed Results
All times are in ```milli seconds```.
Numpy was obtained from https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy with intel math kernel optimization and matched the version required by Numba. 

| Implementation   | Backend |Background | To Gpu| Calibrate| From GPU |
|---               |-----    |----       |-----  |-----     |-----     |
| Pure Python      | Python  | N.A.      | N.A.  | 180000   |  N.A.    |
| Numpy            | Numpy   | 0.08      | N.A.  | 8.6      |  N.A.    |
| Numpy Vectorized | Numba   | 0.09      | N.A.  | 1.27     |  N.A.    |
| Vectorized CPU   | Numba   | 0.09      | N.A.  | 3.2      |  N.A.    |
| Parallel         | Numba   | 0.14      | N.A.  | 15.53    |  N.A.    |
| Vectorized GPU   | Numba   | 0.09      | 3.64  | 16.20    |  4.22    |
| Vectorized GPU d | Numba   | 0.09      | 3.63  | 16.26    |  4.21    |
| Cuda Kernel      | Numba   | 0.08      | 3.08  |  1.27    |  4.17    |
| CuPy             | CuPy    | 0.26      | 0.59  |  1.20    |  2.22    |

```
calibrate_cp        :    CPU:   39.753 us   +/-14.521 (min:   31.100 / max:  202.900) us     GPU-0:  389.680 us   +/-20.924 (min:  356.352 / max:  617.472) us
```

As can be seen, Numpy jit compiled with Numba has best performance if no other cacluations are needed as the GPU requires the same amount of computation time but has additional overhead.

## Pre Requisites
* Install CUDA https://developer.nvidia.com/cuda-toolkit
* Install https://scipy.org/
  * ```pip3 install -U scipy```
  * ```pip3 install -U matplotlib```

* Update setuppools ```pip3 install -U setuptools pip```
* Install CuPy ```pip3 install cupy-cuda114```

## Readings

Cupy: https://docs.cupy.dev/en/stable/user_guide/  
Numba: https://github.com/ContinuumIO/gtc2020-numba and https://www.youtube.com/watch?v=CQDsT81GyS8  
Numba: https://developer.nvidia.com/blog/numba-python-cuda-acceleration/  
Numba: https://numba.pydata.org/  

## Calibration
Implementation of the calibration routines for the differente backends

* CuPy  
```
def calibrate_cp(data, ff, bg, result):
    cp.multiply(cp.subtract(data, bg), ff, out=result)
```
The ouput should be preallocated with ``` result  = cp.empty_like(data) ```. By specifybg ```out = ...``` data will not be returned to host but copied to ```out```.

* Numba on Cuda  
```
@vectorize(['uint16(uint8, uint8, uint16)'], target='cuda')
def vector_gpu(data_cube,background,flatfield):
    return (data_cube - background) * flatfield
```
I did not verify that this is applying numpy type rules for element wise calculations.

* Numba on CPU  
```
target='cpu'
```

* Numba on Parallel  
```
target='parallel'
```

* Numba and Numpy
```
@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def vector_np(data_cube, background, flatfield):
    return np.multiply(np.subtract(data_cube, background), flatfield)
```
This is the fastest implementation and uses elementwise calculations. I did not verify that data types are applied correctly e.g. (255-0)*255 = 65025 and not 255.

* Numba with Custom Kernel

Here we create blocks of data and specify how many threads per block we use. The following breakup resulted in fastest execuation for the specific data. The numbers were determined experimentally (increasing/decreasing each by factor of 2 and assuming max in z is 14)
```
blockdim = (2, 4, 4)
griddim  = (14, 8, 8)
```
The GPU conducts the simple math operations on each element and the problem is split up along the 3 axis with each thread operating on a block of data. 
```
# blockDim.x,y,z gives number of threads in a block, in the particular direction
# gridDim.x,y.z  gives the number of blocks in a grid, in the particular direction
# blockDim.x * gridDim.x gives the number of threads in a grid in the x direction
# blockIdx  is the block  index in the grid
# threadIdx is the thread index within the block
@cuda.jit
def calibrate(data_cube,background_indx,flatfield, result):
    depth  = data_cube.shape[0] 
    height = data_cube.shape[1]
    width  = data_cube.shape[2]

    startZ, startX, startY = cuda.grid(3)
    gridZ = cuda.gridDim.z * cuda.blockDim.z;
    gridX = cuda.gridDim.x * cuda.blockDim.x;
    gridY = cuda.gridDim.y * cuda.blockDim.y;
    for z in range (startZ,depth,gridZ):
        for x in range(startX, height, gridX):
            for y in range(startY, width, gridY):
                I  = data_cube[z,x,y]
                bg = data_cube[background_indx, x, y]
                ff = flatfield[x, y]
                result[z, x, y] = (I-bg)*ff
```
The kernel is called like:
```
calibrate[griddim, blockdim](data_cuda, background_indx_cuda, flatfield_cuda, results_cuda) 
```
## Data
For simulated image data, we have 14 images in one cube matching the resolution of our camera (540x720), we have a synthetic flatfield, we search in the image cube for the image with lowest intensity as that is the background. You might need to adjust the names here. Flatfield remains constant while background and data_cube change for each data capture.

```
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
```

## Transferring Data to GPU
When calculating on the GPU we need to send the data to the GPU and retrieve the data to the host computer when GPU is finished. If we dont synchronize the calculations, measuring time to retrieve the data will include GPU computation time as retrieving will first have to wait until GPU is finished.

### Numba
```
data_gpu = cuda.to_device(data)    # send
data     = data_gpu.copy_to_host() # retrieve
```

### CuPy
```
data_cp = cp.array(data, copy=True)  # send
data    = cp.asnumpy(data_cp)        # retrieve
```
CuPy datatransfer is fastest, and since Numba and CuPy data types are the same, one should transfter data using CuPy also for Numba accelerated functions.

## Syncing
Computations on the GPU run in the background. You will need to wait until GPU is finished with calculations using:

* Numba ``` cuda.synchronize() ```
* CuPy ```cp.cuda.stream.get_current_stream().synchronize()  ```  
if you want to measure accurate execuation times.

## Lowest Intesity Image in Data Stack
To find the lowest intensity image we use a few pixels in the image and sum the intensity. Then we find the smallest sum.
```
inten = np.zeros(14, 'uint16') # pre allocate
_ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
background_indx = np.argmin(inten) # one of the 14 images is the background
```