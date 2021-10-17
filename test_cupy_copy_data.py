import cupy as cp
import numpy as np
import time

#
# Pre run with large arrays
#

# Simulated image data, we have 14 images
data_cube_8  = (np.random.randint(0, 255, (32, 2048, 2048), 'uint8'))  # data will likley be 8 or 12 bit
data_cube_16 = (np.random.randint(0, 255, (32, 2048, 2048), 'uint16'))  # data will likley be 8 or 12 bit
data_cube_float  = np.cast['float32'](data_cube_16)  # data will likley be 8 or 12 bit

# copy data to GPU
data_cube_8_gpu      = cp.array(data_cube_8, copy=True)
data_cube_16_gpu     = cp.array(data_cube_16, copy=True)
data_cube_float_gpu  = cp.array(data_cube_float, copy=True)

cs=cp.cuda.get_current_stream()

# copy data to CPU
data_cube_8      = cp.asnumpy(data_cube_8, stream=cs, order='C')
data_cube_16     = cp.asnumpy(data_cube_16, stream=cs, order='C')
data_cube_float  = cp.asnumpy(data_cube_float, stream=cs, order='C')

#
# Now with real size
#

# Simulated image data, we have 14 images
data_cube_8  = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
data_cube_16 = (np.random.randint(0, 255, (14, 540, 720), 'uint16'))  # data will likley be 8 or 12 bit
data_cube_float  = np.cast['float32'](data_cube_16)  # data will likley be 8 or 12 bit

# copy data to GPU
data_cube_8_gpu      = cp.array(data_cube_8, copy=True)
data_cube_16_gpu     = cp.array(data_cube_16, copy=True)
data_cube_float_gpu  = cp.array(data_cube_float, copy=True)

cs=cp.cuda.get_current_stream()

# copy data to CPU
data_cube_8      = cp.asnumpy(data_cube_8, stream=cs, order='C')
data_cube_16     = cp.asnumpy(data_cube_16, stream=cs, order='C')
data_cube_float  = cp.asnumpy(data_cube_float, stream=cs, order='C')

# Looop
uint8_to = 0
uint16_to = 0
float_to  = 0
uint8_from = 0
uint16_from = 0
float_from  = 0

############################################################################
#
# Measure Performance
# 
############################################################################


for i in range(100):

    # copy image data to GPU
    uint8_to_time = time.perf_counter()
    data_cube_8_gpu  = cp.array(data_cube_8, copy=True)
    uint16_to_time = time.perf_counter()
    data_cube_16_gpu  = cp.array(data_cube_16, copy=True)
    float_to_time = time.perf_counter()
    data_cube_float_gpu  = cp.array(data_cube_float, copy=True)

    # copy image data to CPU
    uint8_from_time = time.perf_counter()
    data_cube_8 = cp.asnumpy(data_cube_8_gpu, stream=cs, order='C')
    uint16_from_time = time.perf_counter()
    data_cube_16 = cp.asnumpy(data_cube_16_gpu, stream=cs, order='C')
    float_from_time = time.perf_counter()
    data_cube_float = cp.asnumpy(data_cube_float_gpu, stream=cs, order='C')
    end_time = time.perf_counter()

    uint8_to  = uint8_to      + ( uint16_to_time   - uint8_to_time)
    uint16_to = uint16_to     + ( float_to_time    - uint16_to_time)
    float_to  = float_to      + ( uint8_from_time  - float_to_time)
    uint8_from  = uint8_from  + ( uint16_from_time - uint8_from_time)
    uint16_from = uint16_from + ( float_from_time  - uint16_from_time)
    float_from  = float_from  + ( end_time         - float_from_time)

print('uint8  to   execution time is : {}'.format(uint8_to/100.0)) # 5.9ms
print('uint16 to   execution time is : {}'.format(uint16_to/100.0)) # 5.9ms
print('float  to   execution time is : {}'.format(float_to/100.0)) # 5.9ms
print('uint8  from execution time is : {}'.format(uint8_from/100.0)) # 5.9ms
print('uint16 from execution time is : {}'.format(uint16_from/100.0)) # 5.9ms
print('float  from execution time is : {}'.format(float_from/100.0)) # 5.9ms
