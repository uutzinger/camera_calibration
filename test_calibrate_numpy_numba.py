import numpy as np
import cupy as cp
from numba import jit, uint8, uint16, vectorize, cuda
import time

@vectorize(['uint16(uint8, uint16, uint8)'], nopython=True, fastmath=True)
def calibrate(data_cube, flatfield, bg):
    # return (data_cube - bg) * flatfield # 16bit multiplication
    return np.multiply(np.subtract(data_cube, bg), flatfield) # 16bit multiplication

# Simulated image data, we have 14 images
data_cube        = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))   # data will likley be 8 or 12 bit
bg               = (np.random.randint(0, 255, (540, 720), 'uint8'))   # data will likley be 8 or 12 bit
flatfield        = np.cast['uint16'](255.*np.random.random((540, 720))) # we can scale flatfield so that 255=100%

n_time  = 0
total_time = 0
i_time = 0

# Intensities
inten = np.zeros(14, dtype=np.uint16)

# Run in
data_cube_ff = calibrate(data_cube, flatfield, bg)

for i in range(10000):

    start_time = time.perf_counter()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    # minimum intensity is at which frame  index
    frame_idx_bg = np.argmin(inten)
    bg = data_cube[frame_idx_bg, :, :]

    numba_start_time = time.perf_counter()
    data_cube_ff = calibrate(data_cube, flatfield, bg)
    numba_end_time = time.perf_counter()

    i_time  = i_time + (numba_start_time - start_time)
    n_time  = n_time + (numba_end_time - numba_start_time)
    total_time = total_time + (numba_end_time - start_time)
    
print('Numpy BG           execution time is   : {}'.format(i_time/10000.0))     # 0.12ms
print('Numba              execution time is   : {}'.format(n_time/10000.0))     # 2.59ms
print('Total              execution time is   : {}'.format(total_time/10000.0)) # 2.71ms 
