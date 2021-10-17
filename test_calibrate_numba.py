import numpy as np
from numba import vectorize, uint16, uint8, jit
import time

#@vectorize(['uint16(uint8, uint16, uint8)'], target='cpu')
@vectorize([uint16(uint8, uint16, uint8)], target='cuda')
#@jit(uint16(uint8, uint16, uint8))
def calibrate(data_cube, flatfield, bg):
    return (data_cube - bg) * flatfield

# Simulated image data, we have 14 images
data_cube      = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
bg             = (np.random.randint(0, 255,     (540, 720), 'uint8'))  # where we keep bg
flatfield      = np.cast['uint16'](2**8.*np.random.random((540, 720))) # we can scale flatfield so that 255=100%
inten          = np.zeros(14, 'uint16')                         # help to find background image
data_cube_corr = np.zeros((14, 540, 720), 'uint16')             # result

n_time  = 0
total_time = 0
i_time = 0

# Run in
for i in range(10):
    data_cube_corr=calibrate(data_cube, flatfield, bg)

# Measure
for i in range(1000):

    # Find image with lowest intensity which is the background
    start_time = time.perf_counter()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    frame_idx_bg = np.argmin(inten) # search for minimum intensity 
    bg = data_cube[frame_idx_bg, :, :]

    # Apply calibration
    numba_start_time = time.perf_counter()
    data_cube_corr = calibrate(data_cube, flatfield, bg)
    numba_end_time = time.perf_counter()

    i_time     = i_time     + (numba_start_time - start_time)
    n_time     = n_time     + (numba_end_time   - numba_start_time)
    total_time = total_time + (numba_end_time   - start_time)
    
print('Find bg execution time is : {}'.format(i_time/1000.0))     # 0.13ms
print('Numba   execution time is : {}'.format(n_time/1000.0))     # 3.9ms
print('Total   execution time is : {}'.format(total_time/1000.0)) # 4.0ms 

# Numba 
# CPU   CUDA
#0.1ms   0.1 ms
#3.3 ms 21.2 ms
#3.4 ms 21.3 ms
