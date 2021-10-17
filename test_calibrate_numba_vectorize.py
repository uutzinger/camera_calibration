import numpy as np
from timeit import default_timer as timer
from numba import vectorize

# Simulated image data, we have 14 images
data_cube      = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
bg             = (np.random.randint(0, 255,     (540, 720), 'uint8'))  # where we keep bg
flatfield      = np.cast['uint16'](2**8.*np.random.random((540, 720))) # we can scale flatfield so that 255=100%
inten          = np.zeros(14, 'uint16')                         # help to find background image
data_cube_corr = np.zeros((14, 540, 720), 'uint16')             # result

@vectorize(['uint16(uint8, uint16, uint8)'], target='cpu')
def calibrate_cpu(data_cube, flatfield, bg):
    return (data_cube - bg) * flatfield

@vectorize(['uint16(uint8, uint16, uint8)'], target='cuda')
def calibrate_gpu(data_cube, flatfield, bg):
    return (data_cube - bg) * flatfield

####################################################################################################################
# CUDA
####################################################################################################################

n_time  = 0
total_time = 0
i_time = 0

# Run in
data_cube_corr = calibrate_gpu(data_cube, flatfield, bg)

for i in range(1000):

    start_time = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten) # search for minimum intensity 
    bg = data_cube[background_indx, :, :]

    numba_start_time = timer()
    data_cube_corr = calibrate_gpu(data_cube, flatfield, bg)
    numba_end_time = timer()

    i_time  = i_time + (numba_start_time - start_time)
    n_time  = n_time + (numba_end_time - numba_start_time)
    total_time = total_time + (numba_end_time - start_time)
    
print('Numpy BG   execution time is : {}'.format(i_time/1000.0))     #  0.1ms  0.1ms
print('Numba CUDA execution time is : {}'.format(n_time/1000.0))     # 40.4ms  2.7ms
print('Total      execution time is : {}'.format(total_time/1000.0)) # 40.5ms  2.8ms

####################################################################################################################
# CPU
####################################################################################################################

n_time  = 0
total_time = 0
i_time = 0

# Run in
data_cube_corr = calibrate_cpu(data_cube, flatfield, bg)

for i in range(1000):

    start_time = timer()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    background_indx = np.argmin(inten) # search for minimum intensity 
    bg = data_cube[background_indx, :, :]

    numba_start_time = timer()
    data_cube_corr = calibrate_cpu(data_cube, flatfield, bg)
    numba_end_time = timer()

    i_time  = i_time + (numba_start_time - start_time)
    n_time  = n_time + (numba_end_time - numba_start_time)
    total_time = total_time + (numba_end_time - start_time)
    
print('Numpy BG  execution time is : {}'.format(i_time/1000.0))     #  0.1ms  0.1ms
print('Numba CPU execution time is : {}'.format(n_time/1000.0))     # 40.4ms  2.7ms
print('Total     execution time is : {}'.format(total_time/1000.0)) # 40.5ms  2.8ms
