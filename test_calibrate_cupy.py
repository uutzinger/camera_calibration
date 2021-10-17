import cupy as cp
import numpy as np
import time

# Simulated image data, we have 14 images
data_cube        = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
data_cube_bg     = (np.random.randint(0, 255, (14, 540, 720), 'uint16'))  # data will likley be 8 or 12 bit
data_cube_ff     = (np.random.randint(0, 255, (14, 540, 720), 'uint16'))  # data will likley be 8 or 12 bit
# Simulated flatfield data
flatfield        = np.cast['uint16'](255.*np.random.random((540, 720))) # we can scale flatfield so that 255=100%

############################################################################
#
# CuPy approach
#
############################################################################
# Looop
cp_time = 0
s_time  = 0
m_time  = 0
i_time  = 0
t_time  = 0

# Intensities
inten = np.zeros(14, dtype=np.uint16)

############################################################################
#
# Dryrun for jit
#
############################################################################

# copy data to GPU
data_cube_gpu        = cp.asarray(data_cube)
data_cube_bg_gpu     = cp.asarray(data_cube_bg)
data_cube_ff_gpu     = cp.asarray(data_cube_ff)
flatfield_gpu        = cp.asarray(flatfield)

# measure intensities in the image to determine bg frame
_ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten) # just take a few points
# which frame has minimum intensity?
frame_idx_bg = np.argmin(inten)

# Subtract background from images
for frame_idx in range(0,14):
    if frame_idx != frame_idx_bg:
        _ = cp.subtract(data_cube_gpu[frame_idx, :, :], data_cube_gpu[frame_idx_bg, :, :], out=data_cube_bg_gpu[frame_idx, :, :])

# Multiple flatfield to images
frame_idx = 0
for frame_idx in range(0,14):
    if frame_idx != frame_idx_bg:
        _ = cp.multiply(data_cube_bg_gpu[frame_idx, :, :], flatfield_gpu, out = data_cube_ff_gpu[frame_idx, :, :])

data_cube_res_gpu = (data_cube_ff_gpu / 255)
data_cube_res = cp.asnumpy(data_cube_res_gpu)

############################################################################
#
# Real run
#
############################################################################

for i in range(10000):

    idx_start_time = time.perf_counter()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    # minimum intensity is at which frame  index
    frame_idx_bg = np.argmin(inten)

    t1_start_time = time.perf_counter()
    # copy image data to GPU
    data_cube_gpu  = cp.asarray(data_cube)

    s_start_time = time.perf_counter()
    # Subtract background from images
    for frame_idx in range(0,14):
        if frame_idx != frame_idx_bg:
            _ = cp.subtract(data_cube_gpu[frame_idx, :, :], data_cube_gpu[frame_idx_bg, :, :], out=data_cube_bg_gpu[frame_idx, :, :])

    m_start_time = time.perf_counter()
    # Multiple flatfield to images
    frame_idx = 0
    for frame_idx in range(0,14):
        if frame_idx != frame_idx_bg:
            _ = cp.multiply(data_cube_bg_gpu[frame_idx, :, :], flatfield_gpu, out = data_cube_ff_gpu[frame_idx, :, :])

    t2_start_time = time.perf_counter()
    # copy image data from GPU to CPU
    data_cube_ff = cp.asnumpy(data_cube_ff_gpu)

    end_time = time.perf_counter()

    s_time     = s_time  + (  m_start_time -   s_start_time) # subtract
    m_time     = m_time  + ( t2_start_time -   m_start_time) # multiply
    cp_time    = cp_time + (      end_time - idx_start_time) # total
    i_time     = i_time  + ( t1_start_time - idx_start_time) # find bg
    t_time     = t_time  + (  s_start_time -  t1_start_time) + (end_time - t2_start_time) # transfer data gpu-cpu
                                                                        # uint16
print('CuPy          execution time is   : {}'.format(cp_time/10000.0)) # 5.9ms
print('Find BG       execution time is   : {}'.format(i_time/10000.0))  # 0.09ms
print('Transfer      execution time is   : {}'.format(t_time/10000.0))  # 5.35ms
print('CuPy subtract execution time is   : {}'.format(s_time/10000.0))  # 0.32ms
print('CuPy multiply execution time is   : {}'.format(m_time/10000.0))  # 0.22ms
