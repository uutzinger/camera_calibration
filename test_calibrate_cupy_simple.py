import cupy as cp
import numpy as np
from timeit import default_timer as timer
from cupyx.time import repeat

# Loop
cp_time   = 0
s_time    = 0
m_time    = 0
i_time    = 0
t_time    = 0
to_time   = 0.
from_time = 0.

def calibrate(data_cube, flatfield, bg, result):
    cp.multiply(cp.subtract(data_cube, bg), flatfield, out=result) # 16bit multiplication

############################################################################
#
# Dryrun copy large arrays, does not make difference here
#
############################################################################

# Simulated image data, we have 14 images
data_cube              = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))
data_cube_result       = (np.random.randint(0, 255, (14, 540, 720), 'uint16'))
flatfield              = np.cast['uint16'](255.*np.random.random((540, 720)))
inten                  = np.zeros(14, 'uint16')
data_cube_cp           = cp.array(data_cube, copy=True)
flatfield_cp           = cp.array(flatfield, copy=True)
data_cube_result_cp    = cp.empty_like(data_cube_cp)
background_cp          = cp.array(data_cube[0,:,:])

############################################################################
#
# Dryrun for jit with regular arrays
#
############################################################################

# measure intensities in the image to determine bg frame
_ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten) # just take a few points
# which frame has minimum intensity?
frame_idx_bg  = np.argmin(inten)
background_cp = cp.array(data_cube[frame_idx_bg,:,:])

# Subtract background from images and multiply with flatfield
calibrate(data_cube_cp, flatfield_cp, background_cp, data_cube_result_cp)
# cp.multiply(cp.subtract(data_cube_gpu, data_cube_gpu[frame_idx_bg, :, :]), flatfield_gpu, out=data_cube_result_gpu) # 16bit multiplication

data_cube_result = cp.asnumpy(data_cube_result_cp)

print(repeat(calibrate, (data_cube_cp, flatfield_cp, background_cp, data_cube_result_cp), n_repeat=1000, n_warmup=10))

############################################################################
#
# Real run
#
############################################################################

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
    calibrate(data_cube_cp, flatfield_cp, background_cp, data_cube_result_cp)
    # cp.multiply(cp.subtract(data_cube_gpu, data_cube_gpu[frame_idx_bg, :, :]), flatfield_gpu, out=data_cube_result_gpu) # 16bit multiplication
    end_time = timer()
    s_time = s_time + (end_time - start)

    start = timer()
    # copy image data from GPU to CPU
    data_cube_result = cp.asnumpy(data_cube_result_cp)
    end_time = timer()
    from_time = from_time + (end_time - start)

print('Find BG           execution time is : {}'.format(i_time    / 1000.0)) # 0.2ms
print('Transfer to GPU   execution time is : {}'.format(to_time   / 1000.0)) # 0.61ms
print('CuPy sub/mult     execution time is : {}'.format(s_time    / 1000.0)) # 0.11ms
print('Transfer from GPU execution time is : {}'.format(from_time / 1000.0)) # 2.99ms
