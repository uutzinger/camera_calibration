import numpy as np
#import cv2 as cv
import time

# Simulated image data, we have 14 images
data_cube        = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))   # data will likley be 8 or 12 bit
data_cube_bg     = (np.random.randint(0, 255, (14, 540, 720), 'uint8'))  # data will likley be 8 or 12 bit
data_cube_ff     = (np.random.randint(0, 255, (14, 540, 720), 'uint16'))  # data will likley be 8 or 12 bit
# Simulated flatfield data
flatfield        = np.cast['uint16'](255.*np.random.random((540, 720))) # we can scale flatfield so that 255=100%

np_time = 0
s_time  = 0
m_time  = 0
i_time  = 0

# Intensities
inten = np.zeros(14, dtype=np.uint16)

for i in range(10000):
    ############################################################################
    #
    # Numpy approach with ROI
    #
    ############################################################################

    start_time = time.perf_counter()
    _ = np.sum(data_cube[:,::64,::64], axis=(1,2), out = inten)
    # minimum intensity is at which frame  index
    frame_idx_bg = np.argmin(inten)

    s_start_time = time.perf_counter()
    # Subtract background from images
    for frame_idx in range(0,14):
        if frame_idx != frame_idx_bg:
            _ = np.subtract(data_cube[frame_idx, :, :], data_cube[frame_idx_bg, :, :], out=data_cube_bg[frame_idx, :, :]) # 8bit subtractions

    m_start_time = time.perf_counter()
    # Multiple flatfield to images
    for frame_idx in range(0,14):
        if frame_idx != frame_idx_bg:
            _ = np.multiply(data_cube_bg[frame_idx, :, :], flatfield, out = data_cube_ff[frame_idx, :, :]) # 16bit multiplication

    end_time = time.perf_counter()

    np_time = np_time + (end_time - start_time)
    s_time  = s_time + (m_start_time - s_start_time)
    m_time  = m_time + (end_time - m_start_time)
    i_time  = i_time + (s_start_time - start_time)
    
print('Numpy subtract     execution time is   : {}'.format(s_time/10000.0))     # 2.80ms
print('Numpy multiply     execution time is   : {}'.format(m_time/10000.0))     # 6.10ms
print('Numpy BG           execution time is   : {}'.format(i_time/10000.0))     # 0.09ms
print('Numpy              execution time is   : {}'.format(np_time/10000.0))    # 8.89ms

# For lookup table conversion
# lut[image]
# np.take(lut, image)

############################################################################
#
# OpenCV CUDA approach
#
############################################################################


#cu_cv2_data_cube  = cv2.cuda_GpuMat() # 14 images
#cu_cv2_flatfield  = cv2.cuda_GpuMat() # one flat field
#cu_cv2_data_cubeC = cv2.cuda_GpuMat() # the corrected data cube

# Upload constant flatfield
#before_time = time.perf_counter()
#cu_cv2_flatfield.upload(cu_cv2_flatfield)
#after_time = time.perf_counter()
#print("Flatfield upload:{}s".format(after_time-before_time))

# Initialize the workflow
#jit_time = time.time()
#_ = cv2.cuda.gemm(cuMat1, cuMat2,1,None,0,None,1)
#current_time = time.time()

# Looop

# Upload data cube
#cu_cv2_data_cube.upload(cu_cv2_data_cube)

# Find background image in datacube

# Subtract background from images

# Multiple flatfield to images
#_ = cv2.cuda.gemm(cuMat1, cuMat2,1,None,0,None,1)
