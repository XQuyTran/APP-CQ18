from numba import cuda
from feature_extract_v0 import (BLOCK_SIZE, convert_rgb2gray_use_kernel,
                                                    gaussian_blur_kernel)
import math, numpy as np
from cv2 import imread, normalize, cvtColor, COLOR_BGR2HSV

from feature_extract import applyCannyThreshold

@cuda.jit
def conv_no_padding_ones_kernel(src, ksize, out):
    c, r = cuda.grid(2)
    offset = (ksize[0] // 2, ksize[1] // 2)
    r += offset[0]
    c += offset[1]

    out_val = 0
    if r < src.shape[0] - 1 and c < src.shape[1] - 1:
        for in_r in range(r - offset[0], r + offset[0] + 1):
            for in_c in range(c - offset[1], c + offset[1] + 1):
                out_val += src[in_r, in_c]

        out[r - offset[0], c - offset[1]] = out_val

@cuda.jit
def _blk_count_nonzero(src, blk_dim_x, blk_dim_y, out):
    '''
    Hàm xác định số phần tử khác 0 theo từng khối trong mảng đầu vào

    Tham số:
    - src (array): mảng đầu vào.
    - blk_dim_x, blk_dim_y (int): lần lượt là kích thước theo chiều rộng và cao của khối được xét.
    - out (array): mảng kết quả.
    '''
    c, r = cuda.grid(2)
    if c < src.shape[1] and r < src.shape[0] and src[r, c] > 0:
        cuda.atomic.add(out, (r // blk_dim_y, c // blk_dim_x), 1)

@cuda.jit
def _apply_threshold(src, threshold, criteria_arr, zip_x, zip_y):
    c, r = cuda.grid(2)
    if c < src.shape[1] and r < src.shape[0]:
        blk_r = r // zip_y
        blk_c = c // zip_x

        if criteria_arr[blk_r, blk_c] >= threshold:
            src[r, c] = 1
        else:
            src[r, c] = 0

def zipImageKernel(src, zip_x, zip_y, ratio, block_size=(32, 32)):
    '''
    Hàm nén ảnh src theo từng khối zip_x * zip_y với tỷ lệ ratio, chạy song song trên GPU.
    Khối ảnh được nén khi số điềm ảnh != 0 >= zip_x * zip_y * ratio.
    Kết quả thu được mảng đánh dấu những khối điểm ảnh được nén.

    Đầu vào:
    - src (numba.cuda.cudadrv.devicearray.DeviceNDArray): ảnh đầu vào.
    - zip_x, zip_y (int): kích thước khối theo chiều ngang và dọc.
    - ratio (float): tỷ lệ nén.
    - block_size (tuple): kích thước khối chạy song song trên GPU.

    Đầu ra:
    - mask_image (numba.cuda.cudadrv.devicearray.DeviceNDArray): mảng đánh dấu những khối ảnh được nén.
    '''

    # tính số lượng khối.
    zip_rs = math.ceil(src.shape[0] / zip_y)
    zip_cs = math.ceil(src.shape[1] / zip_x)
    nonzero_count = cuda.device_array((zip_rs, zip_cs), np.uint32)

    if not cuda.is_cuda_array(src):
        d_src = cuda.to_device(src)
    else:
        d_src = src
    grid_size = (math.ceil(src.shape[1] / block_size[0]), math.ceil(src.shape[0] / block_size[1]))

    # đếm số điểm ảnh !=0 và đánh dấu những khối được nén.
    _blk_count_nonzero[grid_size, block_size](d_src, zip_x, zip_y, nonzero_count)
    _apply_threshold[grid_size, block_size](d_src, zip_x*zip_y*ratio, nonzero_count, zip_x, zip_y)

    # cuda.synchronize()
    return d_src

@cuda.jit
def _apply_threshold_overlap(src, threshold, criteria_arr, zip_x, zip_y, mask_size):
    c, r = cuda.grid(2)
    end = False
    if r < src.shape[0] and c < src.shape[1]:
        init_block = (r // zip_y, c // zip_x)
        for block_r in range(init_block[0], max(-1, init_block[0] - mask_size - 1), -1):
            for block_c in range(init_block[1], max(-1, init_block[1]  - mask_size - 1), -1):
                if criteria_arr[block_r, block_c] >= threshold:
                    src[r, c] = 1
                    end = True
                    break

            if end:
                break

def joinNeighborPixelKernel(src, zip_x, zip_y, mask_size, ratio, block_size=(32, 32)):
    '''
    Hàm liên kết những khối điểm ảnh lân cận để lấp những điểm ảnh khuyết, chạy song song trên GPU.
    Kết quả là mảng đánh dấu những điểm ảnh được liên kết và lấp khuyết.

    Đầu vào:
    - src (numba.cuda.cudadrv.devicearray.DeviceNDArray): ảnh đầu vào.
    - zip_x, zip_y (int): kích thước khối theo chiều ngang và dọc.
    - mask_size (int): kích thước 1 khối lân cận xem xét liên kết.
    - ratio (float): tỷ lệ nén.
    - block_size (tuple): kích thước khối chạy song song trên GPU.

    Đầu ra:
    - mask_image (numba.cuda.cudadrv.devicearray.DeviceNDArray): mảng đánh dấu những khối ảnh được liên kết.
    '''
    zip_rs = math.ceil(src.shape[0] / zip_y)
    zip_cs = math.ceil(src.shape[1] / zip_x)
    blk_nonzero_count = cuda.device_array((zip_rs, zip_cs), np.uint32)

    mask_shape = (zip_rs*mask_size, zip_cs*mask_size)
    mask_nonzero_count = cuda.device_array(mask_shape, np.uint32)

    if not cuda.is_cuda_array(src):
        d_src = cuda.to_device(src)
    else:
        d_src = src

    grid_size = (math.ceil(src.shape[1] / block_size[0]), math.ceil(src.shape[0] / block_size[1]))
    _blk_count_nonzero[grid_size, block_size](d_src, zip_x, zip_y, blk_nonzero_count, (0, 0))

    grid_size_conv_func = (math.ceil(mask_shape[1] / block_size[0]), math.ceil(mask_shape[0] / block_size[1]))
    conv_no_padding_ones_kernel[grid_size_conv_func, block_size](blk_nonzero_count, (mask_size, mask_size), mask_nonzero_count)

    threshold = mask_size * mask_size * zip_x * zip_y * ratio
    _apply_threshold_overlap[grid_size, block_size](d_src, threshold, mask_nonzero_count, zip_x, zip_y, mask_size)

    return d_src

@cuda.jit
def m_ij_k(img,m_,i,j): # img 
    temp_ij=0
    r ,c= cuda.grid(2)
    if r < img.shape[0] and c < img.shape[1]:
        temp_ij=img[r,c]*(r**i)*(c**j)
        cuda.atomic.add(m_, (i,j),temp_ij)


def cvHuMoments_kernel(d_in_img):
    m_=np.zeros((4, 4), dtype=np.float64)
    mu_=np.zeros((4, 4), dtype=np.float64)
    nu_=np.zeros((4, 4), dtype=np.float64)
    hu_=np.zeros(7,np.float64) 

    # tính m_
    d_m_ = cuda.to_device(m_)
    grid_size = (math.ceil(d_in_img.shape[1] / BLOCK_SIZE[0]),
                        math.ceil(d_in_img.shape[1] / BLOCK_SIZE[1]))
    for i in range(4):
        for j in range(4):
            if (i,j) not in [(1,3),(2,2),(2,3)]:      
                m_ij_k[grid_size, BLOCK_SIZE](d_in_img, d_m_, i, j)
        if i==3:    
            break

    m_ = d_m_.copy_to_host(m_)
    # xbar ybar
    xbar=m_[1,0]/m_[0,0]
    ybar=m_[0,1]/m_[0,0]
    # tính mu
    mu_[1,1] = m_[1,1] - xbar*m_[0,1]
    mu_[0,2] = m_[2,0] - xbar*m_[1,0]
    mu_[2,0] = m_[0,2] - ybar*m_[0,1]
    mu_[1,2] = m_[2,1] - 2*xbar*m_[1,1] - ybar*m_[2,0] + 2*(xbar**2)*m_[0,1]
    mu_[2,1] = m_[1,2] - 2*ybar*m_[1,1] - xbar*m_[0,2] + 2*(ybar**2)*m_[1,0]
    mu_[0,3] = m_[3,0] - 3*xbar*m_[2,0] + 2*(xbar**2)*m_[1,0]
    mu_[3,0] = m_[0,3] - 3*ybar*m_[0,2] + 2*(ybar**2)*m_[0,1]

    #tính nu  nu_ji = mu_ji / [m00^(((i+j)/2)+1)]
    for i in range(4):
        for j in range(4):
            nu_[i,j] = mu_[i,j]/(m_[0,0]**(((i+j)/2)+1))
 
    hu_[0] =  nu_[2][0] + nu_[0][2]
    hu_[1] = (nu_[2][0] - nu_[0][2])**2 + 4*nu_[1][1]**2
    hu_[2] = (nu_[3][0] - 3*nu_[1][2])**2 + (3*nu_[2][1] - nu_[0][3])**2
    hu_[3] = (nu_[3][0] + nu_[1][2])**2 + (nu_[2][1] + nu_[0][3])**2
    hu_[4] = (nu_[3][0] - 3*nu_[1][2])*(nu_[3][0] + nu_[1][2])*((nu_[3][0] + nu_[1][2])**2 - 3*(nu_[2][1] + nu_[0][3])**2) +\
        (3*nu_[2][1] - nu_[0][3])*(nu_[2][1] + nu_[0][3])*(3*(nu_[3][0] + nu_[1][2])**2 - (nu_[2][1] + nu_[0][3])**2)
    hu_[5] = (nu_[2][0] - nu_[0][2])*((nu_[3][0] + nu_[1][2])**2 - (nu_[2][1] + nu_[0][3])**2) + \
          4*nu_[1][1]*(nu_[3][0] + nu_[1][2])*(nu_[2][1] + nu_[0][3])
    hu_[6] = (3*nu_[2][1] - nu_[0][3])*(nu_[2][1] + nu_[0][3])*(3*(nu_[3][0] + nu_[1][2])**2-(nu_[2][1] + nu_[0][3])**2) -\
          (nu_[3][0] - 3*nu_[1][2])*(nu_[1][2] + nu_[0][3])*(3*(nu_[3][0] + nu_[1][2])**2-(nu_[2][1] + nu_[0][3])**2)

    return hu_


@cuda.jit
def apply_mask_kernel(image, mask):
    c, r = cuda.grid(2)
    if c < image.shape[1] and r < image.shape[0]:
        for i in range(image.shape[2]):
            image[r, c, i] *= mask[r, c]


@cuda.jit
def _histogram_kernel(d_input,dhist):
    '''
    2d & 1d
    '''
    i = cuda.grid(1)
    if i < d_input.size:
        x,y,z=d_input[i][0]//32,d_input[i][1]//32,d_input[i][2]//32
        index=x*8*8 +y*8 + z
        cuda.atomic.add(dhist,index, 1)

def histogram_kernel(image, block_size):
    hsv_image = cvtColor(image, COLOR_BGR2HSV)
    d_hsv_image = cuda.to_device(hsv_image)

    d_flat_image = d_hsv_image.reshape(-1, d_hsv_image.shape[-1])
    hist_figure = np.zeros(8*8*8, np.float32)

    grid_size = math.ceil(d_flat_image.shape[0] / block_size[0])
    _histogram_kernel[grid_size, block_size[0]](d_flat_image, hist_figure)

    normalize(hist_figure, hist_figure)

    return hist_figure.ravel()

def getFigureForImage4(path):
    img = imread(path)

    d_img = cuda.to_device(img)
    d_gray = convert_rgb2gray_use_kernel(d_img)
    d_gray = gaussian_blur_kernel(d_gray, (3, 3), 0)
    gray = d_gray.copy_to_host()

    mask_img = applyCannyThreshold(gray, 12)

    mask_img = zipImageKernel(mask_img, 8, 8, 0.12, BLOCK_SIZE)
    mask_img = zipImageKernel(mask_img, 16, 16, 0.2, BLOCK_SIZE)
    
    mask_img = joinNeighborPixelKernel(mask_img, 8, 8, 3, 0.15, BLOCK_SIZE)
    mask_img = joinNeighborPixelKernel(mask_img, 16, 16, 3, 1 / 3, BLOCK_SIZE)

    grid_size = (math.ceil(img.shape[1] / BLOCK_SIZE[0]),
                        math.ceil(img.shape[0] / BLOCK_SIZE[1]))
    apply_mask_kernel[grid_size, BLOCK_SIZE](d_img, mask_img)
    
    img = d_img.copy_to_host(img)
    hist_figure = histogram_kernel(img, BLOCK_SIZE)

    hu_monents = cvHuMoments_kernel(d_gray)

    fig = np.concatenate((hist_figure, hu_monents))
    return fig