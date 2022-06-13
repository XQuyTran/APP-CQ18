import cv2 as cv, numpy as np
from numba import njit, cuda, void, uint8
from feature_extract import (convert_rgb2gray, create_gaussian_filter,
                                                applyCannyThreshold)
import math

# phiên bản tự cài dặt
# --------------------------------------------------------------------
@cuda.jit
def apply_kernel_device(image, kernel, out):
    '''
    Hàm thực hiện phép tích chập tại một điểm ảnh với bộ lọc.
    Tại vùng biên, phần tử lân cận gần nhất được chọn làm phần tử đệm.
    Độ dời khi duyệt là 1.
    Chạy song song trên GPU.

    Đầu vào (numba.cuda.cudadrv.devicearray.DeviceNDArray):
    - image: ảnh đầu vào.
    - kernel: bộ lọc.
    - out: mảng lưu kết quả.
    '''
    out_c, out_r = cuda.grid(2)
    offset = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    last_row = image.shape[0] - 1
    last_col = image.shape[1] - 1

    out_pixel = 0.
    if out_r < image.shape[0] and out_c < image.shape[1]:
        for filter_r, r in enumerate(range(out_r - offset[0], out_r + offset[0] + 1)):
            for filter_c, c in enumerate(range(out_c - offset[1], out_c + offset[1] + 1)):
                in_r = min(max(0, r), last_row)
                in_c = min(max(0, c), last_col)
                out_pixel += image[in_r, in_c] * kernel[filter_r, filter_c]

        out[r, c] = np.uint8(math.floor(out_pixel))

def gaussian_blur_kernel(image, ksize, sigmaX, sigmaY=0, block_size=(32, 32)):
    '''
    Hàm làm mờ ảnh sử dụng bộ lọc Gauss.

    Đầu vào:
    - image (np.array): ảnh đầu vào.
    - ksize (tuple): kích thước bộ lọc Gauss, thể hiện số dòng, số cột.
    - sigmaX, sigmaY (float): độ lệch chuẩn cho giá trị bộ lọc theo chiều dọc và ngang.
    - block_size (tuple): kích thước khối chạy song song trên GPU.

    Đầu ra
    - blur (np.array): ảnh kết quả sau khi áp dụng bộ lọc
    '''
    d_kernel = cuda.to_device(create_gaussian_filter(ksize, sigmaX, sigmaY))
    d_image = cuda.to_device(image)
    d_out = cuda.device_array(d_image.shape, np.uint8)

    grid_size = (math.ceil(image.shape[1] / block_size[0]),
                        math.ceil(image.shape[0] / block_size[1]))
    apply_kernel_device[grid_size, block_size](d_image, d_kernel, d_out)

    return d_out.copy_to_host()

# @njit
# def zipImage(src, zip_x, zip_y, ratio):
#     '''
#     Hàm dùng để nén ảnh với threshold là ratio sẽ trả ra kết quả là ảnh với block tương ứng với zip x zip
#     và nếu block nào không đủ pixel yêu cầu thì là 0 còn ngược lại thì là 1.
#     '''
#     rs, cs = src.shape
#     zip_rs = math.ceil(rs / zip_y)
#     zip_cs = math.ceil(cs / zip_x)
#     for idx in range(0, zip_rs * zip_y, zip_y):
#         for jdx in range(0, zip_cs * zip_x, zip_x):
#             block_img = src[idx : idx + zip_y, jdx : jdx + zip_x]
#             num_pixel = np.sum(block_img > 0)
#             if num_pixel >= zip_x*zip_y * ratio:
#                 block_img[:,:] = 1
#             else:
#                 block_img[:,:] = 0

#     return src

@cuda.jit
def blk_count_nonzero(src, blk_dim_x, blk_dim_y, out, offset):
    '''
    Hàm xác định số phần tử khác 0 theo từng khối trong mảng đầu vào

    Tham số:
    - src (array): mảng đầu vào.
    - blk_dim_x, blk_dim_y (int): lần lượt là kích thước theo chiều rộng và cao của khối được xét.
    - out (array): mảng kết quả.
    '''
    c, r = cuda.grid(2)
    c += offset[0]
    r += offset[1]
    if c < src.shape[1] and r < src.shape[0] and src[r, c] > 0:
        cuda.atomic.add(out, (r // blk_dim_y, c // blk_dim_x), 1)

@cuda.jit
def apply_threshold(src, threshold, criteria_arr, zip_x, zip_y, offset, assign_zero):
    c, r = cuda.grid(2)
    c += offset[0]
    r += offset[1]
    if c < src.shape[1] and r < src.shape[0]:
        blk_r = r // zip_y
        blk_c = c // zip_x

        if criteria_arr[blk_r, blk_c] >= threshold:
            src[r, c] = 1
        elif assign_zero:
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
    blk_count_nonzero[grid_size, block_size](d_src, zip_x, zip_y, nonzero_count, (0, 0))
    apply_threshold[grid_size, block_size](d_src, zip_x*zip_y*ratio, nonzero_count, zip_x, zip_y, 
                                                                (0, 0), True)

    cuda.synchronize()
    return d_src


# @njit
# def joinNeighborPixel(src, zip_x, zip_y, mask_size, ratio):
#     '''
#     Hàm dùng để liên kết các ô xung quanh để lấp khuyết sẽ trả ra kết quả là một ma trận mask.
#     '''
#     rs, cs = src.shape
#     zip_rs = math.ceil(rs / zip_y)
#     zip_cs = math.ceil(cs / zip_x)
#     half = int(mask_size / 2)

#     dst = src.copy()
#     for idx in range(half , zip_rs - half):
#         for jdx in range(half, zip_cs - half):
#             start_row_mask = (idx - half ) * zip_y
#             end_row_mask = (idx + half + 1) * zip_y
#             start_col_mask = (jdx - half) * zip_x
#             end_col_mask = (jdx + half + 1) * zip_x

#             mask_block = src[start_row_mask : end_row_mask, start_col_mask : end_col_mask]
#             block_img = dst[idx * zip_y : (idx + 1) * zip_y, jdx * zip_x : (jdx + 1) * zip_x]
#             num_pixel = np.sum(mask_block > 0)
#             if num_pixel >= mask_size * mask_size * zip_x * zip_y * ratio:
#                 block_img[:,:] = 1
                
#     return dst

def joinNeighborPixelKernel(src, zip_x, zip_y, mask_size, ratio, block_size):
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
    nonzero_count = cuda.device_array((zip_rs*mask_size, zip_cs*mask_size), np.uint32)

    if not cuda.is_cuda_array(src):
        d_src = cuda.to_device(src)
    else:
        d_src = src

    zip_blk_x = zip_x * mask_size
    zip_blk_y = zip_y * mask_size

    grid_size = (math.ceil(src.shape[1] / block_size[0]), math.ceil(src.shape[0] / block_size[1]))

    for start_c in range(0, src.shape[1], zip_x):
        for start_r in range(0, src.shape[0], zip_y):
            grid_size = (math.ceil((src.shape[1] - start_c) / block_size[0]),
                                math.ceil((src.shape[0] - start_r) / block_size[1]))
            blk_count_nonzero[grid_size, block_size](d_src, zip_blk_x, zip_blk_y,
                                                                            nonzero_count, (start_c, start_r))


    threshold = mask_size * mask_size * zip_x * zip_y * ratio

    for start_c in range(0, src.shape[1], zip_x):
        for start_r in range(0, src.shape[0], zip_y):
            grid_size = (math.ceil((src.shape[1] - start_c) / block_size[0]),
                                math.ceil((src.shape[0] - start_r) / block_size[1]))
            apply_threshold[grid_size, block_size](d_src, threshold, nonzero_count, zip_blk_x, zip_blk_y,
                                                                        (start_c, start_r), False)

    return d_src


@njit
def cvMoments(img,m_,mu_,nu_):
    '''
    input:  ảnh 2D
        m_ mảng 3x3 : M ~ raw Moments
        mu_ 3x3: mu ~ central moments
        nu_ 3x3: nu ~ normalized central moments 
    output: nu_
    '''
    # tính m_
    for i in range(4):
        for j in range(4):
            temp=0
            if (i,j) not in [(1,3),(2,2),(2,3)]:
                for x in range(img.shape[0]):
                    for y in range(img.shape[1]):
                        temp=temp+img[x,y]*(x**i)*(y**j)
                m_[i,j]=temp
            if i==3:
                break

    # xbar ybar
    # print(f'dividend: {m_[0,0]}')
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
 
    return nu_

# HuMoments
@njit
def cvHuMoments(eta,hu_):
    '''
    eta: output của cvMoments
    hu_: Mảng 1D 7 giá trị 
    '''
    hu_[0] =  eta[2][0] + eta[0][2]
    hu_[1] = (eta[2][0] - eta[0][2])**2 + 4*eta[1][1]**2
    hu_[2] = (eta[3][0] - 3*eta[1][2])**2 + (3*eta[2][1] - eta[0][3])**2
    hu_[3] = (eta[3][0] + eta[1][2])**2 + (eta[2][1] + eta[0][3])**2
    hu_[4] = (eta[3][0] - 3*eta[1][2])*(eta[3][0] + eta[1][2])*((eta[3][0] + eta[1][2])**2 - 3*(eta[2][1] + eta[0][3])**2) +\
                (3*eta[2][1] - eta[0][3])*(eta[2][1] + eta[0][3])*(3*(eta[3][0] + eta[1][2])**2 - (eta[2][1] + eta[0][3])**2)

    hu_[5] = (eta[2][0] - eta[0][2])*((eta[3][0] + eta[1][2])**2 - (eta[2][1] + eta[0][3])**2) + \
          4*eta[1][1]*(eta[3][0] + eta[1][2])*(eta[2][1] + eta[0][3])

    hu_[6] = (3*eta[2][1] - eta[0][3])*(eta[2][1] + eta[0][3])*(3*(eta[3][0] + eta[1][2])**2-(eta[2][1] + eta[0][3])**2) -\
          (eta[3][0] - 3*eta[1][2])*(eta[1][2] + eta[0][3])*(3*(eta[3][0] + eta[1][2])**2-(eta[2][1] + eta[0][3])**2)

    return hu_

@njit
def compute_hist(img, hist):
    '''
    Color histogram: thống kê số lần xuất hiện các mức sáng trong ảnh với bins=8, 
                     phạm vi [0,255] cho mỗi kênh màu
    input: 
            img: numpy.ndarray with shape=(h, w, 3)
            hist: numpy.ndarray with shape=(8,8,8)
    '''
    h, w,d = img.shape[:3] 
    for i in range(h): 
        for j in range(w): 
            x,y,z=img[i][j][0]//32,img[i][j][1]//32,img[i][j][2]//32
            hist[x][y][z] =hist[x][y][z] + 1 
    return hist

# --------------------------------------------------------------------
#cài đặt sử dụng thư viện
def fd_histogram(image):
    bins = 8
    # convert the image to HSV color-space
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # compute the color histogram
    hist  = cv.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 255, 0, 255, 0, 255])

    # normalize the histogram
    cv.normalize(hist, hist)

    # return the histogram
    return hist.flatten()

def fd_hu_moments(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature

# cài đặt song song hóa
@cuda.jit(void(uint8[:, :, ::1], uint8[:, ::1]))
def convert_rgb2gray_kernel(in_pixels, out_pixels):
    c, r = cuda.grid(2)
    if r < out_pixels.shape[0] and c < out_pixels.shape[1]:
        out_pixels[r, c] = (in_pixels[r, c, 0] * 0.114 + 
                            in_pixels[r, c, 1] * 0.587 + 
                            in_pixels[r, c, 2] * 0.299)

BLOCK_SIZE = (32, 32)
def convert_rgb2gray_use_kernel(image):
    height, width = image.shape[:2]
    grid_size = (math.ceil(width / BLOCK_SIZE[0]),math.ceil(height / BLOCK_SIZE[1]))
    d_in_img = cuda.to_device(image)
    d_gray_img = cuda.device_array((height, width), dtype=np.uint8)
    convert_rgb2gray_kernel[grid_size, BLOCK_SIZE](d_in_img, d_gray_img)
    gray=d_gray_img.copy_to_host()
    return gray.astype(np.int32)

def fd_hu_moments3(image):
    image=convert_rgb2gray_use_kernel(image)
    image = image.astype(np.uint8)
    feature = cv.HuMoments(cv.moments(image)).flatten()
    return feature

def getFigureForImage3(path):
    img = cv.imread(path)

    gray = convert_rgb2gray_use_kernel(img)
    # gray = gray.astype(np.uint8)
    # gray = cv.GaussianBlur(gray, (3, 3), 0)
    gray = gaussian_blur_kernel(gray, (3, 3), 0)

    mask_img = applyCannyThreshold(gray, 12)

    mask_img = zipImageKernel(mask_img, 8, 8, 0.12, BLOCK_SIZE)
    mask_img = zipImageKernel(mask_img, 16, 16, 0.2, BLOCK_SIZE)
    
    # mask_img = joinNeighborPixel(mask_img.copy_to_host(), 8, 8, 3, 0.15)
    # mask_img = joinNeighborPixel(mask_img, 16, 16, 3, 1 / 3)
    mask_img = joinNeighborPixelKernel(mask_img, 8, 8, 3, 0.15, BLOCK_SIZE)
    mask_img = joinNeighborPixelKernel(mask_img, 16, 16, 3, 1 / 3, BLOCK_SIZE).copy_to_host()

    for chanel in range(3):
        img[:,:,chanel] = img[:,:,chanel] * mask_img

    hist_figure = fd_histogram(img).astype(np.float64)
    hu_monents = fd_hu_moments3(img)

    fig = np.concatenate((hist_figure, hu_monents))
    return fig

# test song song hóa

def compare_gray(path):
    img = cv.imread(path)
    height, width = img.shape[:2]

    gray_img = np.empty((height, width), dtype=img.dtype)

    gray1= convert_rgb2gray(img,gray_img)
    gray2= convert_rgb2gray_use_kernel(img)

    print('Convert rgb to grayscale error:',np.mean(np.abs(gray2- gray1)), '\n')
