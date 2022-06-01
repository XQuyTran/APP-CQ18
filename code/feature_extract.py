import cv2 as cv, numpy as np
from numba import njit
from math import exp

# phiên bản tự cài dặt
# --------------------------------------------------------------------
@njit
def create_gaussian_blur_kernel(shape, sigma):
    '''
    Hàm tạo bộ lọc Gaussian có kích thước shape x shape với các giá trị có độ lệch chuẩn sigma.

    Đầu vào:
    - shape (int): kích thước bộ lọc
    - sigma (float): độ lệch chuẩn.

    Đầu ra:
    - h  (np.array): bộ lọc Gaussian

    Tham khảo:
    https://stackoverflow.com/questions/8204645/implementing-gaussian-blur-how-to-calculate-convolution-matrix-kernel
    '''
    # numpy method
    # h1 = np.full(shape, np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2))
    # h2 = np.full(shape[::-1], np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2)).T

    # hg = np.exp(-(h1 ** 2 + h2 ** 2) / (2 * sigma**2))
    # return hg / np.sum(hg)

    h1 = np.empty(shape)
    h2 = np.empty(shape)

    for i, val in enumerate(range(-(shape[1] - 1) / 2, (shape[1] - 1) / 2)):
        h1[:, i] = val

    for i, val in enumerate(range(-(shape[0] - 1) / 2, (shape[0] - 1) / 2)):
        h2[i] = val

    hg = np.empty(shape)
    hg_sum = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            hg[i, j] = exp(-(h1[i, j]**2 + h2[i, j]**2) / (2 * sigma**2))
            hg_sum += hg[i, j]

    for i in range(shape[0]):
        for j in range(shape[1]):
            hg[i, j] /= hg_sum

    return hg

@njit
def gaussian_blur(image, kernel_shape, sigma):
    kernel = create_gaussian_blur_kernel(kernel_shape, sigma)
    start_id = kernel.shape[0] // 2
    last_row = image.shape[0] - 1
    last_col = image.shape[1] - 1

    out = np.empty_like(image)
    for r in range(image.shape[0]):
        r_start = r - start_id
        for c in range(image.shape[1]):
            c_start = c - start_id
            for filter_r in range(kernel_shape[0]):
                for filter_c in range(kernel_shape[1]):
                    in_r = min(max(0, r_start + filter_r), last_row)
                    in_c = min(max(0, c_start + filter_c), last_col)

                    out[r, c] += image[in_r, in_c] * kernel[in_r, in_c]

    return out

def applyCannyThreshold(frame, val):
    ratio = 1.2
    kernel_size = 3
    low_threshold = val
    img_blur = cv.GaussianBlur(frame, (3, 3), 0)
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = frame * (mask[:,:].astype(frame.dtype))
    return dst


@njit
def zipImage(src, zip_x, zip_y, ratio):
    '''
    Hàm dùng để nén ảnh với threshold là ratio sẽ trả ra kết quả là ảnh với block tương ứng với zip x zip
    và nếu block nào không đủ pixel yêu cầu thì là 0 còn ngược lại thì là 1.
    '''
    rs, cs = src.shape
    zip_rs = int(rs / zip_y)
    zip_cs = int(cs / zip_x)
    for idx in range(0, zip_rs * zip_y, zip_y):
        for jdx in range(0, zip_cs * zip_x, zip_x):
            block_img = src[idx : idx + zip_y, jdx : jdx + zip_x]
            num_pixel = np.sum(block_img > 0)
            if num_pixel >= zip_x*zip_y * ratio:
                block_img[:,:] = 1
            else:
                block_img[:,:] = 0

    return src

@njit
def joinNeiboorPixel(src, zip_x, zip_y, mask_size, ratio):
    '''
    Hàm dùng để liên kết các ô xung quanh để lấp khuyết sẽ trả ra kết quả là một ma trận mask.
    '''
    rs, cs = src.shape
    zip_rs = int(rs / zip_y)
    zip_cs = int(cs / zip_x)
    half = int(mask_size / 2)
    dst = src.copy()
    for idx in range(half , zip_rs - half):
        for jdx in range(half, zip_cs - half):
            start_row_mask = (idx - half ) * zip_y
            end_row_mask = (idx + half + 1) * zip_y
            start_col_mask = (jdx - half) * zip_x
            end_col_mask = (jdx + half + 1) * zip_x
            mask_block = src[start_row_mask : end_row_mask, start_col_mask : end_col_mask]
            block_img = dst[idx * zip_y : (idx + 1) * zip_y, jdx * zip_x : (jdx + 1) * zip_x]
            num_pixel = np.sum(mask_block > 0)
            if num_pixel >= mask_size * mask_size * zip_x * zip_y * ratio:
                block_img[:,:] = 1
    return dst


@njit
def convert_rgb2gray(in_pixels, out_pixels):
    '''
    Convert color image to grayscale image.
 
    in_pixels : numpy.ndarray with shape=(h, w, 3)
                h, w is height, width of image
                3 is colors with BGR (blue, green, red) order
        Input RGB image
    
    out_pixels : numpy.ndarray with shape=(h, w)
        Output image in grayscale
    '''
    for r in range(len(in_pixels)):
        for c in range(len(in_pixels[0])):
            out_pixels[r, c] = (in_pixels[r, c, 0] * 0.114 + 
                                in_pixels[r, c, 1] * 0.587 + 
                                in_pixels[r, c, 2] * 0.299)
    return out_pixels

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

def fd_histogram2(image, mask=None):
    bins = 8
    # convert the image to HSV color-space
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # compute the color histogram
    hist = np.zeros((8,8,8), np.float32) 
    hist  = compute_hist(image,hist)

    # normalize the histogram
    cv.normalize(hist, hist)

    # return the histogram
    return hist.flatten()

def getFigureForImage2(path):
    img = cv.imread(path)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    mask_img = applyCannyThreshold(gray, 12)
    mask_img = zipImage(mask_img, 8, 8, 0.12)
    mask_img = zipImage(mask_img, 16, 16, 0.2)
    mask_img = joinNeiboorPixel(mask_img, 8, 8, 3, 0.15)
    mask_img = joinNeiboorPixel(mask_img, 16, 16, 3, 1 / 3)

    for chanel in range(0, 3):
        img[:,:,chanel] = img[:,:,chanel] * mask_img

    hist_figure = fd_histogram2(img).astype(np.float64)

    #huMoments
    m=np.zeros((4, 4), dtype=np.float64)
    mu=np.zeros((4, 4), dtype=np.float64)
    nu=np.zeros((4, 4), dtype=np.float64)

    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    height, width = img.shape[:2]
    gray_img = np.empty((height, width), dtype=img.dtype)

    img=convert_rgb2gray(img,gray_img)
    nu2=cvMoments(img,m,mu,nu)

    hu=np.zeros(7,float) 
    hu_moments = cvHuMoments(nu2, hu)

    fig = np.concatenate((hist_figure, hu_moments))
    return fig

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


def getFigureForImage(path):
    img = cv.imread(path)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    mask_img = applyCannyThreshold(gray, 12)
    mask_img = zipImage(mask_img, 8, 8, 0.12)
    mask_img = zipImage(mask_img, 16, 16, 0.2)
    mask_img = joinNeiboorPixel(mask_img, 8, 8, 3, 0.15)
    mask_img = joinNeiboorPixel(mask_img, 16, 16, 3, 1 / 3)

    for chanel in range(3):
        img[:,:,chanel] = img[:,:,chanel] * mask_img

    hist_figure = fd_histogram(img).astype(np.float64)
    hu_monents = fd_hu_moments(img)

    fig = np.concatenate((hist_figure, hu_monents))
    return fig
