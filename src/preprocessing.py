import numpy as np
import cv2

def create_sobel_mask(img, thresh_min = 0, thresh_max = 0.09):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    arcdir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(gray)
    binary[(arcdir >= thresh_min) & (arcdir <= thresh_max)] = 1
    return binary

def create_sobel_max_mask(img, orient=(1,0), thresh_min = 25, thresh_max =255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, orient[0], orient[1], ksize=3)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def create_sobel_mag_mask(img, thresh_min = 25, thresh_max = 255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobelmag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*sobelmag/np.max(sobelmag))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sbinary

def create_lab_mask(img, thresh_min = 190, thresh_max = 255):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab_img[:,:,2]
    binary_l = np.zeros_like(l_channel)
    binary_l[ (l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1
    return binary_l

def create_hls_mask(img, thresh_min = 220, thresh_max = 255):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls_img[:,:,1]
    binary_s = np.zeros_like(s_channel)
    binary_s[ (s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return binary_s

def create_hls_s_mask(img, thresh_min = 160, thresh_max = 255):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls_img[:,:,2]
    binary_s = np.zeros_like(s_channel)
    binary_s[ (s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return binary_s

def preprocess_img(img):
    img = cv2.GaussianBlur(img,(5,5), 0)
    sobel_mask = create_sobel_mask(img)
    sobel_mask_mag = create_sobel_mag_mask(img)
    sobel_x = create_sobel_max_mask(img, (1,0))
    sobel_y = create_sobel_max_mask(img, (0,1))
    lab_mask = create_lab_mask(img)
    hls_mask = create_hls_mask(img)
    hls_s_mask = create_hls_s_mask(img)
    tresholded_img = np.zeros_like(sobel_mask)
    tresholded_img[ (((sobel_mask == 1) & (sobel_mask_mag == 1) ) | (sobel_x == 1) & (sobel_y == 1)  | ((hls_s_mask == 1) | (lab_mask == 1) | (hls_mask == 1)))] = 255
    return tresholded_img