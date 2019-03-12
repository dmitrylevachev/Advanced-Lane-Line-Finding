import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2

class Line():
    def __init__(self, fit=(0,0,0), history_len=1):
        self.fit = fit
        self.history = []
        self.history_len = history_len
        self.source_points = ([], [])
        self.detected = False

    def lost_line(self):
        self.detected = False
        self.history = []

    def update(self, y, x):
        self.source_points = (y, x)
        fit = np.polyfit(y, x, 2)
        if len(self.history) > self.history_len:
            del self.history[0]
        history_copy = self.history.copy()
        history_copy.append(fit)
        self.fit = np.mean(history_copy, axis=0)
        self.history.append(self.fit)
    

    def get_points(self, img):
        left_fit = self.fit
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        left_fitx[(left_fitx > img.shape[1]) | (left_fitx < 0)] = 0
        return ploty, left_fitx

    def get_line_zone(self, img, margin = 100, color = (0,255,0)):
        ploty, left_fitx = self.get_points(img)

        left_line = np.array([np.transpose(np.vstack([np.int32(left_fitx - margin), np.int32(ploty)]))])
        right_line = np.array([np.flipud(np.transpose(np.vstack([np.int32(left_fitx + margin), np.int32(ploty)])))])

        lines = np.hstack((left_line, right_line))
        line_zone = np.zeros_like(img)
        cv2.fillPoly(line_zone, lines, color)
        return line_zone

    def curvature_radius(self, y):
        A = self.fit[0]
        B = self.fit[1]
        ym_per_pix = 30/720
        y_real =  y * ym_per_pix
        return (1 + (2 * A * y_real + B)**2)**(1.5) / np.absolute(2 * A)
        

    def draw(self, img, color=(0, 255, 255), margin=1):
        ploty, left_fitx = self.get_points(img)
        left_line = np.array([np.transpose(np.vstack([np.int32(left_fitx), np.int32(ploty)]))])
        left_line_margin = np.array([np.flipud(np.transpose(np.vstack([np.int32(left_fitx) - margin, np.int32(ploty)])))])
        left_margin = np.hstack((left_line_margin, left_line))
        lane_lines = np.zeros_like(img)
        cv2.fillPoly(lane_lines, left_margin, color)
        return lane_lines

    def drqw_with_history(self, img, color=(0, 255, 255), margin=1):
        lines = np.zeros_like(img)
        for fit in self.history:
            line = Line(fit)
            lines += line.draw(img, color=(255, 255, 0), margin = 2)
        main_line = self.draw(img, color=(0, 0, 255), margin = 2)
        lines[main_line != 0] = 0
        lines += main_line
        return lines
    
    def draw_source_points(self, img, color=(0, 0, 255)):
        new_image = img.copy()
        new_image[self.source_points[0], self.source_points[1]] = color
        return new_image

def calculate_diff_from_center(img, detected_lines):
    line2_bottom = detected_lines[1].get_points(img)[1][img.shape[0] - 1]   
    line1_bottom = detected_lines[0].get_points(img)[1][img.shape[0] - 1]

    lane_center = line1_bottom + (line2_bottom - line1_bottom) // 2
    car_center = img.shape[1] // 2

    diff = lane_center - car_center
    x_real =  3.7/700
    diff *= x_real
    return diff


def estimate_basises(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    mid_point = np.int(histogram.shape[0]//2)
    left_base = np.argmax(histogram[:mid_point])
    right_base = np.argmax(histogram[mid_point:]) + mid_point
    return (left_base, right_base)


def find_with_sliding_windows(img, base, margin = 50, windows_num = 10):
    minpix = 50

    nonzero = np.nonzero(img[:,:,0])
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    
    hight = img.shape[0]//windows_num

    left_current = base
    left_lane_indexes = []
            
    for win_n in range(windows_num):
        win_low_y = img.shape[0] - (win_n + 1) * hight
        win_high_y = img.shape[0] - win_n * hight

        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin

        cv2.rectangle(img,(win_xleft_low,win_low_y),(win_xleft_high, win_high_y),(0,255,0),2)
                
        good_left_indexes = ((nonzeroy >= win_low_y) & (nonzeroy < win_high_y) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        if len(good_left_indexes) > minpix:
            left_current = np.int(np.mean(nonzerox[good_left_indexes]))
                        
                
        left_lane_indexes.append(good_left_indexes)

    left_lane_inds = np.concatenate(left_lane_indexes)

    return (nonzerox[left_lane_inds], nonzeroy[left_lane_inds])

def find_within_line_range(img, line, margin = 50):
    nonzero = np.nonzero(img[:,:,0])
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    right_fit = line.fit

    right_lane_inds = ((nonzerox >= (right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2] - margin)) \
                    & (nonzerox < (right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2] + margin))).nonzero()[0]
    
    return (nonzerox[right_lane_inds], nonzeroy[right_lane_inds])

def detect_line(binary, lines):
    binary = np.dstack((binary,binary,binary))
    left_base, right_base = estimate_basises(binary[:,:,0])

    if not lines[0].detected:
        leftx, lefty = find_with_sliding_windows(binary, left_base)
    else:
        leftx, lefty = find_within_line_range(binary, lines[0])

    if not lines[1].detected:
        rightx, righty = find_with_sliding_windows(binary, right_base)
    else:
        rightx, righty = find_within_line_range(binary, lines[1])

    if not lines[0].detected or not lines[1].detected:
        if lefty != [] and leftx != [] and righty != [] and rightx != []:
            line1 = Line(np.polyfit(lefty, leftx, 2))
            line2 = Line(np.polyfit(righty, rightx, 2))
            lines = (line1, line2)
            lines[0].detected = True
            lines[1].detected = True
    else:
        if lefty != [] and leftx != [] and 2000 < len(leftx):
            lines[0].update(lefty, leftx)
            lines[0].detected = True
        else:
            lines[0].lost_line()

        if  righty != [] and rightx != [] and 2000 < len(rightx):
            lines[1].update(righty, rightx)
            lines[1].detected = True
        else:
            lines[1].lost_line()
                        
    return lines


    