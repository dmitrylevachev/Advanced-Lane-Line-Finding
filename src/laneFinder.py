import numpy as np
import os
import cv2
import detection as det

import preprocessing as prep

import argparse
import camera as cam


def prepare_frame(frame, camera):
    undist = camera.undistort(frame)
    transform_points = cam.get_transform_points(frame)
    warped_img = camera.warpPerspective(undist, transform_points['src'], transform_points['dst'])
    tresholded_img = prep.preprocess_img(warped_img)
    return tresholded_img

def combine_images(frame, green_zone, lane_lines, binary):
    transform_points = cam.get_transform_points(frame)
    green_zone = camera.warpPerspective(green_zone, transform_points['dst'], transform_points['src'])
    lane_lines = camera.warpPerspective(lane_lines, transform_points['dst'], transform_points['src'])
    frame = cv2.addWeighted(frame, 1., green_zone, 0.4, 0)
    frame = cv2.addWeighted(frame, 1., lane_lines, 1., 0)
    frame = cv2.resize(frame, (0, 0), None, .5, .5)
    binary = cv2.resize(binary, (0, 0), None, .5, .5)
    out_img = np.hstack((frame, binary))
    return out_img

def draw_green_zone(img, lines):
    green_zone = np.zeros_like(img)
    left_line_points = lines[0].get_points(img)
    right_line_points = lines[1].get_points(img)

    left_line = np.array([np.transpose(np.vstack([np.int32(left_line_points[1]), np.int32(left_line_points[0])]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([np.int32(right_line_points[1]), np.int32(right_line_points[0])])))])

    lines = np.hstack((left_line, right_line))
    cv2.fillPoly(green_zone, lines, (0,255, 0))
    return green_zone

def draw_lane_lines(img, lines):
    lane_lines = np.zeros_like(img)
    lane_lines = lines[0].draw(img, margin=10) + lines[1].draw(img, margin=10)    
    return lane_lines

def draw_search_area(img, lines):
    line = lines[0].drqw_with_history(img, (255, 0, 0)) + lines[1].drqw_with_history(img, (0, 0, 255))

    img = lines[0].draw_source_points(img, (0,0,255))
    img = lines[1].draw_source_points(img, (255,0,0))

    line_nz = np.nonzero(line)

    img[line_nz[0], line_nz[1]] = 0
    img = img + line

    if lines[0].detected:
        left_zone = lines[0].get_line_zone(img, margin = 50)
    else:
        left_zone = lines[0].get_line_zone(img, margin = 50, color=(0,0,255))

    if lines[1].detected:
        right_zone = lines[1].get_line_zone(img, margin = 50)
    else:
        right_zone = lines[1].get_line_zone(img, margin = 50, color=(0,0,255))

    lines_zones = left_zone + right_zone
    img = cv2.addWeighted(img, 1., lines_zones, 0.4, 0)
    return img

def print_information_text(img, detected_lines):
    cv2.putText(img, "Curvature radius: {0:.2f} m".format(detected_lines[0].curvature_radius(img.shape[0])), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    if detected_lines[0].detected:
        cv2.putText(img, "Left line detected", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(img, "Left line lost", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if detected_lines[1].detected:
        cv2.putText(img, "Right line detected", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(img, "Right line lost", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return img

def process_frame(frame, detected_lines, camera):
    prepared_img = prepare_frame(frame, camera)
    detected_lines = det.detect_line(prepared_img, detected_lines)
    prepared_img = np.dstack((prepared_img, prepared_img, prepared_img))

    frame = print_information_text(frame, detected_lines)
    search_area = draw_search_area(prepared_img, detected_lines)
    green_zone = draw_green_zone(prepared_img, detected_lines)
    lane_lines = draw_lane_lines(prepared_img, detected_lines)
    combined_img = combine_images(frame, green_zone, lane_lines, search_area)
    return detected_lines, combined_img
            
def process_video(input_path, output_path, camera):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4)*0.5)))
    detected_lines = (det.Line(), det.Line())
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            detected_lines, combined_img = process_frame(frame, detected_lines, camera)
            cv2.imshow('frame', combined_img)
            writer.write(combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.release()
    cap.release()
    cv2.destroyAllWindows()


def process_image(input_path, output_path, camera):
    frame = cv2.imread(input_path)
    detected_lines = (det.Line(), det.Line())
    detected_lines, combined_img = process_frame(frame, detected_lines, camera)
    while(True):
        cv2.imshow('frame', combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite(output_path, combined_img)
    

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--vid", type=str, required=False, help="path to input video")
    ap.add_argument("-i", "--img", type=str, required=False, help="path to input image")
    ap.add_argument("-c", "--cal", type=str, required=False, default="../camera_cal", help="path to folder with calibration images")
    ap.add_argument("-o", "--o_dir", type=str, required=False, default="../output_images", help="path to output folder")
    ap.add_argument("-n", "--o_name", type=str, required=False, default="output.mp4", help="output file\'s name")
    args = vars(ap.parse_args())

    camera = cam.Camera()
    camera.calibrate(args['cal'])
    
    output_path = os.path.join(args['o_dir'], args['o_name'])

    if args['vid'] != None:
        print('vid')
        input_path = args['vid']
        process_video(input_path, output_path, camera)

    if args['img'] != None:
        print('img')
        input_path = args['img']
        process_image(input_path, output_path, camera)
