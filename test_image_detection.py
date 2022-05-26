#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries
import sys
import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Helper functions **_Show Image Helper Function_** This helper function is used to plot the image , with a given
# title has a default value of **_"Image"_** , we also pass to the function color map argument that it has default
# value of **_gray_**

def show_image(image, title="Image", color_map="gray"):
    plt.imshow(image, cmap=color_map)
    plt.title(title)
    plt.show()


# **_Region Of Interest Helper Functions_** We will crop part of the photo where we are have more interest in this
# region to determine the lane lines , the steps are the following we make a black image with all zeros ,
# then we have the vertices of ploy to draw on the black image , then making masking using python function
# `cv.bitwise_and()`

# In[4]:


def region_of_intereset(image, vertices):
    blank = np.zeros_like(image)
    cv.fillPoly(blank, vertices, color=255)
    cropped_region = cv.bitwise_and(image, blank)
    return cropped_region


# **_Fill Area Between Lane Lines Helper Function_** we here after detecting the lane lines we just color the area
# between them , by passing to the funtion the 4 vertices of the two lane lines

# In[5]:


def fill_area_between_lane_lines(image, polyfill_vertices):
    point1 = ()
    point2 = ()
    point3 = ()
    point4 = ()
    for i in range(0, 4):
        x1, y1 = polyfill_vertices[i]
        if i == 0:
            point1 = (x1, y1)
            continue
        elif i == 1:
            point2 = (x1, y1)
            continue
        elif i == 2:
            point3 = (x1, y1)
            continue
        elif i == 3:
            point4 = (x1, y1)
            continue
    polyfill_vertices_in_right_order = [point1, point2, point4, point3]
    cv.fillPoly(image, np.array([polyfill_vertices_in_right_order], np.int64), (92, 255, 92))


# **_Three Channel Image Conversion Helper Function_** here we take image of signle channel and convert it to 3
# channels in the purpose of concatenate it with other images in the debugging mode

def three_channle_image_conversion(source_image, converted_image):
    blank_image = np.zeros_like(source_image)
    blank_image[:, :, 0] = converted_image
    blank_image[:, :, 1] = converted_image
    blank_image[:, :, 2] = converted_image
    return blank_image


# **_Debuggind Mode Image Hepler Function_** we here concatenate all the image stages that it has passed through in
# the pipeline in one image

# > This function is activated only when Debugging Mode is set **_Activated_**

def debugging_mode_image(test_image, gray_test_image, RGB_test_image, hls_test_image, s, canny_test_image,
                         cropped_test_image):
    v_image1 = np.vstack([cv.resize(test_image, (320, 320)),
                          cv.resize(three_channle_image_conversion(test_image, gray_test_image), (320, 320))])
    h_image1 = np.hstack([cv.resize(RGB_test_image, (960, 400)), cv.resize(v_image1, (320, 400))])
    h_image2 = np.hstack(
        [cv.resize(hls_test_image, (320, 320)), cv.resize(three_channle_image_conversion(test_image, s), (320, 320))])
    h_image3 = np.hstack([cv.resize(three_channle_image_conversion(test_image, canny_test_image), (320, 320)),
                          cv.resize(three_channle_image_conversion(test_image, cropped_test_image), (320, 320))])
    h_image4 = np.hstack([cv.resize(h_image2, (640, 320)), cv.resize(h_image3, (640, 320))])
    debugging_image = np.vstack([cv.resize(h_image1, (1280, 400)), cv.resize(h_image4, (1280, 320))])
    return debugging_image


# Yolo Helper Function this is the yolo algorithm is used to detect cars on the roads, this algorithms uses neural
# networks, its version is YOLOv3-416
def yolo(image):
    weights_path = os.path.abspath("C:/Users/mahmo/computer_vision_project/yolov3.weights")
    config_path = os.path.abspath("C:/Users/mahmo/computer_vision_project/yolov3.cfg")
    labels_path = os.path.abspath("C:/Users/mahmo/computer_vision_project/coco.names")
    net = cv.dnn.readNetFromDarknet(config_path, weights_path)
    names = net.getLayerNames()
    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), crop=False, swapRB=False)
    net.setInput(blob)
    layers_output = net.forward(layers_names)
    boxes = []
    confidences = []
    classIDs = []
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.85:
                box = detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype("int")
                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
    if len(idxs) == 0:
        return image
    else:
        labels = open(labels_path).read().strip().split("\n")
        for i in idxs.flatten():
            (x, y) = [boxes[i][0], boxes[i][1]]
            (w, h) = [boxes[i][2], boxes[i][3]]
            cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.putText(image, "{}:{}".format(labels[classIDs[i]], confidences[i]), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 0, 0), 2)
        return image


# **_Make Coordinates Helper Function_**

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    # slope, intercept = line_parameters
    height = image.shape[0]
    y1 = height
    y2 = int(y1 * 0.65)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


# **_Average Slope Intercept Helper Functions_**

# In[18]:


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])


# **_Display Lines Helper Function_**

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            #             print (x1, ' ', y1, ' ', x2, ' ', y2) ################3
            cv.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255),
                    4)  # cv.line(line_image,(x1,y1),(x2,y2),(0,0,255),4)
            # cv.line(line_image,(x1.astype(np.int64),y1.astype(np.int64)),(x2.astype(np.int64),y2.astype(np.int64)),
            # (0,0,255),4)     # cv.line(line_image,(x1,y1),(x2,y2),(0,0,255),4)
    return line_image


# ### Offset of the car `compute_offset(image, lines)`: is used to get the offset and direction of the car. The car
# center is in the middle of the image. the lane center is calculated using starting points of the lane in the x-axis
# corresbonding to the height of the image in the y-axis.

def compute_offset(image, lines):
    x = []
    line_image = np.zeros_like(image)
    car_center = image.shape[1] / 2
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            x.append(x1)
    lane_center = (x[0] + x[1]) / 2
    offset = abs(lane_center - car_center) * (3.7 / 700)
    if x[0] > (1280 - x[1]):
        direction = 'left'
    elif x[0] < (1280 - x[1]):
        direction = 'right'
    else:
        direction = 'center'
    return offset, direction


def frame_process(test_image, debug_mode):
    RGB_test_image = np.copy(test_image)
    gray_test_image = cv.cvtColor(RGB_test_image, cv.COLOR_RGB2GRAY)
    image_width = RGB_test_image.shape[1]
    image_height = RGB_test_image.shape[0]
    hls_test_image = cv.cvtColor(RGB_test_image, cv.COLOR_RGB2HLS)
    h, l, s = cv.split(hls_test_image)
    ret, thresh1 = cv.threshold(gray_test_image, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    combined = cv.addWeighted(thresh1, 0.3, s, 0.7, 0)
    blurred_test_image = cv.GaussianBlur(combined, (5, 5), 0)
    canny_test_image = cv.Canny(blurred_test_image, 70, 255)
    vertices_of_region_of_interest = [(100, 660), (image_width / 2, image_height / 2 + 65), (1200, 660)]
    cropped_test_image = region_of_intereset(canny_test_image, np.array([vertices_of_region_of_interest], np.int64))
    hough_lines = cv.HoughLinesP(cropped_test_image, 1, np.pi / 180, 20, minLineLength=2, maxLineGap=50)
    polyfill_vertices = []
    averaged_lines = average_slope_intercept(RGB_test_image, hough_lines)
    line_image = display_lines(RGB_test_image, averaged_lines)
    offset, direction = compute_offset(RGB_test_image, averaged_lines)
    if averaged_lines is not None:
        for line in averaged_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(RGB_test_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=4)
            # cv.line(RGB_test_image , (x1.astype(np.int64) , y1.astype(np.int64)) , (x2.astype(np.int64) ,y2.astype(
            # np.int64)) , (0 , 0 , 255) , thickness = 4)
            polyfill_vertices.append((x2, y2))
            polyfill_vertices.append((x1, y1))
    fill_area_between_lane_lines(RGB_test_image, polyfill_vertices)
    debugging_image = debugging_mode_image(test_image, gray_test_image, RGB_test_image, hls_test_image, s,
                                           canny_test_image, cropped_test_image)
    if debug_mode == 0:
        cv.putText(img=RGB_test_image, text="Vehicle is {0:.2f} m {1} of center".format(offset, direction),
                   org=(50, 50), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(255, 255, 255), thickness=3)
        return yolo(RGB_test_image)
    elif debug_mode == 1:
        cv.putText(img=debugging_image, text="Vehicle is {0:.2f} m {1} of center".format(offset, direction),
                   org=(50, 50), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(255, 255, 255), thickness=3)
        return debugging_image


# sys.argv[1] ==> input image/video path
# sys.argv[2] ==> output image/video path
# sys.argv[3] ==> the input file is image/video: 0 ==> image, 1 ==> video
# sys.argv[4] ==> debug mode: 0 ==> deactivate, 1 ==> activate


# ## Testing the previous functions on the test images
def main():
    debug_mode = int(sys.argv[4])
    if int(sys.argv[3]) == 0:  # image
        # ### Read the test image
        test_image = cv.imread(sys.argv[1])
        '''cv.imshow('test image', test_image)
        cv.waitKey(0)'''
        print(debug_mode)
        # show_image(test_image)
        # cv.waitKey(0)
        # print(os.getcwd())
        outputImage = frame_process(test_image, debug_mode)
        show_image(outputImage)
        directory = sys.argv[2]
        imageName = 'output.jpg'
        os.chdir(directory)
        isSaved = cv.imwrite(imageName, outputImage)
        if isSaved:
            print('Successfully saved')
    elif int(sys.argv[3]) == 1:  # video
        cap = cv.VideoCapture(sys.argv[1])
        fps = cap.get(cv.CAP_PROP_FPS)
        # print(fps)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # print(frame_height, frame_width)
        directory = sys.argv[2]
        os.chdir(directory)
        videoName = 'output.mp4'
        frame_size = (frame_width, frame_height)
        outputVideo = cv.VideoWriter(videoName, cv.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        # paused = False

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            processed_frame = frame_process(frame, debug_mode)
            outputVideo.write(processed_frame)
            print('processing video ...')

            cv.imshow("result", processed_frame)
            if cv.waitKey(1) == ord('q'):
                break
        print('Successfully saved')
        cap.release()
        outputVideo.release()


if __name__ == "__main__":
    main()
