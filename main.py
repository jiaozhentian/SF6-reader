import cv2
import argparse
import collections
import numpy as np
from __init__ import *

from src import inference, visible

METER_SCALE = [-0.1, 0.9]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./data/test/demo1.jpg")
    parser.add_argument("--model", type=str, default="./saved_models/*****")
    parser.add_argument("--mask_path", type=str, default="./temp/prediction_mask.png")
    return parser

def image_segmentation(image_path, mask_path, model):
    inference_server = inference.InferenceServer(model_path=model)
    prediction_mask = inference_server(image_path)
    cv2.imwrite(mask_path, prediction_mask.astype(np.uint8))
    prediction_mask = cv2.imread(mask_path, 0)
    return prediction_mask

def mask_visible(image_path, mask_path):
    image_src = cv2.imread(image_path, 1)
    image_src = image_src[:,:,::-1]
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (image_src.shape[1], image_src.shape[0]), interpolation=cv2.INTER_NEAREST)
    visible.vis_segmentation(image_src, mask)

def image_denoising(image):
    image = cv2.medianBlur(image, 5)
    return image

def cosine_law(a, b, c):
    return np.arccos((a**2 + b**2 - c**2) / (2 * a * b))

if __name__ == '__main__':
    args = make_parser().parse_args()
    image_path = args.image_path
    model_path = args.model
    mask_path = args.mask_path
    image_src = cv2.imread(image_path, 1)
    mask_borad = np.zeros(image_src.shape[:2], dtype=np.uint8)
    prediction_mask_origin = image_segmentation(image_path, mask_path, model_path)
    mask_visible(image_path, mask_path)
    
    # 取出指针的像素
    prediction_mask = prediction_mask_origin.copy()
    prediction_mask[prediction_mask == 2] = 0
    prediction_mask_point = image_denoising(prediction_mask)
    prediction_mask_point = cv2.resize(prediction_mask_point, (image_src.shape[1], image_src.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('./temp/point_mask.png', prediction_mask_point.astype(np.uint8))
    mask_visible(image_path, './temp/point_mask.png')

    # 取出表盘的像素
    prediction_mask = prediction_mask_origin.copy()
    prediction_mask[prediction_mask == 1] = 0
    print(collections.Counter(prediction_mask.flatten()))
    prediction_mask = image_denoising(prediction_mask)
    cv2.imwrite('./temp/scale_mask.png', prediction_mask.astype(np.uint8))
    mask_visible(image_path, './temp/scale_mask.png')

    # 处理mask
    prediction_mask = cv2.resize(prediction_mask, (image_src.shape[1], image_src.shape[0]), interpolation=cv2.INTER_NEAREST)/2
    # connect all regions
    prediction_mask = cv2.morphologyEx(prediction_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
    # prediction_mask = cv2.dilate(prediction_mask, np.ones((20, 20), np.uint8), iterations=1)
    # find contours
    contours, hierarchy = cv2.findContours(np.uint8(prediction_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find circles
    circles = []
    for contour in contours:
        rrt = cv2.fitEllipse(contour)
        # if the a of ellipse is too small, abandon it
        if rrt[1][0] < 0.3 * image_src.shape[1] or rrt[1][1] < 0.3 * image_src.shape[0]:
            continue
        circles.append(rrt)
    # draw circles
    # the circle heart is the most important in this step
    for circle in circles:
        cv2.ellipse(image_src, circle, (0, 0, 255), 1)
        cv2.ellipse(mask_borad, circle, (255, 255, 255), 1)
        cv2.circle(image_src, (np.int(circle[0][0]), np.int(circle[0][1])), 4, (0, 0, 255), -1, 8, 0)
        cv2.circle(prediction_mask, (np.int(circle[0][0]), np.int(circle[0][1])), 4, (225, 225, 255), -1, 8, 0)
   
    mask_borad = np.multiply(mask_borad, prediction_mask)
    # get the corner of the mask_borad
    mask_borad = cv2.dilate(mask_borad, np.ones((3, 3), np.uint8), iterations=1)
    '''
    cv2.imshow('mask_borad', mask_borad)
    cv2.imshow('prediction_mask', prediction_mask)
    cv2.imshow('image_src', image_src)
    cv2.waitKey(0)
    '''
    # detect the corner of the arc
    corners = cv2.goodFeaturesToTrack(np.uint8(mask_borad), 100, 0.5, 10)
    # find the top 2 corners of the y axis
    corners = np.int0(corners)
    corners = np.squeeze(corners)
    # get the start & end points of the dial
    corners = corners[np.argsort(corners[:, 1])[-2:]]
    corners = corners[np.argsort(corners[:, 0])]
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image_src, (x, y), 3, (0, 255, 255), -1, 10, 0)
        cv2.circle(prediction_mask, (x, y), 3, (255, 255, 255), -1, 10, 0)
        # draw line between corners and meter heart
        cv2.line(image_src, (x, y), (np.int(circle[0][0]), np.int(circle[0][1])), (0, 255, 255), 2)

    # get the min area rect of the pooint segmentation
    image_point = cv2.imread('./temp/point_mask.png', 0)
    cnts, hierarchy = cv2.findContours(image_point.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # delete the contour with area less than 60
    cnts = [c for c in cnts if cv2.contourArea(c) > 60]
    point_squeeze = np.squeeze(cnts[0])
    rect_point = cv2.minAreaRect(point_squeeze)
    point_heart = rect_point[0]
    cv2.circle(image_src, (np.int(point_heart[0]), np.int(point_heart[1])), 3, (255, 0, 255), -1, 10, 0)
    
    # draw line between corners and meter heart
    cv2.line(image_src, (np.int(point_heart[0]), np.int(point_heart[1])), (np.int(circle[0][0]), np.int(circle[0][1])), (255, 0, 255), 2)
    # caculate the angle between circle, the vertex is the meter heart
    a = np.sqrt((corners[0][0] - circle[0][0]) ** 2 + (corners[0][1] - circle[0][1]) ** 2)
    b = np.sqrt((corners[1][0] - circle[0][0]) ** 2 + (corners[1][1] - circle[0][1]) ** 2)
    c = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    acute_angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    angle = 2*np.pi - acute_angle
    # calculate the angle between corner and start circle, the vertex is the meter heart
    a = np.sqrt((corners[0][0] - circle[0][0]) ** 2 + (corners[0][1] - circle[0][1]) ** 2)
    b = np.sqrt((point_heart[0] - circle[0][0]) ** 2 + (point_heart[1] - circle[0][1]) ** 2)
    c = np.sqrt((point_heart[0] - corners[0][0]) ** 2 + (point_heart[1] - corners[0][1]) ** 2)
    point_angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    # calculate the straight line equation of the start corner to the meter heart
    k_0 = (corners[0][1] - circle[0][1]) / (corners[0][0] - circle[0][0])
    b_0 = corners[0][1] - k_0 * corners[0][0]
    k_1 = (corners[1][1] - circle[0][1]) / (corners[1][0] - circle[0][0])
    b_1 = corners[1][1] - k_1 * corners[1][0]
    # judge the point in the left or right of the start corner line
    if k_0 * point_heart[0] + b_0 < point_heart[1] and k_1 * point_heart[0] + b_1 > point_heart[1]:
        point_angle = 2*np.pi - point_angle

    # calculate the proportion of the point_angle in the dial
    point_proportion = point_angle / angle
    print(point_proportion)
    meter_value = point_proportion * (max(METER_SCALE) - min(METER_SCALE)) + min(METER_SCALE)
    # 图像畸变数值补偿
    if meter_value > 0.1 and meter_value < 0.3:
        meter_value += 0.03
    elif meter_value > 0.55 and meter_value < 0.7:
        meter_value -= 0.03
    print("The value of the meter is: {}".format(meter_value))
    # draw the meter value
    cv2.putText(image_src, str(round(meter_value, 2)), (np.int(point_heart[0]), np.int(point_heart[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    

    cv2.imshow("origin", image_src)
    cv2.imshow("mask", prediction_mask)
    cv2.imshow("mask_borad", mask_borad)
    cv2.waitKey(0)

