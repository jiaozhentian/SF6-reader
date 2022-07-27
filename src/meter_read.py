import cv2
import numpy as np
from collections import Counter

imgfile = './data/test/test2.jpg'
pngfile = './data/results/test2.png'

if __name__ == '__main__':
    src_img = cv2.imread(imgfile)
    seg_map = cv2.imread(pngfile, 0)
    seg_map = cv2.resize(seg_map, (src_img.shape[1], src_img.shape[0]))
    # dilate the mask to close gaps
    kernel = np.ones((5, 5), np.uint8)
    seg_map = cv2.dilate(seg_map, kernel, iterations=1)
    count_result = Counter(seg_map.flatten())
    print(count_result)
    seg_map = seg_map * 100
    cv2.imshow('mask', seg_map)
    cv2.waitKey(0)
