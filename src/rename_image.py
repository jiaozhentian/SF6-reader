import os
import cv2

def rename_image(path):
    for file in os.listdir(path):
        os.rename(os.path.join(path, file), os.path.join(path, '000'+str(file)))


if __name__ == "__main__":
    rename_image('./data/total_labeled/SF6_densimeter_0719/annotations/')
