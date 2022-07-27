# meter-reader

These example codes are used to read meter data from a single meter image, especially for the SF6 meter.

# Introduction

In this project, we suppose that the meter has been detected and the meter image is ready to be read.

# Algorithm

1. The semantic segmentation is used to extract the pointer and the meter scale first.

2. The morphological manipulation has been implemented to get the hole meter scale.

3. To extract the meter scale precisely, the ellipse fitting is used.

4. The corner detection is used to get the start & end points of the meter scale.

5. At last, the coordinates of the start point, end point and pointer point are used to calculate the included angle with cosine law.

6. If you want to read a meter from a large image, you need add detection model and classification model. What is more, you need add OCR model to extract the meter reading text automatically.

# Note

1. We provide these codes just for experiment, not for production.

2. We will never provide the annotated data, please do not contact us for data.

3. The semantic segmentation model and the core code is not provided, if you want to use it, please contact us.

# Experiments



![Figure_1.png](F:\My_app\meter-reader\data\experiment_results\Figure_1.png)



![Figure_2.png](F:\My_app\meter-reader\data\experiment_results\Figure_2.png)



![Figure_3.png](F:\My_app\meter-reader\data\experiment_results\Figure_3.png)



![Figure_4.jpg](F:\My_app\meter-reader\data\experiment_results\Figure_4.jpg)