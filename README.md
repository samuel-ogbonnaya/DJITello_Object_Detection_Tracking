# Sportseye
Computer Vision Project for detecting and tracking sports objects e.g footballs

The aim of this project is to develop and deploy AI software deployed on the DJI Tello drone to enable it to automatically record football matches using Computer vision.

The intial application was developed using a generic YOLOv3 object detection algorithm and this has been integrated with multiple object tracking algorithms such as GOTURN, CRST, MOSSE etc.

The application was updated to utilise a custom faster R-CNN model, the model was trained using Tensorflow and a custom dataset to detect footballs for the initlal application. Labelimg was used for labelling the training and test dataset images.


