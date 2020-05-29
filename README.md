# Requirements
TensorFlow Object Detection API

Python 3.6.10

OpenCV 4

DJI Tello Drone

# Sportseye
Computer Vision Project for detecting and tracking sports objects e.g footballs

The aim of this project is to develop and deploy AI software deployed on the DJI Tello drone to enable it to automatically record football matches using Computer vision.

The intial application was developed using a generic YOLOv3 object detection algorithm and this has been integrated with multiple object tracking algorithms such as GOTURN, CRST, MOSSE etc.

The application was updated to utilise a custom faster R-CNN model, the model was trained using Tensorflow Object Detction API and a custom dataset to detect footballs for the initlal application. Labelimg was used for labelling the training and test dataset images.

The detector algorithms has been integrated with tracking algorithm and a controller algorithm created for altering the drone movements accordingly.

Tested with Python 3.6, but it also may be compatabile with other versions.

# Limtiations
The camera on the Tello Drone is fixed and therefore the drone cannot see underneath itself. This limits the field of vision of tracking.

There is no access to drone control during takeoff, therefore the user has to ensure the ball is still in frame during the take-off routin to ensure tacking is successfully intialised.

# Next Steps
There next key step is obtaining better hardware i.e. Drone with camera gimbal for tilting.
Faile safe etc will have to be incorporated into the application

# Author
Sam Ogbonnaya

Tello SDK utilised:https://github.com/damiafuentes/DJITelloPy

