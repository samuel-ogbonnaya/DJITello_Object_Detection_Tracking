# Computer Vision Project for detecting and tracking footballs

The aim of this project is to develop and deploy AI software on the DJI Tello drone to enable it to automatically detect and track footballs

The intial application was developed using a generic YOLOv3 object detection algorithm and this has been integrated with multiple object tracking algorithms such as GOTURN, CRST, MOSSE etc.

The application was updated to utilise a custom faster R-CNN model, the model was trained using Tensorflow Object Detction API and a custom dataset to detect footballs for the initlal application. Labelimg was used for labelling the training and test dataset images.

The detector algorithms has been integrated with tracking algorithm and a controller algorithm created for altering the drone movements accordingly.

Tested with Python 3.6, but it also may be compatabile with other versions.

## Requirements
- TensorFlow Object Detection API
- Python >= 3.6
- OpenCV 4
- [DJI Tello Drone](https://store.dji.com/uk/shop/tello-series)

## Usage
- Install and train you object detection model using instruction from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#)
- Connect to the a tello drone via wifi
- Ensure the ball is in front of the tello drone camera before take-off
- Run the main.py script

## Demo

## Limtiations
- The camera on the Tello Drone is fixed and therefore the drone cannot see underneath itself. This limits the field of vision of tracking.
- There is no access to drone control during takeoff, therefore the user has to ensure the ball is still in frame during the take-off routine to ensure tracking is successfully intialised.

## Next Steps
- The next key step is obtaining better hardware i.e. Drone with camera gimbal for tilting.
- Fail safes etc will have to be incorporated into the application
- See this [project](link)

## Author
Sam Ogbonnaya
Also thanks to Damia Fuentes for his [Tello SDK](https://github.com/damiafuentes/DJITelloPy)
