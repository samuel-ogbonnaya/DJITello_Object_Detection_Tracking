# Computer Vision Project for detecting and tracking footballs

The aim of this project is to develop and deploy AI software on the DJI Tello drone to enable it to automatically detect and track footballs

+ The initial application was developed using a generic YOLOv3 object detection algorithm and this has been integrated with multiple object tracking algorithms such as GOTURN, CRST, MOSSE etc.
+ The application was updated to utilise a custom faster R-CNN model, the model was trained using Tensorflow Object Detction API and a custom dataset to detect footballs for the initlal application. 
+ Labelimg was used for labelling the training and test dataset images.
+ I also added an additional function in the visualisation_utils.py scripts from the tensorflow object detection API
+ The detector algorithms has been integrated with tracking algorithm and a controller algorithm created for altering the drone movements accordingly.

Tested with Python 3.6, but it also may be compatabile with other versions.

## Requirements
- Software
  - TensorFlow Object Detection API
  - Python >= 3.6
  - OpenCV 4
  - [Tello SDK](https://github.com/damiafuentes/DJITelloPy)
- Hardware
    - [DJI Tello Drone](https://store.dji.com/uk/shop/tello-series)

## Usage
- Install and train your object detection model using instruction from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#)
- Add in the following function into the visualisation_utils.py in the utils folder of the TF API:
```
def return_coordinates(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    keypoint_scores=None,
    keypoint_edges=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_boxes=False,
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):

  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_keypoint_scores_map = collections.defaultdict(list)
  box_to_score_map = {} # update
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(boxes.shape[0]):
    if max_boxes_to_draw == len(box_to_color_map):
      break
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if keypoint_scores is not None:
        box_to_keypoint_scores_map[box].extend(keypoint_scores[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in six.viewkeys(category_index):
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(round(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, round(100*scores[i]))
        if not skip_track_ids and track_ids is not None:
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
        box_to_display_str_map[box].append(display_str)
        box_to_score_map[box] = scores[i]  # update
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # updated below
  # Draw all boxes onto image.
  coordinates_list = []
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    height, width, channels = image.shape
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    coordinates_list = [[ymin, ymax, xmin, xmax], [(box_to_score_map[box] * 100)]]

  return coordinates_list
```
- Connect to the a tello drone via wifi
- Ensure the ball is in front of the tello drone camera before take-off
- Run the main.py script

### Horizontal Axis Demo
 ![Drone Tracking on horizontal axis](demo/demo.gif)
 
### Vertical Axis Demo
 ![Drone Tracking on vertical axis](demo/demo_v1.gif)

## Limitations
- The camera on the Tello Drone is fixed and therefore the drone cannot see underneath itself. This limits the field of vision of tracking.
- There is no access to drone control during takeoff, therefore the user has to ensure the ball is still in frame during the take-off routine to ensure tracking is successfully intialised.

## Next Steps
- The next key step is obtaining better hardware i.e. Drone with camera gimbal for tilting.
- Fail safes etc will have to be incorporated into the application
- See this [project](https://github.com/samuel-ogbonnaya/ParrotAnafi_ComputerVision)

## Resources
Thanks to Damia Fuentes for his [Tello SDK](https://github.com/damiafuentes/DJITelloPy)
