# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:46:19 2020

@author: isogb
"""


from djitellopy import Tello
import cv2
import numpy as np
import os
import sys
sys.path.append('C:/Users/isogb/Documents/Computer_Vision/TensorFlow/models/slim') # point to your tensorflow dir
sys.path.append('C:/Users/isogb/Documents/Computer_Vision/TensorFlow/models') # point ot your slim dir
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import time
          
class Sportseye:
    
    # Number of classes to detect
    NUM_CLASSES = 1
    MODEL_NAME = 'trained_inference_graphs'   
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    
    def __init__ (self, mode, retry_count = 2):
        
        self.mode = mode
        
        #Webcam 
        if self.mode == 1:
            self.cap = cv2.VideoCapture(0)
        
        #Drone
        elif self.mode == 2:
            self.tello = Tello()
            self.tello.connect()
            self.tello.streamon()
        
        
        self.retry_count = retry_count
        self.left_right = 0
        self.for_back = 0
        self.up_down = 0
        self.yaw = 0
        
    def TFdetector(self):    
        
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,'sportseye_v1_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,'training','label_map.pbtxt')
        
        # Load a (frozen) Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
             serialized_graph = fid.read()
             od_graph_def.ParseFromString(serialized_graph)
             tf.import_graph_def(od_graph_def, name='')
        
        # Loading label map
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        
        # Detection
        counter = 1 
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while counter < self.retry_count:  # set to a counter e.g try and detect ball maximum of three time, by checking of coordinate output is empty or not
                   
                    # Read frame from Webcam
                    if self.mode == 1:
                        self.ret, self.image_np = self.cap.read()
                    
                    #Read frame from Drone  
                    elif self.mode == 2:
                         self.image_np  = self.tello.get_frame_read().frame
                         self.ret = self.tello.get_frame_read().grabbed

                    if not self.ret:
                        break
                    
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    self.image_np_expanded = np.expand_dims(self.image_np, axis=0)
                    # Extract image tensor
                    self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detections
                    self.num_detections = self.detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    
                    # Actual detection.
                    (self.boxes, self.scores, self.classes, self.num_detections) = sess.run([self.boxes, self.scores, self.classes, self.num_detections],feed_dict={ self.image_tensor: self.image_np_expanded})
                    
                    self.coord = self.bounding_box_coordinates()
                    
                    if self.coord: # if object detected
                        return self.coord
                        break
                    else:
                        counter += 1
                    
    def display_bounding_box(self, image):
        self.image_np = image
        return vis_util.visualize_boxes_and_labels_on_image_array(
                        self.image_np,
                        np.squeeze(self.boxes),
                        np.squeeze(self.classes).astype(np.int32),
                        np.squeeze(self.scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.9)
     
    def bounding_box_coordinates(self):
        
        ## need try and exception here e.g if min(scores) < 90 then throw error
        self.coordinates = vis_util.return_coordinates(
                        self.image_np,
                        np.squeeze(self.boxes),
                        np.squeeze(self.classes).astype(np.int32),
                        np.squeeze(self.scores),
                        self.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.8)
        return self.coordinates

    def yolo_detector():
        pass
    
    
    def takeoff(self):
        print('Taking-off')
        self.tello.takeoff()
        return 
    
    def end_sequence(self):
        print('Ending')
        self.tello.streamoff()
        self.tello.land() 
        self.tello.end()  
        return
    
    def tracker_selection(self, selection):
        
        # Set up trackers.
        self.selection = selection
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        self.tracker_type = tracker_types[self.selection]
        
        if  self.tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        elif  self.tracker_type == 'MIL':
             self.tracker = cv2.TrackerMIL_create()
        elif  self.tracker_type == 'KCF':
             self.tracker = cv2.TrackerKCF_create()
        elif  self.tracker_type == 'TLD':
             self.tracker = cv2.TrackerTLD_create()
        elif  self.tracker_type == 'MEDIANFLOW':
             self.tracker = cv2.TrackerMedianFlow_create()
        elif  self.tracker_type == 'GOTURN':
             self.tracker= cv2.TrackerGOTURN_create()
        elif  self.tracker_type == 'MOSSE':
             self.tracker= cv2.TrackerMOSSE_create()
        elif  self.tracker_type == "CSRT":
             self.tracker = cv2.TrackerCSRT_create()
             
        return self.tracker
    
    def tracking_init(self):
        
        self.bbox = tuple(self.TFdetector()[0])
        
        # Converts Tensorflow object detector output into correct format for tracker intialisation
        x = self.bbox[2]
        y = self.bbox[0]
        w = self.bbox[3]-self.bbox[2]
        h = self.bbox[1]-self.bbox[0]
        
        self.bounding_box = (x, y, w, h)
        self.tracker.init(self.image_np, self.bounding_box)
        
        print('First detection Bbox: {0}'.format(self.bounding_box))
        
        # Update tracker
        self.tracking, self.bbox = self.tracker.update(self.image_np)
        print('Tracking Status: {0}'.format(self.tracking))
        return self.tracking
    
    def ObjectTracking(self, selection):
         
        # Tracker selection
        self.tracker = self.tracker_selection(selection)
        
        # Intialise the tracker
        self.tracking = self.tracking_init()
        
        if self.mode == 2:
            if self.tracking and not self.tello.takeoff():  #Only take-off when the drone isn't flying and drone camera is tracking
                self.takeoff()
        
        while self.tracking is True: 
            
            # Read frame from Webcam
            if self.mode == 1:
                self.ret, self.image_np = self.cap.read()
            
            #Read frame from Drone  
            elif self.mode == 2:
                 self.image_np  = self.tello.get_frame_read().frame
                 self.ret = self.tello.get_frame_read().grabbed
            
            self.tracking, self.bbox = self.tracker.update(self.image_np)
            print('Tracking Status: {0}'.format(self.tracking))
                       
            p1 = (int(self.bbox[0]), int(self.bbox[1]))
            p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
            
            #Bounding Box
            cv2.rectangle(self.image_np, p1, p2, (255,0,0), 2, 1) 
            
            # Centre of frame/Drone
            self.drone_vector = (int(self.image_np.shape[1]/2), int(self.image_np.shape[0]/2))
            cv2.circle(self.image_np, (self.drone_vector), 10, (255,0,0), 2)
            
            # Centre of Bounding Box
            self.object_vector = int((self.bbox[0]) + (self.bbox[2]/2)), int((self.bbox[1]) + (self.bbox[3]/2))
            cv2.circle(self.image_np, (self.object_vector), 10, (255,255,0), 2)
            
            
            self.deficit_vector = (self.drone_vector[0] - self.object_vector[0], self.drone_vector[1] - self.object_vector[1])
            print('Deficit vector: {0}'.format(self.deficit_vector))
            
            
            # Drone Controller function
            # self.DroneController()

            # Tracking Success
            cv2.putText(self.image_np, "Tracking Success", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
            # Display tracker type on frame
            cv2.putText(self.image_np, self.tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            
            # Display result
            cv2.imshow('object Tracking', cv2.resize(self.image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                if self.mode == 1:
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
                
                elif self.mode == 2:
                    cv2.destroyAllWindows()
                    self.end_sequence()
                    break
            
            
        if not self.tracking: #if tracking is not succesful, try object tracking again
            
            print('Not tracking')
            self.ObjectTracking(selection)


    def DroneController(self):
        
        if self.deficit_vector[0] > 150: # if the x vector is positive (i.e. to the left of the drone, then pan drone right)
            print('yaw_right')
            self.yaw = 20 #yaw right
        
        elif self.deficit_vector[0] < -150:
            print('yaw_left')
            self.yaw = -20 #yaw left

        else:
            self.yaw = 0 #do nothing
         
            
        if self.deficit_vector[1] > 150: # if the y vector is positive (i.e. above the drone, then move drone down)
            print('move_down')
            self.up_down = -20 #move down 
            
        elif self.deficit_vector[1] < -150:
            print('move_up')
            self.up_down = 20 #move up 
   
        else:
            self.up_down = 0 #do nothing
    
    
    def update_position(self):
        
        '''
        Send RC control via four channels. Command is sent every self.TIME_BTW_RC_CONTROL_COMMANDS seconds.
        Arguments:
            left_right_velocity: -100~100 (left/right)
            forward_backward_velocity: -100~100 (forward/backward)
            up_down_velocity: -100~100 (up/down)
            yaw_velocity: -100~100 (yaw)
        Returns:
            bool: True for successful, False for unsuccessful
        '''
    
        self.tello.send_rc_control(self.left_right, self.for_back, self.up_down,
                                       self.yaw)


def main():
    
    ''' 
    Mode 1 = Webcam input
    Mode 2 = Drone input
    
    '''
    mode = 2
    selection = 6
    sportseye = Sportseye(mode)
    sportseye.ObjectTracking(selection)


    ''' 
    Trackers from best to worst: 6, 7, 4, 2
    Re-detection works with 6.
    '''
    
    
if __name__ == '__main__':
    main()