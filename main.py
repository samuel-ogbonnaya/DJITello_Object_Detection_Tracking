# -*- coding: utf-8 -*-
"""
Created on Wed May 27 00:42:12 2020

@author: isogb
"""


from djitellopy import Tello
import threading
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
import math

          
class Sportseye:
    
    # Number of classes to detect
    NUM_CLASSES = 1
    MODEL_NAME = 'trained_inference_graphs'   
    
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    
    def __init__ (self, mode, selection, no_takeoff_mode, retry_count = 2):
        
        self.mode = mode
        self.selection = selection
        self.tracking = False
        self.tracking_counter = 0
        self.no_takeoff_mode = no_takeoff_mode
        
        #Webcam 
        if self.mode == 1:
            self.cap = cv2.VideoCapture(0)
        
        #Drone
        elif self.mode == 2:
            self.tello = Tello()
            self.tello.connect()
            self.tello.streamon()
            self.tello.set_speed(10)
        
        
        self.retry_count = retry_count
        self.left_right = 0
        self.for_back = 0
        self.up_down = 0
        self.yaw = 0
        
        
    def TFdetector(self):    
        
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,
                                         'sportseye_v1_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,
                                           'training','label_map.pbtxt')
        
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
        categories = label_map_util.convert_label_map_to_categories(label_map, 
                                                                    max_num_classes=self.NUM_CLASSES, use_display_name=True)
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
                    (self.boxes, self.scores,
                     self.classes, self.num_detections) = sess.run([self.boxes,
                                                                    self.scores,
                                                                    self.classes,
                                                                    self.num_detections],
                                                                    feed_dict=
                                                                    {self.image_tensor:
                                                                     self.image_np_expanded})
                    
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
                        min_score_thresh=0.7)
        return self.coordinates

    
    def yolo_detector():
        pass
    
    
    def takeoff(self):
        
        '''Only take-off when the drone isn't flying and drone camera has been
        tracking for more than 20 consecutive frames to enable user stablise the ball'''
        
        
        while True:
            
            if (self.mode == 2 and self.tracking and self.tracking_counter > 20
                and self.tello.is_flying == False and self.no_takeoff_mode == False):
                
                print('take off condition met')
                self.tello.takeoff()
                break
            
            elif self.mode == 1 or self.no_takeoff_mode:
                print('No takeoff mode') 
                break
            
            else:
                continue
        return 
    
    
    def end_sequence(self):
        while True:
            if (self.no_takeoff_mode == False and self.tello.is_flying == True):
                print('landing')
                self.tello.land() 
            
            print('Ending')
            self.tello.streamoff()
            self.tello.end() 
            break
        
        return
    
    
    def tracker_selection(self):
        
        # Set up trackers.
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 
                         'GOTURN', 'MOSSE', 'CSRT']
        
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
    
    
    def tracking_initialisation(self):
        
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
   
        
    def ObjectTracking(self):
        
         # Tracker selection
        self.tracker = self.tracker_selection()
        
        
        start = time.time()
        
        # Intialise the tracker
        self.tracking = self.tracking_initialisation()  
        
        end = time.time()
        
        print('TFDetector function took: {} to run'.format(end-start))
       
        while self.tracking is True: 
            
            # Ensure consistent tracking before taking off
            if self.tracking_counter < 30:
                self.tracking_counter = self.tracking_counter + 1
            
            #update position of drone
            if self.mode == 2 and not self.no_takeoff_mode:
                self.update_position()
            
            # Read frame from Webcam
            if self.mode == 1:
                self.ret, self.image_np = self.cap.read()
            
            #Read frame from Drone  
            elif self.mode == 2:
                 self.image_np  = self.tello.get_frame_read().frame
                 self.ret = self.tello.get_frame_read().grabbed
            
            self.tracking, self.bbox = self.tracker.update(self.image_np)
            print('Tracking Status: {0}'.format(self.tracking))
            
            #Normal Bounding Box parameters
            x1 = self.bbox[0]
            y1 = self.bbox[1]
            w1 = self.bbox[2]
            h1 = self.bbox[3]
            #x2 = x1 + w1
            #y2 = y1 + h1
            cx = x1 + (w1/2)
            cy = y1 + (h1/2)
            rad = int(w1/2)
            increase_factor = 3
            rad_area = int((math.pi * math.pow(rad,2)) * increase_factor)
            factored_rad = int(math.sqrt((rad_area)/math.pi))
            
            #Bounding Box
            #cv2.rectangle(self.image_np, p1, p2, (255,0,0), 2, 1) 
            
            # Bounding Circle
            cv2.circle(self.image_np, (int(cx), int(cy)), factored_rad, (255,0,0), 2)
                       
            # Centre of frame/Drone
            self.drone_vector = (int(self.image_np.shape[1]/2), 
                                 int(self.image_np.shape[0]/2))
            
            cv2.circle(self.image_np, (self.drone_vector), 10, (255,0,0), 2)
            
            # Centre of Bounding Circle/Box
            self.object_vector = (int(cx), int(cy))
            
                                        
            cv2.circle(self.image_np, (self.object_vector), 10, (255,255,0), 2)
            
            
            self.deficit_vector = (self.drone_vector[0] - self.object_vector[0],
                                   self.drone_vector[1] - self.object_vector[1])
            
            print('Deficit vector: {0}'.format(self.deficit_vector))
            
            
            # Drone Controller function
            if self.mode ==2 and not self.no_takeoff_mode:
                self.DroneController()

            # Tracking Success
            cv2.putText(self.image_np, "Tracking Success", (100,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
            # Display tracker type on frame
            cv2.putText(self.image_np, self.tracker_type + " Tracker", 
                        (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            
            # Display result
            cv2.imshow('Object Tracking', cv2.resize(self.image_np, (800, 600)))

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
            self.ObjectTracking()
            


    def DroneController(self):
        
        # Lateral Control
        if self.deficit_vector[0] > 200: # if the x vector is positive (i.e. to the left of the drone, then pan drone left)
            print('yaw_left')
            self.yaw = -30 #yaw left
        
        elif self.deficit_vector[0] < -200:
            print('yaw_right')
            self.yaw = 30 #yaw right

        else:
            self.yaw = 0 #do nothing
         
        #Vertical Control
        if self.deficit_vector[1] > 200: # if the y vector is positive (i.e. above the drone, then move drone up)
            print('Moving Drone Up')
            self.up_down = 30 #move Up
            
        elif self.deficit_vector[1] < -200:
            print('Moving Drone Down')
            self.up_down = -30 #move Down 
   
        else:
            self.up_down = 0 #do nothing
            
        #Longitudinal Control
        if self.deficit_vector:
            pass
            
            
      
    
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
    no_takeoff_mode = False  # For diagnostics
    selection = 6  # tracker selection
    sportseye = Sportseye(mode, selection, no_takeoff_mode)
    
    # Utilisation of threading for tracking and taking off
    t1 = threading.Thread(target=sportseye.ObjectTracking, args=())
    t2 = threading.Thread(target=sportseye.takeoff, args=())
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    ''' 
    Trackers from best to worst: 6, 7, 4, 2
    Re-detection works with 6.
    '''
    
    
if __name__ == '__main__':
    main()
