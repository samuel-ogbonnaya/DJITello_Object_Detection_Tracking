# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:21:05 2020
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
          
class Sportseye:
    # Number of classes to detect
    NUM_CLASSES = 1
    MODEL_NAME = 'trained_inference_graphs'   
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    
    def __init__ (self, cap, retry_count = 3):
        self.cap = cap
        self.retry_count = retry_count

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
                    # Read frame from camera
                    self.ret, self.image_np = self.cap.read()
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
                        min_score_thresh=0.95)
        return self.coordinates

    def yolo_detector():
        pass
    

    def ObjectTracking(self, selection):
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
        
        # Tracking intialisation 
        self.bounding_box = None
        tracking = False
        
        self.bbox = tuple(self.TFdetector()[0])
        
        # Converts Tensorflow object detctor output into correct format for tracker intialisation
        x = self.bbox[2]
        y = self.bbox[0]
        w = self.bbox[3]-self.bbox[2]
        h = self.bbox[1]-self.bbox[0]
        
        self.bounding_box = (x, y, w, h)
        self.tracker.init(self.image_np, self.bounding_box)

        while True:
            
            # Read a new frame
            self.ret, self.image_np = self.cap.read()
            
            if not self.ret:
                break
            
            # Check if an object has been detected
            if self.bounding_box is None and tracking is False:
                print('object not detected')
                
                #Attempt Re-detection - Should use try and exception here (for loop retry count), else raise exception
                self.bbox = tuple(self.TFdetector()[0])
                x = self.bbox[2]
                y = self.bbox[0]
                w = self.bbox[3]-self.bbox[2]
                h = self.bbox[1]-self.bbox[0] 
                self.bounding_box = (x, y, w, h)
                self.tracker.init(self.image_np, self.bounding_box)
                
                if self.bounding_box:
                    print('object re-detected')
                else:
                    print('object not detected again')
                    break
            else:
                tracking = True
                      
            if tracking:
                # Update tracker
                self.status, self.bbox = self.tracker.update(self.image_np)
                # print(self.bbox)
                
                # Draw bounding box
                if self.status:
                    p1 = (int(self.bbox[0]), int(self.bbox[1]))
                    p2 = (int(self.bbox[0] + self.bbox[2]), int(self.bbox[1] + self.bbox[3]))
                    cv2.rectangle(self.image_np, p1, p2, (255,0,0), 2, 1)

                    # Tracking Success
                    cv2.putText(self.image_np, "Tracking Success", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
     
                    # Display tracker type on frame
                    cv2.putText(self.image_np, self.tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
                    
                    # Display result
                    cv2.imshow('object Tracking', cv2.resize(self.image_np, (800, 600)))

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        break
                else :
                    tracking = False
                    self.bounding_box = None
                    
                    # Tracking failure
                    cv2.putText(self.image_np, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                    
                    # will need try and exception here
                    
    def DroneController(self):
        pass


def main():
    cap = cv2.VideoCapture(0)
    sportseye = Sportseye(cap)
    sportseye.ObjectTracking(7)
    #  tested with 4, 6 & 7 , all work 7 seems to be best soo far
    #  2 is temperamental but shows that algorithm works i.e when detection fails midtracking, can re-detect and continue

if __name__ == '__main__':
    main()
