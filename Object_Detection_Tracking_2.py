# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 23:37:59 2019

@author: isogb
"""

import cv2
import numpy as np
import sys
import time
from apscheduler.schedulers.background import BackgroundScheduler


# Set up trackers.
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[0]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
elif tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
elif tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()


# Implement detector algorithm to detect a ball (object) and store output in bounding box:
def ball_detector():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Capture frame from webcam:
    #ret, img = capture.read()
    global frame
    #frame = cv2.resize(frame, None, fx=0.4, fy=0.4)  #need to understand this line 
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            global confidence_out
            confidence = scores[class_id]
            if confidence > 0.5:
                confidence_out = confidence 
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id) 
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    #Bounding Box Output
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            global bbox 
            bbox = tuple(boxes[i]) #Bounding Box Output
            #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            #print(bbox)

# Read video from webcam
video = cv2.VideoCapture(0)

# Exit if video not opened.
if not video.isOpened():
    print ("Could not open video")
    sys.exit()
 
object_detected = False #initialise object detected as False
print('object_detected = {}'.format(object_detected))

while True:
    
    # Read a new frame
    ok, frame = video.read()

    if not ok:
        break
    
    # Start timer
    timer = cv2.getTickCount()
    
    if object_detected is False:
        bbox = () #reset bbox as empty tuple, so doesnt use value stored in memory from previous run
        confidence_out = float(0) # same as bbox variable
    
        ## detect an object, if object detected is False (intial state)
        ball_detector() 
            
        if (len(bbox) == 4) and (confidence_out > 0.9):  # if an object is detected 
            tracker.init(frame, bbox)  #Initialize the tracker with a known bounding box that surrounded the target.
            object_detected = True  #set object detected state to true
            print('object_detected_tracking = {}'.format(object_detected))
        else:
            object_detected = False
      
    
    if object_detected is True: ##if we have detected an object

        print('object_detected = {}'.format(object_detected))

        ## begin tracking 
        ok, bbox = tracker.update(frame)  #Update tracker
        #print(ok)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
     
        # Draw a bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            #cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            print('Tracking success') 

        else:
            # Tracking failure
            #cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            print('Tracking failure')
            object_detected = False
    
    print('object_detected = {}'.format(object_detected))                
    
            
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
 
    # Display FPS on frame
    #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
    # Display result
    cv2.imshow("Tracking", frame)
 
    # Exit if ESC pressed
    key = cv2.waitKey(1) & 0xff
    
    if key == ord('q'):
        video.release()
        cv2.destroyAllWindows()


#sched.shutdown()