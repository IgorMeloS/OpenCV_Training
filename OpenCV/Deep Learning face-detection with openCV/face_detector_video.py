# =============================================================================
# Face detections using Deep Learning with OpenCV
# Model trained based on SSD model by the caffe module
# =============================================================================


#### importing libraries
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

#### Setting the files for the model
caffee_module = "deploy.prototxt.txt" # the parameters to run the dnn model
model = "res10_300x300_ssd_iter_140000.caffemodel" # the desired model

#### Loading the model

net = cv2.dnn.readNetFromCaffe(caffee_module , model)

### Setting the video from webcam

vs = VideoStream(src=0).start() # to make the video from webcam
time.sleep(2.0)

# loop over the frames from the video stream

while True:
    frame = vs.read() # frame from the webcam
    frame = imutils.resize(frame, width=400) # resizing the frame
    
    (h, w) = frame.shape[:2] # coordinates from the frame
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0)) # setting the blob to obtain the predictions
    net.setInput(blob) # setting the DNN with the blob
    detections = net.forward() # object to detect
    
    # Main loop to make predictions
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] # confidence, it's means the value of the probability of the prediction
        if confidence < 0.5: #to be sure that the passed confidence is greater than the minimum 
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # setting the box for the prediction
        (startX, startY, endX, endY) = box.astype("int") # box coordinates
        text = "{:.2f}%".format(confidence * 100) # to put the confidence in text format
        y = startY - 10 if startY - 10 > 10 else startY + 10 # setting the boundary conditions
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2) # making the box for the predictions with openCV
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) # putting the text over the box
    cv2.imshow("Frame", frame) # displaying the prediction
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): # to stop the video stream press "q"
        break
# to clean
cv2.destroyAllWindows()
vs.stop()

# =============================================================================
# After the usage, you must to reset your kernel to shutdown the camera
# =============================================================================
