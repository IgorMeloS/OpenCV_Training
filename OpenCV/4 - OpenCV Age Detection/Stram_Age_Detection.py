# =============================================================================
# Age Detetction with OpenCV
# usage:  python3 video_age_detection.py
# =============================================================================

# Importing Libraries

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# Loading the pre-trained models (face and age detection)

prototxt_face = os.path.sep.join(["deploy.prototxt"]) # read the model
weights_face = os.path.sep.join(["res10_300x300_ssd_iter_140000.caffemodel"]) # read the weights
faceNet = cv2.dnn.readNet(prototxt_face, weights_face) # building the network with dnn module

prototxt_age = os.path.sep.join(["age_deploy.prototxt"])
weights_age = os.path.sep.join(["age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxt_age, weights_age)

# Function to detect and predict

def detect_function(frame, faceNet, ageNet, minConf=0.5):
    """
        Function to face detection and age prediction.
        Params:
              frame: input image frame
              faceNet: face detection model
              ageNet: age detection model
              minConf: Minimum confidence probability, 0.5 by default
        return: results with predictions
    """
    # Define the list of age range 
    AGE_RAN = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
    
    # Initialize the results list
    results = []
    (h, w) = frame.shape[:2] # extacting the image coordinates
    # To preprocess the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
   
    # Passing the transformed image through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # Loop for the prediction on each image
    for i in range(0, detections.shape[2]):
       
        # Extract the confidence for all predictions
        confidence = detections[0, 0, i, 2]
       
        # Condition to eliminate the low confidences
        if confidence > minConf:
            
            # grabbing the coordinates of the detected faces
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X, Y, dX, dY) = box.astype("int")
            
            # Defining the region of interest to each detected face and applying the blob to them
            face = frame[Y:dY, X:dX]
            # Ensure that the ROI is sufficiently large
           
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False)
            
            # Making the age prediction
            ageNet.setInput(faceBlob)
            preds = ageNet.forward() # age predictions
            i = preds[0].argmax() # grab the great confidence
            age = AGE_RAN[i] # grab the age range
            ageConfidence = preds[0][i] # putting the confidence
           
            # Dictionary with ROI and predictions
            d = {"loc": (X, Y, dX, dY),
                "age": (age, ageConfidence)}
            results.append(d) # appending the results into the the results list
    
    return results



# Video stream

vs = VideoStream(src=0).start() # from imutils
time.sleep(2.0)

# Grabbing the frames and maaking predictions
while True:
      
    # Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # Detect the faces in the frame and for each face in the frame predict the age
    results = detect_function(frame, faceNet, ageNet)
    
    # Loop over the face age detection results
    for r in results:
        # Drawing the bounding box of the face with age
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
        (X, Y, dX, dY) = r["loc"]
        y = Y - 10 if Y - 10 > 10 else Y + 10
        cv2.rectangle(frame, (X, Y), (dX, dY), (0, 0, 255), 2)
        cv2.putText(frame, text, (X, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
   
    # Show the output frame
    cv2.imshow("Age Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

# Closing 
cv2.destroyAllWindows()
vs.stop()

### Reference###
### (https://www.pyimagesearch.com/2020/04/13/opencv-age-detection-with-deep-learning/)