# =============================================================================
# Face detections using Deep Learning with OpenCV
# Model trained based on SSD model by the caffe module
# =============================================================================

#### importing libraries

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
from os import listdir, path

# Setting paths and files

dataset =  path.abspath("dataset")
caffee_module = "deploy.prototxt.txt"
model = "res10_300x300_ssd_iter_140000.caffemodel"

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(caffee_module , model)

# Creating a list of images from the directory
image_list = listdir(dataset)

count = 1 # counter to build the pyplot display
plt.figure(figsize=(20, 20))
# Starting the main loop to make prediction for each image
for j in image_list:
    img = dataset + "/" + j # setting te image
    image = cv2.imread(img) # reading the image with opencv
    (h, w) = image.shape[:2] # extacting the image coordinates
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob) # setting the network
    detections = net.forward() # setting the predictions
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence >  0.5:
            # compute the (x, y)-coordinates of the bounding box for the  object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10 # ssetting the boundary conditions
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
    # Setting the plot display
    ax = plt.subplot(3, 2, count)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(image_list[count-1])
    plt.axis("off")
    count += 1
plt.savefig("face_prediction.jpg")
plt.show()

# =============================================================================
# As we can note, the predictions are very accurate. All detections present a
# confidence above the 99%, the really state-of-art. Using this model, we do
# not need to be confronted with the large number of false positive, situation
# almost present when considered the Viola-Jones algorithm based on Haar cascade.
# =============================================================================
