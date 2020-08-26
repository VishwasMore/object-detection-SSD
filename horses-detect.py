#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:38:08 2020

@author: vishwasmore
"""


# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Defining a function that will do the detections

# We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images, and that will return the frame with the detector rectangle.
def detect(frame, net, transform):
    # We get the height and the width of the frame(image).
    height, width = frame.shape[0:2] 
    # We apply the transformation to our frame.
    frame_t = transform(frame)[0]
    # We convert the frame into a torch tensor.
    x = torch.from_numpy(frame_t).permute(2, 0, 1) #(Blue, Red, Green)
    # We add a fake dimension corresponding to the batch.
    x = Variable(x.unsqueeze(0))
    # We feed the neural network ssd with the image and we get the output y.
    y = net(x)
    # We create the detections tensor contained in the output y.
    detections = y.data
    # We create a tensor object of dimensions [width, height, width, height].
    scale = torch.Tensor([width, height, width, height])
    # For every class: 
    # detections = [batch, number of classes, number of occurence, (score, pt0, pt1, pt2, pt3)]
    for i in range(detections.size(1)):
        # We initialize the loop variable j that will correspond to the occurrences of the class.
        j = 0
        # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
        while detections[0, i, j, 0] >=0.6:
            # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.
            pt = (detections[0, i, j, 1:] *scale).numpy()
            # We draw a rectangle around the detected object.
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            # We put the label of the class right above the rectangle.
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            # We increment j to get to the next occurrence.
            j += 1
    return frame


# Creating the SSD neural network
    
# We create an object that is our neural network ssd.
net = build_ssd('test')
 # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

# Creating the transformation

# We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('epic_horses.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('horses_detected.mp4', fps = fps) # We create an output video with this same fps frequence.
for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    