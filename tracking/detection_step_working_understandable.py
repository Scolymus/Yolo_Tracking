"""
Created on Thu Nov 14 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

Image detection treatment library. Based on the work from BigVision LLC.
It is based on the OpenCV project, which is subject to the license terms
in the LICENSE file found in this distribution and at http://opencv.org/license.html
"""

import cv2 as cv
import sys
import numpy as np
import os.path
from scipy import spatial as sps
import settings as g
import time
import math

confThreshold = 0.2  #Confidence threshold
nmsThreshold = 0.1  #Non-maximum suppression threshold
inpWidth = 3200#960  #608     #Width of network's input image
inpHeight = 3200#960 #608     #Height of network's input image
modelConfiguration = ""
modelWeights = ""
centerl_x = []
centerl_y = []
centerl_c = []

#------------------------------------------------------------------------#
#                                                                        #
#                              init_detection                            #
#                                                                        #
#   Constructor to init network variables. Needed to call it before      #
#   using this class                                                     #
#                                                                        #
#   @Inputs:                                                             #
#     classes_ (String). Route to the classnames file for yolo's network #
#     cfg (String). Route to the cfg file for yolo's network             #
#     weights (String). Route to the weights file for yolo's network     #
#     confThr (float). Confidence threshold                              #
#     nmsThr (float). Non-maximum suppression threshold                  #
#     inpW (int). Width of network's input image                         #
#     inpH (int). Height of network's input image                        #
#     cuda (True/False). Use CUDA backend for GPU processing             #
#                                                                        #
#------------------------------------------------------------------------#
def init_detection(classes_, cfg, weights, confThr, nmsThr, inpW, inpH, cuda):
    global confThreshold, nmsThreshold, inpWidth, inpHeight, classes, modelConfiguration, modelWeights, net
    confThreshold = confThr  #Confidence threshold
    nmsThreshold = nmsThr  #Non-maximum suppression threshold
    inpWidth = inpW     #Width of network's input image
    inpHeight = inpH    #Height of network's input image

    classesFile = classes_
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    modelConfiguration = cfg
    modelWeights = weights

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    if cuda == True:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#------------------------------------------------------------------------#
#                                                                        #
#                             getOutputsNames                            #
#                                                                        #
# Get the names of the output layers                                     #
#                                                                        #
#   @Inputs:                                                             #
#     net (cv2.dnn). Yolo's network                                      #
#                                                                        #
#------------------------------------------------------------------------#
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#------------------------------------------------------------------------#
#                                                                        #
#                                drawPred                                #
#                                                                        #
# Draw the predicted bounding box                                        #
#                                                                        #
#   @Inputs:                                                             #
#     frame (frame). Frame where paint the detection                     #
#     classId (int). Class for the object detected                       #
#     conf (float). Confidence for the object detected                   #
#     left (int). Left border of the square                              #
#     top (int). Top border of the square                                #
#     right (int). Right border of the square                            #
#     bottom (int). Bottom border of the square                          #
#                                                                        #
#------------------------------------------------------------------------#
def drawPred(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    #    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    #We only draw those which are of the same class as now!
    if classId == g.class_part:
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

    '''
    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    '''

#------------------------------------------------------------------------#
#                                                                        #
#                                postprocess                             #
#                                                                        #
# Remove the bounding boxes with low confidence using non-maxima         #
# suppression                                                            #
#                                                                        #
#   @Inputs:                                                             #
#     frame (frame). Frame where object is detected                      #
#     outs (array). Matrix with all the data detected                    #
#                                                                        #
#------------------------------------------------------------------------#
def postprocess(frame, outs):
    global pos_temp, ind_num_part_temp

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    #IF THE NETWORK IS DIFFERENT, THIS MAY LEED TO AN ERROR IN MEMORY OVERFLOW! CHECK IT!
    pos_temp = np.zeros((len(outs)*len(outs[0]),4)).astype(int)
    ind_num_part_temp = np.zeros(g.classes_to_recognize).astype(int)
    
    for out in outs:
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            #print(detection)
            classId = np.argmax(scores)#max(range(len(scores)), key=lambda x: scores[x])#np.argmax(scores)
            #if scores[classId]>confThreshold:
            confidence = scores[classId]
            #print(confidence)
            if confidence > confThreshold:              
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                
                pos_temp[ind_num_part_temp[classId], 0] = classId
                pos_temp[ind_num_part_temp[classId], 1] = 1
                pos_temp[ind_num_part_temp[classId], 2] = center_x
                pos_temp[ind_num_part_temp[classId], 3] = center_y

                ind_num_part_temp[classId] += 1

                #centerl_c.append(confidence)                
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    '''
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:#range(len(boxes)):
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        pos_temp[ind_num_part_temp[classId], 0] = classIds[i]
        pos_temp[ind_num_part_temp[classId], 1] = 1
        pos_temp[ind_num_part_temp[classId], 2] = int(left - (width / 2) )
        pos_temp[ind_num_part_temp[classId], 3] = int(top - (height / 2) )

        ind_num_part_temp[classId] += 1
        print(ind_num_part_temp[classId] )

        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
    '''

def transform_image(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    center_frame = [int(frame.shape[1]/2),int(frame.shape[0]/2)]

    return frame, center_frame

def find_particles(frame):
    timeee = time.time()
    # Create a 4D blob from a frame.
    #blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0], 1, crop=False)
    g.time_blob += time.time() - timeee
    timeee = time.time()

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    g.time_detect += time.time() - timeee

def calculate_distances(class_o, particle_id, top_y, left_x, center_frame):
    global pos_temp, ind_lasts_temp, ind_num_part_temp

    if ind_num_part_temp[class_o]>1:
        particles = np.where(g.pos[:, 1] > 0)
        detections = np.where(pos_temp[:, 0] == class_o)
        detections_to_valid = np.full((ind_num_part_temp[class_o], 2),9999999999999999).astype(int)
        ind_valid = 0
        
        distance_to_center_x = pos_temp[detections[0][:],2]-center_frame[0]
        distance_to_center_y = pos_temp[detections[0][:],3]-center_frame[1]
        distance_to_center = distance_to_center_x * distance_to_center_x + distance_to_center_y * distance_to_center_y
        
        min_distance_to_center = np.where(distance_to_center == np.amin(distance_to_center))[0][0]
        
        xx = pos_temp[detections[0][:], 2] + left_x
        yy = pos_temp[detections[0][:], 3] + top_y

        if ind_num_part_temp[class_o] > 1:	#Only calculate neighbors if we have detected more than 1 item!

            for j in range(ind_num_part_temp[class_o]):
                distances_x = g.pos[:,2]-xx[j]
                distances_y = g.pos[:,3]-yy[j]
                distances = distances_x*distances_x+distances_y*distances_y
                min_distances = np.where(distances == np.amin(distances))[0][0]

                if min_distances == particle_id:
                    detections_to_valid[ind_valid, 0] = distances[min_distances]
                    detections_to_valid[ind_valid, 1] = j
                    ind_valid += 1

        if ind_valid > 0:
            indexx = np.where(detections_to_valid == np.amin(detections_to_valid, axis=0))
            return pos_temp[detections[0][detections_to_valid[indexx[0][1], 1]], :].reshape(1, 4)

        return pos_temp[min_distance_to_center, :].reshape(1, 4)
    else:
        return pos_temp[np.where(pos_temp[:, 0] == class_o)[0], :]

#------------------------------------------------------------------------#
#                                                                        #
#                             detect_particles                           #
#                                                                        #
# Detects the particles in the frame providen                            #
#                                                                        #
#   @Inputs:                                                             #
#     frame (frame). Frame where object is detected                      #
#     general (True/False). If true, return all dots detected. If false, #
#                    only that one closest to the center of the image. It#
#                    activates class_o and particle_id                   #
#     class_o (int). -1: all classes. Otherwise, only for that class     #
#     particle_id (int).                                                 #
#                                                                        #
#------------------------------------------------------------------------#
def detect_particles(frame, general, class_o, particle_id, top_y, left_x):
    frame, center_frame = transform_image(frame)

    find_particles(frame)

    timeee = time.time()

    if general == False:
        positions = np.zeros((1,4)).astype(int)

        if ind_num_part_temp[class_o]>0:      
            position = calculate_distances(class_o, particle_id, top_y, left_x, center_frame)    
            positions = position

    else:
        total_particles = 0

        positions = np.full((particle_id,4),-1).astype(int)
        positions[:np.sum(ind_num_part_temp),4] = pos_temp

    g.time_dist += time.time() - timeee
    return positions, frame
