"""
# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0
"""

import cv2 as cv
import sys
import numpy as np
import os.path
from scipy import spatial as sps

# Because of putting the files in a subfolder, and calling them from an other subfolder I got many problems with python not
# recognizing the imports. Just do import all the files in all the subfolders as if they were local, and do include the following
# lines!
# https://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path?rq=1
import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

# Use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"tracking")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import settings_t as g_t
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
gray = False
vector_to_YOLO = [0,0,0]
names_for_forward = None

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
#     to_gray (bool). Do we have to convert frames to gray scale?        #
#                                                                        #
#------------------------------------------------------------------------#
def init_detection(classes_, cfg, weights, confThr, nmsThr, inpW, inpH, cuda, to_gray):
    global confThreshold, nmsThreshold, inpWidth, inpHeight, classes, modelConfiguration, modelWeights, net, gray, names_for_forward, vector_to_YOLO
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

    gray = to_gray
    if gray:
        vector_to_YOLO = [0]
    else:
        vector_to_YOLO = [0,0,0]

    names_for_forward = getOutputsNames(net)

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
def detect_particles(frame, general, class_o, particle_id, top_y, left_x, fixed_w, fixed_h):

    # Transform to gray if needed, and obtain center of frame coordinates
    frame, center_frame = transform_image(frame)

    # Do detection of particles using YOLO
    find_particles(frame)

    timeee = time.time()

    # Do we do detection for all the image (general) or are we tracking one particle?
    if general == False:
        positions = np.zeros((1,6)).astype(int)

        if ind_num_part_temp[class_o]>0:      
            position = calculate_distances(class_o, particle_id, top_y, left_x, center_frame)    
            positions = position
            if fixed_w > 0:
                positions[0,4] = fixed_w
                positions[0,5] = fixed_h

    else:
        # We transfer particles from temporal variable to a definetely variable
        # with the assumed structure
        positions = np.full((g_t.max_num_particles,6),-1).astype(int)
        positions[:np.sum(ind_num_part_temp),:] = pos_temp[:np.sum(ind_num_part_temp),:]
        if fixed_w > 0:
            positions[:np.sum(ind_num_part_temp),4] = fixed_w
            positions[:np.sum(ind_num_part_temp),5] = fixed_h

    g_t.time_dist += time.time() - timeee
    return positions, frame

#------------------------------------------------------------------------#
#                                                                        #
#                             transform_image                            #
#                                                                        #
# Obtains gray image if needed and center of frame                       #
#                                                                        #
#   @Inputs:                                                             #
#     frame (frame). Frame where object is detected                      #
#                                                                        #
#------------------------------------------------------------------------#
def transform_image(frame):
    if gray:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    center_frame = [int(frame.shape[1]/2),int(frame.shape[0]/2)]

    return frame, center_frame

#------------------------------------------------------------------------#
#                                                                        #
#                             Find_particles                             #
#                                                                        #
# Do particle detection using YOLO                                       #
#                                                                        #
#   @Inputs:                                                             #
#     frame (frame). Frame where object is detected                      #
#                                                                        #
#------------------------------------------------------------------------#
def find_particles(frame):
    timeee = time.time()

    # Create a 4D blob from a frame. 1/255 is because images are 8-bit. vector_to_yolo is a mean
    # value to substract, which in this case is 0 (dimension = number of channels of image)
    # Should I include swapBR=True for RGB images?
    # https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), vector_to_YOLO, 1, crop=False)

    g_t.time_blob += time.time() - timeee
    timeee = time.time()

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(names_for_forward)

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    g_t.time_detect += time.time() - timeee

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

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []

    pos_temp = np.zeros((g_t.max_num_particles,6)).astype(int) # Should we create a smaller object?
    ind_num_part_temp = np.zeros(g_t.classes_to_recognize).astype(int)

    # I think outs are the number of proposals, i.e. the number of masks per yolo layer!    
    for out in outs:
        # For each object found in out. Detection is an array like:
        # [center_x, center_y, width, height, probability of being a particle, probability of being class 1, class2, ..., class N]
        for detection in out:
            # This if is for saying that the probability of this finding to be some kind of particle
            # is over this limit. I'll remove it, but you may consider introduce it
            #if detection[4]>0.001:	
            scores = detection[5:]
            classId = np.argmax(scores)

            # Once accepted we have a particle, we can do the same for the specific kind of particle we got
            confidence = scores[classId]
            if confidence > confThreshold:              
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)

                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
                pos_temp[ind_num_part_temp[classId], 0] = classId
                pos_temp[ind_num_part_temp[classId], 1] = 1
                pos_temp[ind_num_part_temp[classId], 2] = center_x
                pos_temp[ind_num_part_temp[classId], 3] = center_y
                pos_temp[ind_num_part_temp[classId], 4] = int(width / 2)
                pos_temp[ind_num_part_temp[classId], 5] = int(height / 2)

                ind_num_part_temp[classId] += 1
              
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    # This step is quite slow using the CPU. The results I got weren't really bad... so I'll coment it.
    # It may need a review of code. Activate it in case you need it.
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
    if classId == g_t.class_part:
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







def calculate_distances(class_o, particle_id, top_y, left_x, center_frame):
    global pos_temp, ind_lasts_temp, ind_num_part_temp

    if ind_num_part_temp[class_o]>1:
        particles = np.where(g_t.pos[:, 1] > 0)
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

            distances = (g_t.pos[:,2]-xx[:,None])**2 + (g_t.pos[:,3]-yy[:,None])**2
            distances_min_per_detection =  distances[np.where(distances.argmin(axis=1) == particle_id)[0],:]
            distances_interesting = distances_min_per_detection[:,particle_id]
            if distances_interesting.size != 0:
                j = np.where(distances.argmin(axis=1) == particle_id)[0][distances_min_per_detection[:,particle_id].argmin(axis=0)]
                return pos_temp[detections[0][j], :].reshape(1, 6)

        return pos_temp[min_distance_to_center, :].reshape(1, 6)
    else:
        return pos_temp[np.where(pos_temp[:, 0] == class_o)[0], :]




