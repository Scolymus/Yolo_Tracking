# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:28:24 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

"""

import cv2
import glob
import numpy as np
import re
import imutils
from tkinter import *
import subprocess
import shutil
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
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"prepare_yolo")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import cut_images as cutfiles
import IO_data_yolo as iol_l

def start(window):
    # Correct routes if necessary
    if window.txt_yolo.text().endswith(os.path.sep) == False:
        path_yolo = window.txt_yolo.text() + os.path.sep
    else:
        path_yolo = window.txt_yolo.text()

    if window.txt_dataset.text().endswith(os.path.sep) == False:
        path_data = window.txt_dataset.text() + os.path.sep
    else:
        path_data = window.txt_dataset.text()

    # create Tk main frame for dialogs. Hide this frame
    root=Tk()
    root.withdraw()

    # Ask for names of data
    dataname = simpledialog.askstring("Name of dataset", "What is the name of this dataset?",
                                 parent=root)
    if dataname == "":
        dataname = "test"  

    #https://stackoverflow.com/questions/1276764/stripping-everything-but-alphanumeric-chars-from-a-string-in-python
    dataname = re.sub(r'\W+', '', dataname)

    # Create folder where to copy files
    if not os.path.exists(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep):
        os.makedirs(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep)        
    if not os.path.exists(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"weights"+os.path.sep):
        os.makedirs(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"weights"+os.path.sep)    

    # Ask for names of classes
    classesname = []
    for i in range(window.spb_classes.value()):
        classname = simpledialog.askstring("Name of dataset", "What is the name of this dataset?",
                                 parent=root)

        classname = re.sub(r'\W+', '', classname)

        if classname == "":
            classname = i
        classesname.append(classname)

    # Create file with classes names
    f = open(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"classes.name", "w")
    for i in range(len(classesname)):
        if i == 0:
            f.write(str(classesname[i]))
        else:
            f.write("\n"+str(classesname[i]))
    f.close()

    # Convert to grayscale if necessary
    if window.chk_gray.isChecked():
        colormode = 1
        isgray = True
        onlyfiles = sorted(glob.glob(path_data+"images"+os.path.sep+"*.png"), reverse=True)
        for filename in onlyfiles:
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            path = path_data+"images_gray"+os.path.sep+filename[filename.rfind(os.path.sep)+1:]

            if not os.path.exists(path_data+"images_gray"+os.path.sep):	
                os.makedirs(path_data+"images_gray"+os.path.sep)
            cv2.imwrite(path, image)
        final_folder_images = path_data+"images_gray"+os.path.sep
    else:
        colormode = 3
        isgray = False
        final_folder_images = path_data+"images"+os.path.sep

    final_folder_annotations = path_data+"annotations"+os.path.sep

    # Do we need to cut them?
    if window.spbd_cut.value() > 0.:
        cutfiles.cut(final_folder_images, final_folder_annotations, window.spb_classes.value(), window.spbd_cut.value(), 5000, isgray)
        if isgray:
            final_folder_images = path_data+"images_gray"+os.path.sep+"cut_images"+os.path.sep+"all_images"+os.path.sep
            final_folder_annotations = path_data+"images_gray"+os.path.sep+"cut_images"+os.path.sep+"all_annotations"+os.path.sep
        else:
            final_folder_images = path_data+"cut_images"+os.path.sep+"all_images"+os.path.sep
            final_folder_annotations = path_data+"cut_images"+os.path.sep+"all_annotations"+os.path.sep

    # It seems yolo also needs annotations inside images folder...
    iol_l.copytree(final_folder_annotations, final_folder_images)    

    # Create training/testing files
    path_train, path_test = iol_l.split_data_set(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep, dataname, (100.-window.spb_training.value())/100., final_folder_images)

    # Create file with data
    replacements = {'PREPARE_CLASSES':str(window.spb_classes.value()), 'PREPARE_TRAIN':path_train, 'PREPARE_VALID':path_test, 'PREPARE_NAME':path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"classes.name", 'PREPARE_W':path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"weights"}

    with open(os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"v.data") as infile, open(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+str(dataname)+".data", 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():
                line = line.replace(src, target)
            outfile.write(line)

    # Select the cfg to copy in the destiny folder
    if window.rdb_v4t.isChecked():
        cfg = "cfg"+os.path.sep+"v4t.cfg"
        cluster_num = 6
        weight = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4-tiny.weights"
    elif window.rdb_v4.isChecked():
        cfg = "cfg"+os.path.sep+"v4.cfg"
        cluster_num = 9
        weight = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4.weights"
    elif window.rdb_v3t.isChecked():
        cfg = "cfg"+os.path.sep+"v3t.cfg"
        cluster_num = 6
        weight = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov3-tiny.weights"
    elif window.rdb_v3.isChecked():
        cfg = "cfg"+os.path.sep+"v3.cfg"
        cluster_num = 9
        weight = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+"weights"+os.path.sep+"yolov4.weights"

    # Copy weights
    shutil.copyfile(weight, path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"weights"+os.path.sep+"pre.weight")

    # Replace data in the cfg and copy to the folder

    # width and height must be multiple of 32
    temp = int(window.spb_w.value()/32)
    if temp == 0:
        temp = 1
    window.spb_w.setValue(32*temp)
    temp = int(window.spb_h.value()/32)
    if temp == 0:
        temp = 1
    window.spb_h.setValue(32*temp)

    # Batches max is 
    #   - classes*2000
    #   - but not less than number of training images
    #   - but not less than 6000
    total_images = len(glob.glob1(final_folder_images,"*.png"))

    batch_max = window.spb_classes.value()*2000
    if batch_max < total_images:
        batch_max = total_images
    if batch_max < 6000:
        batch_max = 6000

    # Steps is 80%,90% of batchmax
    steps = str(int(0.8*batch_max))+","+str(int(0.9*batch_max))

    # Add tiny option
    if window.chk_tiny.isChecked():
        if window.rdb_v4.isChecked():
            small_layers = 23
            small_stride = 4
        else:
            small_layers = "-1, 11"
            small_stride = 4
    else:
        if window.rdb_v4.isChecked():
            small_layers = 54
            small_stride = 2
        else:
            small_layers = "-1, 36"
            small_stride = 2

    # Add random option
    if window.chk_random.isChecked():
        random_activated = 1
    else:
        random_activated = 0

    # Look for anchors!
    # https://stackoverflow.com/questions/4760215/running-shell-command-and-capturing-the-output
    # more info in the link for other py versions!
    # Calculate anchors

    print("Calculating anchors")
    ip = 'detector calc_anchors Data'+os.path.sep+str(dataname)+os.path.sep+str(dataname)+".data -num_of_clusters "+str(cluster_num)+" -width "+str(window.spb_w.value())+" -height "+str(window.spb_h.value())
    ip = ip.encode('utf-8')
    result = subprocess.run(path_yolo+'darknet', stdout=subprocess.PIPE, input=ip)
    result.stdout.decode('utf-8')
    print(ip)
    #print(result)

    masks = [[] for j in range(3)]
    filters = [[] for j in range(3)]
    masks_num = ["" for j in range(3)]
    text_anchors = ""
    counter_mask = 0
    add_in_case_empty_mask = True

    # Open anchors (we could have it from the answer, but it is easier from the filetext)
    with open(path_yolo+"anchors.txt") as infile:
        for line in infile:
            elements = line.split(",")            
            # anchors are num1, num2   , num3, num4, .... being (num1, num2) first anchor, (num3, num4) second anchor...
            # anchors are organized as follow (multiply both numbers, e.g num1xnum2=anchor1): (this is for yolo3)
            # for tiny:
            #    layer 1: anchors > 60x60 = 3600
            #    layer 2: else
            # for big:
            #    layer 1: anchors > 60x60 = 3600
            #    layer 2: 60x60 > anchors > 30x30 = 900
            #    layer 2: else
            # in yolov4, the layers are upside down.
            for ele in range(int(len(elements)/2)):
                anchor = [int(elements[2*ele]), int(elements[2*ele+1])]
                if anchor[0]*anchor[1] < 900:
                    if anchor not in masks[0]:
                        masks[0].append(anchor)
                elif anchor[0]*anchor[1] < 3600:
                    if anchor not in masks[1]:
                        masks[1].append(anchor)
                else:                    
                    if anchor not in masks[2]:
                        masks[2].append(anchor)

            if add_in_case_empty_mask:
                if len(masks[0]) == 0:
                    masks[0].append([int(20), int(20)])
                elif len(masks[1]) == 0 and window.rdb_v4t.isChecked() == False and window.rdb_v3t.isChecked() == False:
                    masks[1].append([int(50), int(50)])
                elif len(masks[2]) == 0:
                    masks[2].append([int(100), int(100)])

            counter_mask = 0
            for l in range(len(masks)):                              
                for a in range(len(masks[l])):
                    text_anchors = text_anchors + str(masks[l][a][0]) + "," + str(masks[l][a][1]) + ", "
                    masks_num[l] = masks_num[l]+str(counter_mask)+","
                    counter_mask = counter_mask + 1
                if len(masks[l])> 0:
                    masks_num[l] = masks_num[l][:-1]
                filters[l] = (window.spb_classes.value() + 5)*len(masks[l])
            text_anchors = text_anchors[:-2]    

    # Filters are filters=(classes + 5)xanchors_total


    # for masks
    if window.rdb_v4t.isChecked() or window.rdb_v3t.isChecked():
        masks_small = masks_num[0]
        masks_medium = ""
        if len(masks_num[1]) > 0 and len(masks_num[2]) > 0:
            masks_large = masks_num[1]+","+masks_num[2]        
            filters[2] = (window.spb_classes.value() + 5)*(len(masks[1])+len(masks[2]))
        elif len(masks_num[1]) > 0 and len(masks_num[2]) == 0:
            masks_large = masks_num[1]
            filters[2] = (window.spb_classes.value() + 5)*len(masks[1])
        elif len(masks_num[1]) == 0 and len(masks_num[2]) > 0:
            masks_large = masks_num[2]        
            filters[2] = (window.spb_classes.value() + 5)*len(masks[2])
    else:
        masks_small = masks_num[0]
        masks_medium = masks_num[1]
        masks_large = masks_num[2]

    replacements = {'PREPARE_SATURATION':str(window.spbd_sat.value()), 'PREPARE_EXP':str(window.spbd_exp.value()), 'PREPARE_HUE':str(window.spbd_hue.value()), 'PREPARE_BLUR':str(window.spb_blur.value()), 'PREPARE_ANGLE':str(window.spbd_angle.value()),'PREPARE_BATCH':str(window.spb_batch.value()), 'PREPARE_SUBDIV':str(window.spb_divisions.value()),'PREPARE_W':str(window.spb_w.value()), 'PREPARE_H':str(window.spb_h.value()),'PREPARE_MOMENTUM':str(window.spbd_momentum.value()), 'PREPARE_DECAY':str(window.spbd_decay.value()), 'PREPARE_CHANNELS':str(colormode), 'PREPARE_MAX_BATCHES':str(batch_max),'PREPARE_STEPS':str(steps),'PREPARE_CLASSES':str(window.spb_classes.value()),'PREPARE_LEARNING':str(window.spbd_learning.value()),
'PREPARE_BURN':str(window.spb_burn.value()),'PREPARE_SMALL_LAYERS':str(small_layers),'PREPARE_SMALL_STRIDE':str(small_stride),'PREPARE_RANDOM':str(random_activated), 'PREPARE_FILTERS_SMALL':str(filters[0]), 'PREPARE_FILTERS_MEDIUM':str(filters[1]),'PREPARE_FILTERS_LARGE':str(filters[2]), 'PREPARE_ANCHORS_SMALL':str(text_anchors), 'PREPARE_ANCHORS_MEDIUM':str(text_anchors), 'PREPARE_ANCHORS_LARGE':str(text_anchors), 'PREPARE_MASK_SMALL':str(masks_small), 'PREPARE_MASK_MEDIUM':str(masks_medium), 'PREPARE_MASK_LARGE':str(masks_large), 'PREPARE_NUM_ANCHORS':str(counter_mask)}

    cfg_route = os.path.abspath(os.path.dirname(sys.argv[0]))+os.path.sep+"prepare_yolo"+os.path.sep+cfg

    with open(cfg_route) as infile, open(path_yolo+"Data"+os.path.sep+str(dataname)+os.path.sep+"network.cfg", 'w') as outfile:
        for line in infile:
            for src, target in replacements.items():	#items() is for py 3; for py2 use iteritems()
                line = line.replace(src, target)
            outfile.write(line)

    print("Finished!")
    # TODO: remove extra layers DELETED OPTION!     
