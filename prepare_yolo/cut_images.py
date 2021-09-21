# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:28:24 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0
"""

import glob
import cv2
import numpy as np
import imutils

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

import IO_data_yolo as iol_l
import Image_API_yolo as iml_l
import settings_yolo as g_l


image_format=".png"

def cut(folder_images, folder_annotations, classes, cutl, trials, isgray):

    g_l.init()          # Call only once
    g_l.classes_to_recognize = classes
    g_l.cut_length = cutl
    onlyfiles = sorted(glob.glob(folder_images+"*"+image_format), reverse=True)

    for filename in onlyfiles:
        print ('We are looking at file: '+filename+'.')	
        g_l.frame = cv2.imread(filename)
        g_l.rows,g_l.cols,g_l.d = g_l.frame.shape
        dst = g_l.frame.copy()

        # Load particles truth bounding boxes

        iol_l.YOLOv3_to_centers(folder_annotations+filename[filename.rfind(os.path.sep)+1:].replace(image_format,".txt") , g_l.rows, g_l.cols)		
        ''' test for loading!!!!
        show=0
        dst = iml_l.update_image(dst)
        # Starting The Loop So Image Can Be Shown
        if show == 0:
            while(True):
                cv2.imshow("Window",dst)
                k = cv2.waitKey(20) & 0xFF
                if k == ord('q') or k == ord('Q'):
                    break				
                if cv2.getWindowProperty('Window',cv2.WND_PROP_VISIBLE) < 1:        
                    break        				

        cv2.destroyAllWindows()
        '''

        images = iol_l.Particle2image(filename, trials, isgray)

        # Now we copy all files into the correct folders
        folder = filename[:filename.rfind(os.path.sep)+1]+"cut_images"+os.path.sep

        if not os.path.exists(folder+"all_images"+os.path.sep):
            os.makedirs(folder+"all_images"+os.path.sep)
        if not os.path.exists(folder+"all_annotations"+os.path.sep):
            os.makedirs(folder+"all_annotations"+os.path.sep)

        iol_l.copytree(folder+"images_void", folder+"all_images")
        iol_l.copytree(folder+"images", folder+"all_images")
        iol_l.copytree(folder+"annotations_void", folder+"all_annotations")
        iol_l.copytree(folder+"annotations", folder+"all_annotations")
        #iol.split_data_set(mypath+"cut_images/images", "/media/lucas/Phd1/IA/detectors/dataset_images/Janus_topo_mask_3/all/cut_images/images/")#"/home/lpalacios/IA/janus_5um_10x/images")

