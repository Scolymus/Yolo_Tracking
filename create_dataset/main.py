# -*- coding: utf-8 -*-
"""
# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0
"""

import cv2
import glob
import numpy as np
import imutils
from tkinter import *

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
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"create_dataset")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import Image_API as iml
import settings as g

def change_size_fixed_ROI(wROI, hROI, frames_per_video):
    g.w = wROI
    g.h = hROI
    g.frames_per_video = frames_per_video

def start(window, path_in, path_out, method, classes, wROI, hROI, frames_per_video):

    if path_in.endswith(os.path.sep) == False:
        path_in = path_in + os.path.sep
    if path_out.endswith(os.path.sep) == False:
        path_out = path_out + os.path.sep

    # I will have some variables as globals, stored at g
    g.init()
    g.classes_to_recognize = classes	#Num of different kind of objects to recognize
    g.w = wROI
    g.h = hROI
    g.path_in = path_in
    g.path_out = path_out
    g.frames_per_video = frames_per_video

    # create Tk main frame for dialogs. Hide this frame
    g.root=Tk()
    g.root.withdraw()

    # Look for files
    onlyfiles = sorted(glob.glob(path_in+"*.mkv"), reverse=True)
    if len(onlyfiles) == 0:
        print("There are not video files in that folder!")

    # Start loop files
    video = 0	# Counter for progress bar
    for filename in onlyfiles:
        if method:
            g.init_video(0, filename)	# Open video
        else:
            g.init_video(1, filename)	# Open video

        g.filename = filename
        # Making Window For The Image
        #cv2.namedWindow("Main_window")
        cv2.namedWindow("Main_window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Main_window", 0, 1);

		# Adding Mouse CallBack Event
        cv2.setMouseCallback("Main_window",iml.draw_circle_and_zoom)
	
        # We show the video
        while(g.cap.isOpened()):
            g.dst = g.frame.copy()

            # Control keyboard
            leave = iml.keyboard_control(50, True, True)
            if leave == 0:
                break

            # Read
            ret, g.frame = g.cap.read()

            if hasattr(g.frame, 'shape') == False:
                break

            # Update counter frame
            g.at_frame += 1
            if g.at_frame % 100 == 0:
                print("Frame "+str(g.at_frame))

            # Reset values. If we have any particle, save data
            save = False
            for c in range(g.classes_to_recognize):
                g.ind_lasts[c] = -1
                if g.ind_num_part[c] > 0:
                    save = True
                g.ind_num_part[c] = 0

            if save == True:
                iol.save_images(mypath, filename)
                pass

            g.particles = [[] for i in range(g.classes_to_recognize)]
					
        # Close video and window
        g.cap.release()
        cv2.destroyAllWindows()

        # Update progress bar        
        video = video + 1
        window.prb_images.setValue(video*100/len(onlyfiles))

    window.btn_start.setEnabled(True)
    return False
