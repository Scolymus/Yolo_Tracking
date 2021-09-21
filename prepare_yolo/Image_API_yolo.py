"""
Created on Thu May 29 2020

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

Auxiliar image treatment function
"""

import colorsys
import cv2
import imutils
import numpy as np
import tkinter as tk
from tkinter import simpledialog

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

import settings_yolo as g_l
import IO_data_yolo as iol_l



#****************************************************************************#
#                                                                            #
#                             Image manipulation                             #
#                                                                            #
#****************************************************************************#

#------------------------------------------------------------------------#
#                                                                        #
#				Update image with squares around particles               #
#                                                                        #
#   @Inputs:                                                             #
#     image (frame). Frame to update                                     #
#                                                                        #
#------------------------------------------------------------------------#
def update_image(image):
    colour_unselected = (0,0,255)

    # For each particle
    for p in range(len(g_l.particles[0])):   
        # Draw vertexes clicked
        x = g_l.particles[1][p]
        y = g_l.particles[2][p]
        w = g_l.particles[3][p]
        h = g_l.particles[4][p]
        cv2.polylines(image, [np.array([[int(x-w/2),int(y-h/2)], [int(x-w/2),int(y+h/2)], [int(x+w/2),int(y+h/2)], [int(x+w/2),int(y-h/2)]]).reshape((-1, 1, 2))], True, colour_unselected, 2)
  				
    return image

