# -*- coding: utf-8 -*-
"""
# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0
"""
import cv2
import numpy as np
import os
import pickle

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

#------------------------------------------------------------------------#
#                                                                        #
#                          Init global variables                         #
#                                                                        #
#------------------------------------------------------------------------#
def init():
    global cap, rows, cols, rows_rot, cols_rot, w, h, zoomSize, start_old, dst, frame, particles, color, class_part, classes_to_recognize, ind_lasts, ind_num_part, at_frame, num_frames, change_frame_num, stopped, root, num_info, particle_mode, filename, path_in, path_out, frames_per_video

    cap = []
    rows = []
    cols = []
    rows_rot = []
    cols_rot = []
    zoomSize = []
    start_old = []
    dst = []
    frame = []
    particles = []
    color = []
    class_part = []
    classes_to_recognize = []
    ind_lasts = []
    ind_num_part = []
    at_frame = 0
    num_frames = []
    change_frame_num = []
    stopped = []
    root = []
    num_info = False
    particle_mode = False
    w = 0
    h = 0
    zoomSize = 1
    filename = ""
    path_in = ""
    path_out = ""
    frames_per_video = 1

#------------------------------------------------------------------------#
#                                                                        #
#                             Init video.                                #
#  Allocate memory and reset variables.                                  #
#  @count_frames_method (int): 0 fast method; 1 slow method. 0 fails     #
#   with the videos I'm using... mp4 compression. It get's stuck forever.#
#  @number_of_particles_to_init (int): It allocates as much memory for   #
#   these number of particles                                            #
#                                                                        #
#------------------------------------------------------------------------#
def init_video(count_frames_method, filename):

    global cap, rows, cols, dst, frame, particles, zoomSize, start_old, class_part, classes_to_recognize, ind_lasts, ind_num_part, at_frame, num_frames, change_frame_num, frames_per_video

    particles = [[] for i in range(classes_to_recognize)]

    # For each class, in each frame I keep the index of the selected particle, which is the one I'll modify
    # This variable is cleaned to -1 each time we open a frame
    ind_lasts = np.full((classes_to_recognize),-1).astype(int)

    # Same as before, but this counts for the total amount of particles per class. There's an extra item at the begginig of everything
    # which is the sum of all particles in all classes
    ind_num_part = np.zeros((classes_to_recognize+1)).astype(int) #First is total!
    
    # We can do zoom in frames. This resets the zoom
    # CAUTION: This was added for unix. In windows there was zoom by default and it should propperly tested!
    zoomSize = 1
    start_old = [0,0]

    # Set to zero some properties
    at_frame = 0	# Frame now
    num_frames = 1	# Number of frames of this video
    class_part = 0	# Class of particle to modify now

    #------------------------------------------------------------------------
    #			                     Open video
    #------------------------------------------------------------------------
    print ('We are looking at file: '+filename+'.')
    cap = cv2.VideoCapture(filename)

    if count_frames_method == 0:
        iml.read_num_frames()
    else:
        iml.read_num_frames_slow(False)

    # Obtaining data for frame size and copy to modify frame. frame = original; dst = frame modified.
    rows,cols,d = frame.shape
    dst = frame.copy()

    # How many frames do we jump between them?
    if num_frames < frames_per_video:
        advance = num_frames
    else:
        advance = int(num_frames*frames_per_video/100)
    change_frame_num = advance

