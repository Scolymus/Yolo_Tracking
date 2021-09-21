# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0
"""
import cv2
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
import Image_API_t as iml
import numpy as np
import os
import pickle

#------------------------------------------------------------------------#
#                                                                        #
#                          Init global variables                         #
#                                                                        #
#------------------------------------------------------------------------#
def init():
    global cap, rows, cols, zoomSize, start_old, dst, frame, frame_window, pos, pos_tmp_click, pos_tmp_fast, pos_tmp_fast_index, pos_time, vel, color, class_part, classes_to_recognize, ind_lasts, ind_num_part, ind_num_part_click, at_frame, num_frames, change_frame_num, stopped, root, num_info, time_blob, time_detect, time_dist, max_num_particles

    cap = []
    rows = []
    cols = []
    zoomSize = []
    start_old = []
    dst = []
    frame = []
    frame_window = []
    pos = []
    pos_tmp_click = []
    pos_tmp_fast = []
    pos_tmp_fast_index = 0
    pos_time = []
    vel = []
    color = []
    class_part = []
    classes_to_recognize = []
    ind_lasts = []
    ind_num_part = []
    ind_num_part_click = 0
    at_frame = 0
    num_frames = []
    change_frame_num = []
    stopped = []
    root = []
    num_info = False
    time_blob = 0
    time_detect = 0
    time_dist = 0
    max_num_particles = 0

#------------------------------------------------------------------------#
#                                                                        #
#                             Init video.                                #
#                                                                        #
#  Allocate memory and reset variables.                                  #
#                                                                        #
#  @Inputs:                                                              #
#     count_frames_method (int): 0 fast method; 1 slow method. 0 fails   #
#                      with the videos I'm using... mp4 compression. It  #
#                      gets stuck forever.                               #
#                                                                        #
#     num_max_particles (int): It allocates as much memory for these     #
#                      number of particles.                              #
#                                                                        #
#     filename (string). Filename of the video.                          #
#                                                                        #
#------------------------------------------------------------------------#
def init_video(count_frames_method, num_max_particles, filename):

    global cap, rows, cols, dst, frame, zoomSize, start_old, pos, vel, class_part, classes_to_recognize, ind_lasts, ind_num_part, ind_num_part_click, pos_tmp_fast_index, at_frame, num_frames, change_frame_num

    # Variables to store position and speed of all particles only for the last frame
    # each particle -> class,index,x,y,width/2,height/2 (that's why 6)
    pos = np.zeros((num_max_particles,6)).astype(int)
    vel = np.zeros((num_max_particles,4)).astype(int)

    # current postion for knowing the selected particle
    ind_lasts = np.full((classes_to_recognize),-1).astype(int)

    # current amount of particles per class
    ind_num_part = np.zeros((classes_to_recognize+1)).astype(int) #First is total!

    # When the user clicks for adding a particle, this variable counts how many clicks have done
    ind_num_part_click = 0
    pos_tmp_fast_index = 0
    
    # reset zoom!
    zoomSize = 1
    start_old = [0,0]

    # reset video frames
    at_frame = 0
    num_frames = 1
    class_part = 0

    # Load video. There are 2 methods: the slow and the fast one. Fast failed to me with mkv (mp4) videos
    # In the software there is the slow one per default, but you can change it!
    print ('We are looking at file: '+filename+'.')
    cap = cv2.VideoCapture(filename)
    if count_frames_method == 0:
        iml.read_num_frames()
    else:
        iml.read_num_frames_slow(False)

    # Obtaining data for frame size and copy to modify frame. frame = original; dst = frame modified.
    rows,cols,d = frame.shape
    dst = frame.copy()

    change_frame_num = 1

#------------------------------------------------------------------------#
#                                                                        #
#                             Clone variables                            #
#                                                                        #
#  Copy or clone a list Using the Slice Operator. Original source:       #
#   https://www.geeksforgeeks.org/python-cloning-copying-list/           #
#                                                                        #
#------------------------------------------------------------------------#
def Clone(li1):
    li_copy = li1[:]
    return li_copy

#------------------------------------------------------------------------#
#                                                                        #
#                              Save variables                            #
#                                                                        #
#   Save a variable into a file in a binary file. Original source:       #
#  https://stackoverflow.com/questions/3685265/                          #
#           how-to-write-a-multidimensional-array-to-a-text-file         #
#                                                                        #
#------------------------------------------------------------------------#
def save_variables(filename):
    global cap, rows, cols, zoomSize, start_old, dst, frame, pos, pos_tmp_click, pos_tmp_fast, pos_time, vel, color, class_part, classes_to_recognize, ind_lasts, ind_num_part, at_frame, num_frames, change_frame_num, stopped, root, num_info, frame_window

    # Check if folder exists
    if not os.path.exists(filename[:filename.rfind(os.path.sep)+1]+"var"+os.path.sep):
        os.makedirs(filename[:filename.rfind(os.path.sep)+1]+"var"+os.path.sep)

    # Dump all the data into a big array
    tmp_data = []
    tmp_data.append(rows)
    tmp_data.append(cols)
    tmp_data.append(zoomSize)
    tmp_data.append(start_old)
    tmp_data.append(pos)
    tmp_data.append(pos_tmp_click)
    tmp_data.append(pos_tmp_fast)
    tmp_data.append(pos_time)
    tmp_data.append(vel)
    tmp_data.append(color)
    tmp_data.append(class_part)
    tmp_data.append(classes_to_recognize)
    tmp_data.append(ind_lasts)
    tmp_data.append(ind_num_part)
    tmp_data.append(at_frame)
    tmp_data.append(num_frames)
    tmp_data.append(change_frame_num)
    tmp_data.append(stopped)
    tmp_data.append(num_info)
    tmp_data.append(frame_window)

    # Save all data
    filename_out = filename[:filename.rfind(os.path.sep)+1]+"var"+os.path.sep+filename[filename.rfind(os.path.sep)+1:-4]+".dat"
    output = open(filename_out, 'wb')
    pickle.dump(tmp_data, output)
    output.close()

    tmp_data = []

#------------------------------------------------------------------------#
#                                                                        #
#                              Load variables                            #
#                                                                        #
#   Load a variable into a file in a binary file. Original source:       #
#  https://stackoverflow.com/questions/3685265/                          #
#           how-to-write-a-multidimensional-array-to-a-text-file         #
#                                                                        #
#------------------------------------------------------------------------#
def load_variables(filename):
    global cap, rows, cols, zoomSize, start_old, dst, frame, pos, pos_tmp_click, pos_tmp_fast, pos_time, vel, color, class_part, classes_to_recognize, ind_lasts, ind_num_part, at_frame, num_frames, change_frame_num, stopped, root, num_info

    # Load the data file
    filename_in = filename[:filename.rfind(os.path.sep)+1]+"var"+os.path.sep+filename[filename.rfind(os.path.sep)+1:-4]+".dat"
    input_f = open(filename_in, 'rb')
    tmp_data = pickle.load(input_f)
    input_f.close()

    # Take data to this run
    rows = tmp_data[0]
    cols = tmp_data[1]
    zoomSize = tmp_data[2]
    start_old = tmp_data[3]
    pos = tmp_data[4]
    pos_tmp_click = tmp_data[5]

    pos_time = tmp_data[7]
    vel = tmp_data[8]
    color_tmp = tmp_data[9]
    class_part = tmp_data[10]
    classes_to_recognize = tmp_data[11]

    at_frame = tmp_data[14]
    num_frames = tmp_data[15]
    change_frame_num = tmp_data[16]
    stopped = tmp_data[17]
    num_info = tmp_data[18]
    frame_window = tmp_data[19]

    ind_lasts = np.full((classes_to_recognize),-1).astype(int)
    ind_num_part = np.zeros((classes_to_recognize+1)).astype(int) #First is total!

    for i in range(len(tmp_data[12])):
        ind_lasts[i] = tmp_data[12][i]
        ind_num_part[i+1] = tmp_data[13][i]
        ind_num_part[0] += ind_num_part[i+1]

    index = 0
    for t in range(len(tmp_data[6])):
        for c in range(len(tmp_data[6][t])):
            for p in range(len(tmp_data[6][t][c][0])):
                pos_tmp_fast[index, 0] = c
                pos_tmp_fast[index, 1] = t
                pos_tmp_fast[index, 2] = tmp_data[6][t][c][0][p]
                pos_tmp_fast[index, 3] = tmp_data[6][t][c][1][p]
                index += 1

    change_frame_num = 1

    tmp_data = []
