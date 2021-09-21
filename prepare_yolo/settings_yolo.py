# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

"""
import os
import pickle

#------------------------------------------------------------------------#
#                                                                        #
#                          Init global variables                         #
#                                                                        #
#------------------------------------------------------------------------#
def init():
    global cap, rows, cols, d, dst, frame, particles, classes_to_recognize, at_frame, num_frames, root, cut_length

    cap = []
    rows = []
    cols = []
    d = []
    dst = []
    frame = []
    particles = []
    classes_to_recognize = []
    at_frame = 0
    num_frames = []
    root = []
    cut_length = 0

