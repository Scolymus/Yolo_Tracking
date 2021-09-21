# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:28:24 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

"""

import copy
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
import detection_step as dsl
import glob
import Image_API_t as iml_t
import IO_data_t as sv
import numpy as np
import os
import settings_t as g_t
import sys
import time
from tkinter import *


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(50)

# IMAGE (Rp and error are matrixes of the size of the number of kind of objects to recognize)
res = 1.5436 #9.7015 63x  6.1744 40x  1.5436 10x				#Resolution objective->px
Rp=[2.5]	  													#Particle radii in mu
error = [1.3]													#Error measuring
show_image = True                                               #If true, video will be shown in a window

#------------------------------------------------------------------------#
#                                                                        #
#				  Init_tracking: Starts tracking algorithm               #
#                                                                        #
#   @Inputs:                                                             #
#     particles_num (int). Number of particles to allocate in memory     #
#     blob* (int). Resize images to this size to do the detection        #
#     window* (int). Cut image of this size to do the detection          #
#     type_of_tracking (int). Use for click first (1) or load (2) from   #
#                    previous usage.                                     #
#     use_cuda (boolean). If true, use CUDA for OpenCV DNN. Requires     #
#                    OpenCV compiled with CUDA. Min platform 5.4         #
#     path_* (string). Paths to folders                                  #
#                                                                        #
#------------------------------------------------------------------------#
def init_tracking(particles_num, fixed_w_particle, fixed_h_particle, blob_x, blob_y, blob_x_f, blob_y_f, window_w, window_h, type_of_tracking, use_cuda, path_dataset, path_videos):
    g_t.init()          # Call only once. It sets to 0 many global variables

    cuda = use_cuda		#Use CUDA for OpenCV DNN object
    videos_path = path_videos+os.path.sep
    dataset_name = path_dataset[path_dataset.rfind(os.path.sep)+1:]
    classes_names = path_dataset+os.path.sep+"classes.names"
    network_cfg = path_dataset+os.path.sep+"network.cfg"
    network_weights = path_dataset+os.path.sep+"weights"+os.path.sep+"network_best.weights"

    blob_general_x = blob_x_f
    blob_general_y = blob_y_f
    blob_window_x = blob_x
    blob_window_y = blob_y

    p_f_w = fixed_w_particle
    p_f_h = fixed_h_particle

    number_of_particles_to_init = particles_num
    #These are the maximum amount of particles to detect. If your app needs more, increase this number.                             
    #DISCLAMER. In principle, I was just appending new particles, using lists, but I had to rebuild
    #different areas of the code and I decide to use numpy arrays. Although you can still appending in
    #numpy, I thought the program could be written in a more C code, without dynamical array growing_t.
    #The idea is to be ready to port it to numba, in case needed.

    f = open(network_cfg, "r")
    for x in f:
       if "classes" in x:
           x=x.split("=")
           g_t.classes_to_recognize = int(x[1])
           break

    f = open(network_cfg, "r")
    for x in f:
       if "channels" in x:
           x=x.split("=")
           if int(x[1]) == 1:
               needs_gray = True
           else:
               needs_gray = False
           break

    g_t.frame_window = [window_w, window_h]  #Radius time

    detect_first = False
    load_previous_values = False
    if type_of_tracking == 1:
        detect_first = True
    elif type_of_tracking == 2:
        load_previous_values = True

    g_t.root=Tk()
    g_t.root.withdraw()

    g_t.max_num_particles = number_of_particles_to_init
    
    #limit_frame_undetect.append([R[i]/2,cols-R[i]/2,R[i]/2,rows-R[i]/2])

    track(cuda, videos_path, dataset_name, classes_names, network_cfg, network_weights, blob_general_x, blob_general_y, blob_window_x, blob_window_y, number_of_particles_to_init, detect_first, load_previous_values, needs_gray, p_f_w, p_f_h)

def track(cuda, videos_path, dataset_name, classes_names, network_cfg, network_weights, blob_general_x, blob_general_y, blob_window_x, blob_window_y, number_of_particles_to_init, detect_first, load_previous_values, needs_gray, p_f_w, p_f_h):
    # I only tried with videos in mkv, which were compressed in mp4. I guess there's no problem with avi videos.
    # If you need more extensions, add here and check then if the code can run well...
    onlyfiles = sorted(glob.glob(videos_path+"*.mkv")+glob.glob(videos_path+"*.mp4")+glob.glob(videos_path+"*.avi"), reverse=True)

    for filename in onlyfiles:
        # Init/Reset variables
        g_t.init_video(1, number_of_particles_to_init, filename)
        cv2.namedWindow("Main_window")

        # Select how to proceed with tracking
        if load_previous_values == False:	# We will track directly.

            # First we load the network for detecting (almost) all the particles, and we try to detect them
            dsl.init_detection(classes_names, network_cfg, network_weights, 0.2, 0.05, blob_general_x, blob_general_y, cuda, needs_gray)
            print("General Network loaded!")

            # Detect particles for the first time
            g_t.pos, dst2 = dsl.detect_particles(g_t.dst, True, -1, -1, -1, -1, p_f_w, p_f_h)

            #------------------------------------------------------------------------
            #			      Do fast detection or continue as usual
            #------------------------------------------------------------------------
            # We can check all frames first, so after it automatizes when a particles enters into the frame
            # In this mode we just play de video and the user clicks when thinks there is a new particle, but it is not tracked
            # Lately, user will run again the software with this option off and load on, and the software will automatically take 
            # the initial position in the frame the user click as a new particle
            if detect_first == True:
                print("NOTICE: This mode was working fine before the adaptation to numpy. Be aware it wasn't tested yet. Furthermore, you need to change the load_variable mode since now the load_variable_v2 is for loading data from the input without numpy!")
                # each particle -> class,t0,x,y (that's why 4)
                # This is a temporal variable where I only save positions of the first time detected by the user
                g_t.pos_tmp_fast = np.zeros((number_of_particles_to_init,4)).astype(int)

                for p in range(g_t.ind_num_part[0]):
                    g_t.pos_tmp_fast[p, 0] = g_t.pos[p, 0]
                    g_t.pos_tmp_fast[p, 1] = 0
                    g_t.pos_tmp_fast[p, 2] = g_t.pos[p, 2]
                    g_t.pos_tmp_fast[p, 3] = g_t.pos[p, 3]

                # Use proper Callback Event
                cv2.setMouseCallback("Main_window", iml_t.draw_circle_and_zoom_fast)

                # Paint positions
                iml_t.change_frame(0,0,True)

                # Then, we let the user to modify them!
                leave = iml_t.keyboard_control_fast(20, False, show_image)
                if leave == 0:
                    continue

                # We start running the video!
                iml_t.change_frame(0,0,True)

                # We show the video
                while(g_t.cap.isOpened()):
                    g_t.dst = g_t.frame.copy()

                    # Control keyboard
                    leave = iml_t.keyboard_control_fast(50, True, True)
                    if leave == 0:
                        break

                    # Read
                    ret, g_t.frame = g_t.cap.read()

                    if hasattr(g_t.frame, 'shape') == False:
                        break
                    g_t.at_frame += 1
                    if g_t.at_frame % 100 == 0:
                        print("Frame "+str(g_t.at_frame))

                    for c in range(g_t.classes_to_recognize):
                        g_t.ind_lasts[c] = 0

                g_t.save_variables(filename)

                continue
            #The next else is for: We don't want to do preselection, neither we have data to load. Normal working mode!
            else:
                # This will be the real matrix where I will store the data!
                g_t.pos_time = np.full((number_of_particles_to_init, 4, g_t.num_frames),-1).astype(int)
                particles_this_class = np.where(g_t.pos[:, 1] > 0)
                g_t.ind_num_part[0] = len(particles_this_class[0])

                for c in range(g_t.classes_to_recognize):
                    particles_this_class = np.where(g_t.pos[:, 0] == c)
                    g_t.ind_lasts[c] = len(particles_this_class[0])-1
                    g_t.ind_num_part[c+1] = len(particles_this_class[0])

                # Load colors for tracking analysis
                g_t.color = np.zeros((number_of_particles_to_init, 3)).astype(int)   #particle, RGB
                for p in range(g_t.ind_num_part[0]):
                    g_t.color[p] = iml_t.hls2rgb(iml_t.generate_hls())

                # Paint positions
                iml_t.change_frame(0,0,False)

                # Adding Mouse CallBack Event
                cv2.setMouseCallback("Main_window", iml_t.draw_circle_and_zoom)

                # Then, we let the user to modify them!
                leave = iml_t.keyboard_control(20, False, show_image)
                if leave == 0:
                    continue
                iml_t.change_frame(0,0,False)

                g_t.vel = copy.deepcopy(g_t.pos)

        else:
           # Adding Mouse CallBack Event
           cv2.setMouseCallback("Main_window", iml_t.draw_circle_and_zoom)
           g_t.color = np.zeros((number_of_particles_to_init, 3)).astype(int)   #particle, RGB
           g_t.pos_tmp_fast = np.zeros((number_of_particles_to_init,4)).astype(int)
           g_t.load_variables_v2(filename)
           g_t.at_frame = 0
           g_t.ind_lasts = np.full((g_t.classes_to_recognize),-1).astype(int)
           g_t.ind_num_part = np.zeros(g_t.classes_to_recognize+1).astype(int) #First is total!
           g_t.ind_num_part_click = 0
           g_t.pos_time = np.full((number_of_particles_to_init, 4, g_t.num_frames),-1).astype(int)
           g_t.pos = np.full((number_of_particles_to_init, 6),-1).astype(int)
           g_t.vel = np.full((number_of_particles_to_init, 4),-1).astype(int)

        # Now we load the windowed network to detect correctly the particles (avoid blinking)
        dsl.init_detection(classes_names, network_cfg, network_weights, 0.2, 0.1, blob_window_x, blob_window_y, cuda, needs_gray)
        print("Windowed Network loaded!")

        # We start saving the video processed
        if not os.path.exists(videos_path[:videos_path.rfind(os.path.sep)+1]+"tracked"+os.path.sep):
            os.makedirs(videos_path[:videos_path.rfind(os.path.sep)+1]+"tracked"+os.path.sep)

        out = cv2.VideoWriter(videos_path+"tracked/"+filename[filename.rfind("/")+1:],cv2.VideoWriter_fourcc(*'FMP4'), 25, (g_t.cols, g_t.rows), True)
        print("We started saving the video...")

        #------------------------------------------------------------------------
        #			             Do detection every frame
        #------------------------------------------------------------------------
        # Starting The Loop So Image Can Be Shown
        timeee = time.time()
        g_t.pos_tmp_click = np.zeros((int(number_of_particles_to_init/10),3)).astype(int)

        while(g_t.cap.isOpened()):
            g_t.dst = g_t.frame.copy()
            #for each particle of this class
            for p in range(g_t.ind_num_part[0]):
                if g_t.pos[p, 1] < 0:
                    continue
                #take a subwindow around the particle (depends on user conf.)
                xmin_f = int(g_t.pos[p, 2]-g_t.frame_window[g_t.pos[p, 0]]) if g_t.pos[p, 2]-g_t.frame_window[g_t.pos[p, 0]] > 0 else 0
                xmax_f = int(g_t.pos[p, 2]+g_t.frame_window[g_t.pos[p, 0]]) if g_t.pos[p, 2]+g_t.frame_window[g_t.pos[p, 0]] < g_t.cols else g_t.cols
                ymin_f = int(g_t.pos[p, 3]-g_t.frame_window[g_t.pos[p, 0]]) if g_t.pos[p, 3]-g_t.frame_window[g_t.pos[p, 0]] > 0 else 0
                ymax_f = int(g_t.pos[p, 3]+g_t.frame_window[g_t.pos[p, 0]]) if g_t.pos[p, 3]+g_t.frame_window[g_t.pos[p, 0]] < g_t.rows else g_t.rows

                dst2 = g_t.frame[ymin_f:ymax_f,xmin_f:xmax_f].copy()
                #if this subwindow has non size => prompt where is the error error
                if dst2.shape[0] == 0 or dst2.shape[1] == 0:
                    print("Error on class " + str( g_t.pos[p, 0] ) + " particle " + str(p) + " " + str(xmin_f) + " " + str(xmax_f) + " " + str(g_t.pos[0, 2]) + " " + str(g_t.pos[0, 3]) + " s " + str(dst2.shape))

                # Do the detection!
                pos2, dst2 = dsl.detect_particles(dst2, False, g_t.pos[p, 0], p, ymin_f, xmin_f, p_f_w, p_f_h)

                # If we have a particle, we update the results. We also compute the "speed" this particle has
                if pos2[0, 1] > 0:
                    g_t.vel[p, 2] = g_t.pos[p, 2]
                    g_t.pos[p, 2] = xmin_f+pos2[0, 2]#*multiplier_window
                    g_t.vel[p, 2] = g_t.pos[p, 2]-g_t.vel[p, 2]

                    g_t.vel[p, 3] = g_t.pos[p, 3]
                    g_t.pos[p, 3] = ymin_f+pos2[0, 3]#*multiplier_window
                    g_t.vel[p, 3] = g_t.pos[p, 3]-g_t.vel[p, 3]

                    # Save data
                    g_t.pos_time[p, 0, g_t.at_frame] = g_t.pos[p, 2]
                    g_t.pos_time[p, 1, g_t.at_frame] = g_t.pos[p, 3]
                    g_t.pos_time[p, 2, g_t.at_frame] = g_t.pos[p, 4]
                    g_t.pos_time[p, 3, g_t.at_frame] = g_t.pos[p, 5]

                    # Draw info
                    cv2.rectangle(g_t.dst, (int(g_t.pos[p, 2]-g_t.pos[p, 4]), int(g_t.pos[p, 3]-g_t.pos[p, 5])), (int(g_t.pos[p, 2]+g_t.pos[p, 4]), int(g_t.pos[p, 3]+g_t.pos[p, 5])), (int(g_t.color[p, 0]),int(g_t.color[p, 1]),int(g_t.color[p, 2])), 1)
                    # If you want to add some text to differentate classes in the frame, modify the next line:
                    cv2.putText(g_t.dst,str(p), (g_t.pos[p, 2], g_t.pos[p, 3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (int(g_t.color[p, 0]),int(g_t.color[p, 1]),int(g_t.color[p, 2])), 1)

                    # Update pos with speeds. The 2 is because I thought this operation could be very agressive.
                    g_t.pos[p, 2] += int(g_t.vel[p, 2]/2)
                    g_t.pos[p, 3] += int(g_t.vel[p, 3]/2)
                    g_t.vel[p, 2] = g_t.pos[p, 2]
                    g_t.vel[p, 3] = g_t.pos[p, 3]

                else:
                    # Remove particles non used. (only if they leave the window)
                    if g_t.pos[p, 2] < 1.25*g_t.pos[p, 4] or g_t.pos[p, 2] > g_t.cols-1.25*g_t.pos[p, 4] or g_t.pos[p, 3] < 1.25*g_t.pos[p, 5] or g_t.pos[p, 3] > g_t.rows-1.25*g_t.pos[p, 5]:
                        print("remove "+str(p))
                        iml_t.remove_particle(p)
                    else:
                        g_t.pos_time[p, 0, g_t.at_frame] = g_t.pos[p, 2]
                        g_t.pos_time[p, 1, g_t.at_frame] = g_t.pos[p, 3]
                        g_t.pos_time[p, 2, g_t.at_frame] = g_t.pos[p, 4]
                        g_t.pos_time[p, 3, g_t.at_frame] = g_t.pos[p, 5]
                        cv2.rectangle(g_t.dst, (int(g_t.pos[p, 2]-g_t.pos[p, 4]), int(g_t.pos[p, 3]-g_t.pos[p, 5])), (int(g_t.pos[p, 2]+g_t.pos[p, 4]), int(g_t.pos[p, 3]+g_t.pos[p, 5])), (int(g_t.color[p, 0]),int(g_t.color[p, 1]),int(g_t.color[p, 2])), 3)
                        # If you want to add some text to differentate classes in the frame, modify the next line:
                        cv2.putText(g_t.dst,str(p), (g_t.pos[p, 2], g_t.pos[p, 3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)

            # Let user control the frame after the detection, and redo vel with new info
            leave = iml_t.keyboard_control(1, True, show_image)
            if leave == 0:
                sv.save_data(filename)
                break

            # Add new particle if needed
            for p in range(g_t.ind_num_part_click):
                # (tmp_click)
                g_t.pos[g_t.ind_num_part[0], 0] = g_t.pos_tmp_click[p, 0]
                g_t.pos[g_t.ind_num_part[0], 1] = 1
                g_t.pos[g_t.ind_num_part[0], 2] = g_t.pos_tmp_click[p, 1]
                g_t.pos[g_t.ind_num_part[0], 3] = g_t.pos_tmp_click[p, 2]
                g_t.pos[g_t.ind_num_part[0], 4] = 0
                g_t.pos[g_t.ind_num_part[0], 5] = 0

                g_t.vel[g_t.ind_num_part[0], 0] = g_t.pos_tmp_click[p, 0]
                g_t.vel[g_t.ind_num_part[0], 1] = 1
                g_t.vel[g_t.ind_num_part[0], 2] = g_t.pos_tmp_click[p, 1]
                g_t.vel[g_t.ind_num_part[0], 3] = g_t.pos_tmp_click[p, 2]

                g_t.ind_lasts[g_t.pos[p, 0]] = g_t.ind_num_part[0]
                g_t.color[g_t.ind_num_part[0]] = iml_t.hls2rgb(iml_t.generate_hls())
                g_t.ind_num_part[0] += 1
                g_t.ind_num_part[g_t.pos_tmp_click[p, 0]+1] += 1

            g_t.ind_num_part_click = 0
            # (tmp_fast)
            if load_previous_values == True:
                counter_fast = 0
                for p in range(g_t.pos_tmp_fast_index, len(g_t.pos_tmp_fast)):
                    if g_t.pos_tmp_fast[p, 1] == g_t.at_frame:
                        g_t.pos[g_t.ind_num_part[0], 0] = g_t.pos_tmp_fast[p, 0]
                        g_t.pos[g_t.ind_num_part[0], 1] = 1
                        g_t.pos[g_t.ind_num_part[0], 2] = g_t.pos_tmp_fast[p, 2]
                        g_t.pos[g_t.ind_num_part[0], 3] = g_t.pos_tmp_fast[p, 3]
                        g_t.pos[g_t.ind_num_part[0], 4] = 0
                        g_t.pos[g_t.ind_num_part[0], 5] = 0

                        g_t.vel[g_t.ind_num_part[0], 0] = g_t.pos_tmp_fast[p, 0]
                        g_t.vel[g_t.ind_num_part[0], 1] = 1
                        g_t.vel[g_t.ind_num_part[0], 2] = g_t.pos_tmp_fast[p, 2]
                        g_t.vel[g_t.ind_num_part[0], 3] = g_t.pos_tmp_fast[p, 3]

                        g_t.ind_lasts[g_t.pos[p, 0]] = g_t.ind_num_part[0]
                        g_t.color[g_t.ind_num_part[0]] = iml_t.hls2rgb(iml_t.generate_hls())
                        g_t.ind_num_part[0] += 1
                        g_t.ind_num_part[g_t.pos_tmp_fast[p, 0]+1] += 1
                        counter_fast += 1
                    elif g_t.pos_tmp_fast[p, 1] > g_t.at_frame:
                        g_t.pos_tmp_fast_index += counter_fast
                        break
             
            # Save frame and read next
            out.write(g_t.dst)
            ret, g_t.frame = g_t.cap.read()
            if hasattr(g_t.frame, 'shape') == False:
                break
            g_t.at_frame += 1
            if g_t.at_frame % 100 == 0:
                timeee = time.time()-timeee
                print("Frame "+str(g_t.at_frame)+" time "+str(timeee))
                timeee = time.time()
                print("blob "+str(g_t.time_blob)+" detect "+str(g_t.time_detect)+" dist "+str(g_t.time_dist))
                g_t.time_blob = 0
                g_t.time_detect = 0
                g_t.time_dist = 0


        #------------------------------------------------------------------------
        #			             Destroy and leave
        #------------------------------------------------------------------------

        g_t.cap.release()
        out.release()
        cv2.destroyAllWindows()
        if leave != 0:
            sv.save_data(filename)
