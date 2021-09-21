"""
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

import settings as g
import IO_data as iol

#************************************************************************#
#                                                                        #
#				          Read number of frames                          #
#                                                                        #
#************************************************************************#



#------------------------------------------------------------------------#
#                                                                        #
#				          Read number of frames                          #
#                                                                        #
#   This is a fast way of calculating I thought. You could use           #
#   CV_CAP_PROP_FRAME_COUNT instead but sometimes this value is not      #
#   the real one...                                                      #
#------------------------------------------------------------------------#
def read_num_frames():
    g.cap.set(1,0) # Start in the initial frame

    print ('Reading number of frames for this video...')
    num_passed = 1000
    while(True):
        g.num_frames = g.num_frames + num_passed
        g.cap.set(1,g.num_frames)
        #print("pass "+str(num_passed)+" num_frames "+str(g.num_frames))
        if g.cap.grab() == False:
            #print("passsssss "+str(num_passed)+" num_frames "+str(g.num_frames))
            g.num_frames -= num_passed
            g.cap.set(1,g.num_frames-num_passed)
            if num_passed == 1:
                g.cap.set(1,0) # Start in the initial frame
                ret, g.frame = g.cap.read()
                break
            else:
                num_passed /= 10

    print ('Frames read. There are '+str(g.num_frames)+" frames.\n")

#------------------------------------------------------------------------#
#                                                                        #
#				          Read number of frames                          #
#                                                                        #
#   This is a slow way of calculating I thought. It avoids using         #
#   CV_CAP_PROP_FRAME_COUNT                                              #
#------------------------------------------------------------------------#
def read_num_frames_slow(Test=False):
    g.cap.set(1,0) # Start in the initial frame

    limit = 0
    print ('Reading number of frames for this video...')

    if Test == False:
        while(True):
            ret, g.frame = g.cap.read()
            if hasattr(g.frame, 'shape') == False:
                break
            limit = limit +1
    else:
        limit=7200

    g.cap.set(1,0) # Start in the initial frame
    ret, g.frame = g.cap.read()
    g.num_frames = limit

    print ('Frames read. There are '+str(g.num_frames)+" frames.\n")


#************************************************************************#
#************************************************************************#






#****************************************************************************#
#                                                                            #
#                      IN/OUT MOUSE/KEYBOARD control                         #
#                                                                            #
#****************************************************************************#

#----------------------------------------------------------------------------#
#                                                                            #
#                               Mouse Callback                               #
#                                                                            #
# Click method:                                                              #
#    *Right: Adds a new particle to the list                                 #
#    *Left: Adds a new (x,y) coordinate to the particle we are drawing       #
# Mouse wheel + ctrl key: Zoom in/out in the image                           #
# Mouse wheel: Go to next/previous frame                                     #
#                                                                            #
#----------------------------------------------------------------------------#
def draw_circle_and_zoom(event,x,y,flags,params):

#https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
#https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points

    # Add new coordinate to this particle
    if event == cv2.EVENT_LBUTTONDOWN:

        # If we don't have any, we will create one        
        if len(g.particles[ g.class_part ]) > 0 and g.particle_mode == False:
            g.particles[ g.class_part ][g.ind_lasts[g.class_part]][0].append([int(x/g.zoomSize)+g.start_old[0], int(y/g.zoomSize)+g.start_old[1]])
        else:
            create_particle(x,y)

        g.dst = g.frame.copy()
        g.dst = update_image(g.dst)
    elif event == cv2.EVENT_RBUTTONDOWN:
        create_particle(x,y)

        g.dst = g.frame.copy()
        g.dst = update_image(g.dst)
	#ZOOM IN/OUT It was working in ubuntu before using hide tk and qt... anyway, i left it here
    '''
    elif event == cv2.EVENT_MOUSEWHEEL and flags&0xFFFF==cv2.EVENT_FLAG_CTRLKEY:	# In my last version it seems that is not working.... wierd...
        print("Inner zoom")
        zoom(x,y,flags)
    elif event == cv2.EVENT_MOUSEWHEEL:
        change_frame(flags, 0)
    '''
#----------------------------------------------------------------------------#
#                                                                            #
#                Keyboard control while frames are showed                    #
#                                                                            #
#   @Inputs:                                                                 #
#     ms (int). Time in ms for cv2.waitKey function. It delays this          #
#         quantity between frames. I think the minimum is 1...               #
#     auto (True/False). If False it stops the images until user resumes     #
#         with an intro                                                      #
#     show (True/False). Show the window if the user asks for it             #
#                                                                            #
#----------------------------------------------------------------------------#
def keyboard_control(ms, auto, show):

    # Init variables
    finish = True
    dst_tmp = g.dst.copy()

    # Do always
    while(finish):
        # Show frame if asked
        if show == True:
            cv2.imshow("Main_window",g.dst)

        # Capture key every ms miliseconds
        k = cv2.waitKey(ms) & 0xFF
        #print('You pressed %d (0x%x), LSB: %d (%s)' % (k, k, k % 256, repr(chr(k%256)) if k%256 < 128 else '?'))
        if auto == True and g.stopped == False:
            finish = False

        # Answer user request
        if k == ord('q') or k == ord('Q') or k == 27:				#Quit frame
            g.dst = dst_tmp
            return 0
        elif k == ord('\r') or k == 32:						#Next frame
            leave = change_frame(+1, 0)
            if leave == 0:
                g.dst = dst_tmp
                return 0
        elif k == ord('c'):							#Ask for class to paint
            g.class_part = simpledialog.askinteger("Class", "Which object class do you want to paint?",
                                 parent=g.root,
                                 minvalue=1, maxvalue=g.classes_to_recognize)
            g.class_part = g.class_part - 1

        elif k == ord('a') or k == ord('A'):		#ASWD: Typical keys to move the position of the particle
            if g.ind_lasts[g.class_part] > -1:
                for i in range(len(g.particles[g.class_part][g.ind_lasts[g.class_part]][0])):
                    g.particles[g.class_part][g.ind_lasts[g.class_part]][0][i][0] -= 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)
        elif k == ord('d') or k == ord('D'):
            if g.ind_lasts[g.class_part] > -1:
                for i in range(len(g.particles[g.class_part][g.ind_lasts[g.class_part]][0])):
                    g.particles[g.class_part][g.ind_lasts[g.class_part]][0][i][0] += 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)
        elif k == ord('w') or k == ord('W'):
            if g.ind_lasts[g.class_part] > -1:
                for i in range(len(g.particles[g.class_part][g.ind_lasts[g.class_part]][0])):
                    g.particles[g.class_part][g.ind_lasts[g.class_part]][0][i][1] -= 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)
        elif k == ord('s') or k == ord('S'):
            if g.ind_lasts[g.class_part] > -1:
                for i in range(len(g.particles[g.class_part][g.ind_lasts[g.class_part]][0])):
                    g.particles[g.class_part][g.ind_lasts[g.class_part]][0][i][1] += 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)
        elif k == ord('r') or k == ord('R'):		#Remove this particle from dataset
            if g.ind_num_part[g.class_part+1] > 0:                
                del(g.particles[g.class_part][g.ind_lasts[g.class_part]])
                if g.ind_lasts[g.class_part] > 0:
                    g.ind_lasts[g.class_part] -= 1
                g.ind_num_part[g.class_part+1] -= 1
                g.ind_num_part[0] -= 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)

        elif k == ord('z') or k == ord('Z'):		#Move to the next or previous particle in this class
            if g.ind_lasts[g.class_part] > 0:
                g.ind_lasts[g.class_part] -= 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)

        elif k == ord('x') or k == ord('X'):
            if g.ind_lasts[g.class_part]+1 < g.ind_num_part[g.class_part+1]:
                g.ind_lasts[g.class_part] += 1

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)
        elif k == ord('i') or k == ord('I'):
            if g.num_info == True:
                g.num_info = False
            else:
                g.num_info = True

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)
        elif k == ord('m') or k == ord('M'):
            if g.particle_mode == True:
                g.particle_mode = False
            else:
                g.particle_mode = True

            g.dst = g.frame.copy()
            g.dst = update_image(g.dst)

#************************************************************************#
#************************************************************************#





#****************************************************************************#
#                                                                            #
#                          Particle's modification                           #
#                                                                            #
#****************************************************************************#


#----------------------------------------------------------------------------#
#                                                                            #
#                              Create particle                               #
#                                                                            #
# Add 1 particle to the index of particles of the class we are now and also  #
# add 1 particle to the total amount of particles of this class              #
# Finally, add a new particle with one (x,y) coordinates                     #
#----------------------------------------------------------------------------#
def create_particle(x,y):
    g.ind_lasts[g.class_part] = g.ind_num_part[g.class_part+1]
    g.ind_num_part[0] += 1	# Remember first element is for total of particles
    g.ind_num_part[g.class_part+1] += 1

    # If we have constant size
    if g.particle_mode:
        corners = [[0,0],[0,0],[0,0],[0,0]]
        C = [g.w*g.zoomSize, g.h*g.zoomSize]
        xx = x*g.zoomSize
        yy = y*g.zoomSize
        corners[0] = [xx-C[0], yy-C[1]]
        corners[1] = [xx-C[0], yy+C[1]]
        corners[2] = [xx+C[0], yy+C[1]]
        corners[3] = [xx+C[0], yy-C[1]]

        for cor in range(len(corners)):
            for xy in range(2):
                if corners[cor][xy] < 0:
                    corners[cor][xy] = 0
                if corners[cor][xy] > g.cols*g.zoomSize and xy == 0:
                    corners[cor][xy] = g.cols*g.zoomSize -1
                if corners[cor][xy] > g.rows*g.zoomSize and xy == 1:
                    corners[cor][xy] = g.rows*g.zoomSize -1

        g.particles[ g.class_part ].append([[ [ int(corners[0][0]/g.zoomSize)+g.start_old[0], int(corners[0][1]/g.zoomSize)+g.start_old[1] ] ]])
        g.particles[ g.class_part ][g.ind_lasts[g.class_part]][0].append([int(corners[1][0]/g.zoomSize)+g.start_old[0], int(corners[1][1]/g.zoomSize)+g.start_old[1]])
        g.particles[ g.class_part ][g.ind_lasts[g.class_part]][0].append([int(corners[2][0]/g.zoomSize)+g.start_old[0], int(corners[2][1]/g.zoomSize)+g.start_old[1]])
        g.particles[ g.class_part ][g.ind_lasts[g.class_part]][0].append([int(corners[3][0]/g.zoomSize)+g.start_old[0], int(corners[3][1]/g.zoomSize)+g.start_old[1]])
    # Otherwise
    else:
        g.particles[ g.class_part ].append([[ [ int(x/g.zoomSize)+g.start_old[0], int(y/g.zoomSize)+g.start_old[1] ] ]])

#----------------------------------------------------------------------------#
#                             Rotate your squares                            #
#                                                                            #
# I do a rotation of an angle to the points taking as a reference (0,0) for  #
# the rotation the point given at reference.                                 #
#                                                                            #
#----------------------------------------------------------------------------#
def Rotate_pos_time(cnt, angle):
    #https://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation

    cosinus = np.cos(angle)
    sinus = np.sin(angle)

    cnt_rotated = np.zeros((4,2)).astype(int)
    x0 = g.cols/2
    y0 = g.rows/2

    for p in range(len(cnt)):
        x = cnt[p, 0]
        y = cnt[p, 1]

        cnt_rotated[p, 0] = int((x-x0)*cosinus - (y-y0)*sinus + x0)
        cnt_rotated[p, 1] = int((x-x0)*sinus + (y-y0)*cosinus + y0)

    return cnt_rotated

#----------------------------------------------------------------------------#
#                               Create rectangle                             #
#                                                                            #
# Creates a rectangle around the particle p with class c using its vertexes  #
# It will use the colour selected or unselected depending if the index says  #
# we are in that p particle of that c class                                  #
# We can also do not draw the border of the box, but the vertex number       #
# (draw=false) #
#                                                                            #
#----------------------------------------------------------------------------#
def create_rectangle(c, p, draw, image, colour_selected_box, colour_unselected_box):
    cnt = np.array(g.particles[c][p][0])
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if draw:
        if g.class_part == c and p == g.ind_lasts[g.class_part]:
            cv2.drawContours(image,[box],0,colour_selected_box,2)
        else:
            cv2.drawContours(image,[box],0,colour_unselected_box,2)
    
    for i in range(len(box)):
        if g.num_info == True:
            cv2.putText(image,str(box[i,0])+", "+str(box[i,1]), (box[i,0], box[i,1]), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_unselected_box, 1)
    return box


#************************************************************************#
#************************************************************************#





#****************************************************************************#
#                                                                            #
#                             Image manipulation                             #
#                                                                            #
#****************************************************************************#


#----------------------------------------------------------------------------#
#                                                                            #
#                                    Zoom                                    #
#                                                                            #
# Zoom method: Makes zoom in the frame                                       #
#   @Inputs:                                                                 #
#     x (int). x coordinate. 0 is left border                                #
#     y (int). y coordinate. 0 is top border                                 #
#     flags (float). If positive, we zoom in. If negative, we zoom out       #
#                                                                            #
#----------------------------------------------------------------------------#
def zoom(x,y,flags):
    zoomold = g.zoomSize
    if flags > 0:  # Scroll up
        g.zoomSize = g.zoomSize*2
    elif flags < 0:    # Scroll down
        g.zoomSize = g.zoomSize*0.5

    #Limit number of zooms in and out. Maximum 16x. Minimum 1x. 1x is the original size
    if g.zoomSize < 1:
        g.zoomSize = g.zoomSize*2
    elif g.zoomSize > 17:
        g.zoomSize = g.zoomSize/2
    else:
        x = x/zoomold + g.start_old[0]
        y = y/zoomold + g.start_old[1]
        #I try to center it where the mouse is!
        g.dst = g.frame.copy()
        center = [int(x*g.zoomSize),int(y*g.zoomSize)]
        start = [center[0]-g.cols/2,center[1]-g.rows/2]
        end = [center[0]+g.cols/2,center[1]+g.rows/2]

        if start[0]<0:
                end[0] = end[0]+start[0]*-1
                start[0] = 0
        if start[1]<0:
                end[1] = end[1]+start[1]*-1
                start[1] = 0
        if end[0]>g.cols*g.zoomSize:
                start[0] = start[0]-(end[0]-g.cols*g.zoomSize)
                end[0] = g.cols*g.zoomSize
        if end[1]>g.rows*g.zoomSize:
                start[1] = start[1]-(end[1]-g.rows*g.zoomSize)
                end[1] = g.rows*g.zoomSize
        g.start_old[0] = int((start[0])/g.zoomSize)
        g.start_old[1] = int((start[1])/g.zoomSize)

        g.dst = update_image(g.dst)

#----------------------------------------------------------------------------#
#                                                                            #
#                                Change frame                                #
#                                                                            #
#   @Inputs:                                                                 #
#     flags (number). Positive: move forward.                                #
#                     Negative: move backwards.                              #
#                     Zero: go to frame gotof. ERASE ALL POS, VEL, COLOR AND #
#                           INDEXES RELATIVE TO PARTICLES                    #
#     gotof (int). If flag == 0 then moves to frame gotof                    #
#                                                                            #
#----------------------------------------------------------------------------#
def change_frame(flags, gotof):

    # Select how to go to that frame
    if flags > 0:  			# Scroll up
            t_lastf = g.at_frame+g.change_frame_num
    elif flags < 0:   		# Scroll down
            t_lastf = g.at_frame-g.change_frame_num
    else:    				# Go to this frame
            t_lastf = gotof

    iol.save_images()
    g.particles = [[] for i in range(g.classes_to_recognize)]
    g.ind_lasts = np.full((g.classes_to_recognize),-1).astype(int)
    g.ind_num_part = np.zeros((g.classes_to_recognize+1)).astype(int) #First is total!

    # Move to that frame
    if t_lastf < 0:
            g.at_frame = 0
    elif t_lastf >= g.num_frames:
            g.at_frame = 0
            return 0
    else:
            g.at_frame = t_lastf

    g.cap.set(1,g.at_frame) # Start in the initial frame
    ret, g.frame = g.cap.read()
    g.dst = g.frame.copy()
    g.dst = update_image(g.dst)

    text = str(g.at_frame)+"/"+str(g.num_frames)
    cv2.putText(g.dst, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return 1
#------------------------------------------------------------------------#
#                                                                        #
#				Update image with squares around particles               #
#                                                                        #
#   @Inputs:                                                             #
#     image (frame). Frame to update                                     #
#                                                                        #
#------------------------------------------------------------------------#
def update_image(image):

    # Apply zoom correctly
    image = imutils.resize(image, width=(int(g.zoomSize * image.shape[1])))

    corners = [[0,0],[0,0],[0,0],[0,0]]

    # If it is of this kind and it is selected: blue.
    colour_selected = (255,0,0)
    colour_selected_box = (255,100,0)

    # For every kind of object
    for c in range(g.classes_to_recognize):

        # If it is an object of this kind, green. Otherwise, red
        if g.class_part == c:
            colour_unselected = (0,255,0)
            colour_unselected_box = (100,255,100)
        else:
            colour_unselected = (0,0,255)
            colour_unselected_box = (100,100,255)

        # For each particle
        for p in range(len(g.particles[c])):   
            # Draw vertexes clicked
            for i in range(len(g.particles[c][p][0])):
                if g.class_part == c and p == g.ind_lasts[g.class_part]:
                    cv2.circle(image, (g.particles[c][p][0][i][0], g.particles[c][p][0][i][1]), 4, colour_selected)
                else:
                    cv2.circle(image, (g.particles[c][p][0][i][0], g.particles[c][p][0][i][1]), 4, colour_unselected)          

            # WIth more than 2 points we try making a box
            if len(g.particles[c][p][0]) > 2:
                create_rectangle(c, p, True, image, colour_selected_box, colour_unselected_box)

        #if g.num_info == True:
        #    cv2.putText(image,str(p), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_unselected, 1)

    return image[int(g.start_old[1]*g.zoomSize):int(g.start_old[1]*g.zoomSize+g.rows), int(g.start_old[0]*g.zoomSize):int(g.start_old[0]*g.zoomSize+g.cols)]

