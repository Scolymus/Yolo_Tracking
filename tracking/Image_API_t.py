"""
Created on Thu Nov 14 2019

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
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"tracking")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import settings_t as g_t

#------------------------------------------------------------------------#
#                                                                        #
#                         Create dashed squares                          #
#       (Grabbed from Albert's tracking code (py 2.7; cv 3.0 beta)       #
#                                                                        #
#   @Inputs:                                                             #
#     img (frame). Image where to draw                                   #
#     point1 ([x, y]). Top-left corner of the square                     #
#     point1 ([x, y]). Bottom-right corner of the square                 #
#     dashSize (int). Space between dashes                               #
#     color ( (R,B,G) ). RGB colour. From 0 to 255                       #
#     thickness (int). Line thickness                                    #
#                                                                        #
#------------------------------------------------------------------------#
def chunks(l,n):
    #Generator that returns consecutive pieces of the "l" list of size "n"
    for i in range(0,len(l), n):
        yield l[i:(i+n)]

def get_line_points(start,end):
    #Return all the points in a line going from start to end
    dist = int(round(np.linalg.norm(end-start)))
    x = np.linspace(start[0],end[0],dist)
    y = np.linspace(start[1],end[1],dist)
    x, y = np.around(x).astype(int), np.around(y).astype(int)
    return [x,y]

def dashed_line(img, point1, point2, dashSize, color, thickness):
    #Draws a dashed line. Dashes are "dashSize" pixel in length
    point1 = np.array(point1); point2 = np.array(point2)
    x, y = get_line_points(point1,point2)
    xiter, yiter = chunks(x,dashSize), chunks(y,dashSize)
    while True:
        try:
            #Method 1
            x,y = next(xiter), next(yiter)
            p1 = (x[0], y[0]); p2 = (x[-1], y[-1])
            cv2.line(img, p1, p2, color, thickness)
            #Method 2
#            img[yiter.next(), xiter.next()] = color #Paint dash

            next(xiter); next(yiter) #Leave dash blank

        except StopIteration:
            break
    return img

def hls2rgb(colour):
    #colour hls
    tmp = np.zeros(3)
    colour = colorsys.hls_to_rgb(colour[0],colour[1],colour[2])
    tmp[0] = 255*colour[0]
    tmp[1] = 255*colour[1]
    tmp[2] = 255*colour[2]
    #print(255*np.array(colorsys.hls_to_rgb(colour[0],colour[1],colour[2])).astype(int))
    #return 255*np.array(colorsys.hls_to_rgb(colour[0],colour[1],colour[2])).astype(int).flatten()
    return tmp.astype(int)
def generate_hls():
    return [np.random.randint(0,240,1)/240, np.random.randint(80,210,1)/240, np.random.randint(100,240,1)/240]


#----------------------------------------------------------------------------#
#                                                                            #
#                               Mouse Callback                               #
#                                                                            #
# Click method: Add a new particle of the current class at (x,y) coordinates #
# Mouse wheel + ctrl key: Zoom in/out in the image                           #
# Mouse wheel: Go to next/previous frame                                     #
#                                                                            #
#----------------------------------------------------------------------------#
def draw_circle_and_zoom(event,x,y,flags,params):

    #Exit if we are processing some data
    if g_t.stopped == False and event == cv2.EVENT_LBUTTONDOWN:
        print(g_t.ind_num_part[g_t.class_part])
        g_t.pos_tmp_click[g_t.ind_num_part_click, 0] = g_t.class_part
        g_t.pos_tmp_click[g_t.ind_num_part_click, 1] = int(x/g_t.zoomSize)+g_t.start_old[0]
        g_t.pos_tmp_click[g_t.ind_num_part_click, 2] = int(y/g_t.zoomSize)+g_t.start_old[1]
        g_t.ind_num_part_click += 1
        return

    if g_t.stopped == False:
        return

    # Left Mouse Button Down Pressed for adding a new particle
    if event == cv2.EVENT_LBUTTONDOWN:
        g_t.pos[ g_t.ind_num_part[0], 0] = g_t.class_part
        # Since the size of pos_time is the same, I cannot expect to have more particles in g_t.pos
        # Thus, I will add the index 1: 1 is active, -1 inactive.
        # This is different behaviour than in pos_tmp_fast because there, I don't track and don't
        # save in pos_time
        g_t.pos[ g_t.ind_num_part[0], 1] = 1
        g_t.pos[ g_t.ind_num_part[0], 2] = int(x/g_t.zoomSize)+g_t.start_old[0]
        g_t.pos[ g_t.ind_num_part[0], 3] = int(y/g_t.zoomSize)+g_t.start_old[1]

        g_t.ind_lasts[g_t.class_part] = g_t.ind_num_part[0]
        g_t.color[g_t.ind_num_part[0]] = hls2rgb(generate_hls())
        g_t.ind_num_part[0] += 1
        g_t.ind_num_part[g_t.class_part+1] += 1

        g_t.dst = g_t.frame.copy()
        g_t.dst = update_image(g_t.dst, False)

	#ZOOM IN/OUT
    elif event == cv2.EVENT_MOUSEWHEEL and flags&0xFFFF==cv2.EVENT_FLAG_CTRLKEY:
        zoom(x,y,flags)
    elif event == cv2.EVENT_MOUSEWHEEL:
        change_frame(flags, 0, False)

#----------------------------------------------------------------------------#
#                                                                            #
#                               Mouse Callback                               #
#                                                                            #
# This method is a variation of the normal one for when the user does the    #
# initial search of particles along all frames                               #
#                                                                            #
# Click method: Add a new particle of the current class at (x,y) coordinates #
# Mouse wheel + ctrl key: Zoom in/out in the image                           #
# Mouse wheel: Go to next/previous frame                                     #
#                                                                            #
#----------------------------------------------------------------------------#
def draw_circle_and_zoom_fast(event,x,y,flags,params):

    #Exit if we are processing some data
    if g_t.stopped == False and event == cv2.EVENT_LBUTTONDOWN:
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 0] = g_t.class_part
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 1] = g_t.at_frame
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 2] = int(x/g_t.zoomSize)+g_t.start_old[0]
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 3] = int(y/g_t.zoomSize)+g_t.start_old[1]

        g_t.ind_num_part[0] += 1
        g_t.ind_num_part[g_t.class_part+1] += 1

        return
    if g_t.stopped == False:
        return

    # Left Mouse Button Down Pressed for adding a new particle
    if event == cv2.EVENT_LBUTTONDOWN:
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 0] = g_t.class_part
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 1] = g_t.at_frame
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 2] = int(x/g_t.zoomSize)+g_t.start_old[0]
        g_t.pos_tmp_fast[ g_t.ind_num_part[0], 3] = int(y/g_t.zoomSize)+g_t.start_old[1]

        g_t.ind_num_part[0] += 1
        g_t.ind_num_part[g_t.class_part+1] += 1

        g_t.dst = g_t.frame.copy()
        g_t.dst = update_image(g_t.dst, True)

	#ZOOM IN/OUT
    elif event == cv2.EVENT_MOUSEWHEEL and flags&0xFFFF==cv2.EVENT_FLAG_CTRLKEY:
        zoom(x,y,flags)
    elif event == cv2.EVENT_MOUSEWHEEL:
        change_frame(flags, 0, True)

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
    zoomold = g_t.zoomSize
    if flags > 0:  # Scroll up
        g_t.zoomSize = g_t.zoomSize*2
    elif flags < 0:    # Scroll down
        g_t.zoomSize = g_t.zoomSize*0.5

    #Limit number of zooms in and out. Maximum 16x. Minimum 1x. 1x is the original size
    if g_t.zoomSize < 1:
        g_t.zoomSize = g_t.zoomSize*2
    elif g_t.zoomSize > 17:
        g_t.zoomSize = g_t.zoomSize/2
    else:
        x = x/zoomold + g_t.start_old[0]
        y = y/zoomold + g_t.start_old[1]
        #I try to center it where the mouse is!
        g_t.dst = g_t.frame.copy()
        center = [int(x*g_t.zoomSize),int(y*g_t.zoomSize)]
        start = [center[0]-g_t.cols/2,center[1]-g_t.rows/2]
        end = [center[0]+g_t.cols/2,center[1]+g_t.rows/2]

        if start[0]<0:
                end[0] = end[0]+start[0]*-1
                start[0] = 0
        if start[1]<0:
                end[1] = end[1]+start[1]*-1
                start[1] = 0
        if end[0]>g_t.cols*g_t.zoomSize:
                start[0] = start[0]-(end[0]-g_t.cols*g_t.zoomSize)
                end[0] = g_t.cols*g_t.zoomSize
        if end[1]>g_t.rows*g_t.zoomSize:
                start[1] = start[1]-(end[1]-g_t.rows*g_t.zoomSize)
                end[1] = g_t.rows*g_t.zoomSize
        g_t.start_old[0] = int((start[0])/g_t.zoomSize)
        g_t.start_old[1] = int((start[1])/g_t.zoomSize)

        g_t.dst = update_image(g_t.dst, False)

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
#     fast (True/false). True if we are not tracking, but going over all the #
#                        video to click on new particles.                    #
#                                                                            #
#----------------------------------------------------------------------------#
def change_frame(flags, gotof, fast):

    # Select how to go to that frame
    if flags > 0:  			# Scroll up
            t_lastf = g_t.at_frame+g_t.change_frame_num
    elif flags < 0:   		# Scroll down
            t_lastf = g_t.at_frame-g_t.change_frame_num
    else:    				# Go to this frame
            t_lastf = gotof

    # Move to that frame
    if t_lastf < 0:
            g_t.at_frame = 0
    elif t_lastf > g_t.num_frames:
            g_t.at_frame = g_t.num_frames
    else:
        if t_lastf != g_t.at_frame:
            g_t.at_frame = t_lastf

            g_t.pos = np.zeros((g_t.number_of_particles_to_init,6)).astype(int)
            g_t.vel = np.zeros((g_t.number_of_particles_to_init,4)).astype(int)
            g_t.ind_lasts = np.full((g_t.classes_to_recognize),-1).astype(int)
            g_t.ind_num_part = np.zeros(g_t.classes_to_recognize+1).astype(int) #First is total!
            g_t.color = np.zeros((g_t.number_of_particles_to_init,3)).astype(int)   #particle, RGB

    g_t.cap.set(1,g_t.at_frame) # Start in the initial frame
    ret, g_t.frame = g_t.cap.read()
    g_t.dst = g_t.frame.copy()
    g_t.dst = update_image(g_t.dst, fast)

    text = str(g_t.at_frame)+"/"+str(g_t.num_frames)
    cv2.putText(g_t.dst, text, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


#------------------------------------------------------------------------#
#                                                                        #
#				Update image with squares around particles               #
#                                                                        #
#   @Inputs:                                                             #
#     image (frame). Frame to update                                     #
#     fast (True/False). If true, take data from g_t.pos_tmp_fast. If      #
#                        false, take data from g_t.pos                     #
#                                                                        #
#------------------------------------------------------------------------#
def update_image(image, fast):

    # Apply zoom correctly
    image = imutils.resize(image, width=(int(g_t.zoomSize * image.shape[1])))

    corners = [[0,0],[0,0],[0,0],[0,0]]

    # If it is of this kind and it is selected: blue.
    colour_selected = (255,0,0)

    # For every kind of object
    for p in range(g_t.ind_num_part[0]):
        if g_t.pos[p, 1] < 0:
            continue
        if fast == False:
            xx = g_t.pos[p, 2]*g_t.zoomSize
            yy = g_t.pos[p, 3]*g_t.zoomSize
            cc = g_t.pos[p, 0]

            # If it is an object of this kind, green. Otherwise, red
            if g_t.pos[p, 0] == g_t.class_part:
                colour_unselected = (0,255,0)
            else:
                colour_unselected = (0,0,255)

            corners[0] = [xx-g_t.pos[p, 4], yy-g_t.pos[p, 5]]
            corners[1] = [xx-g_t.pos[p, 4], yy+g_t.pos[p, 5]]
            corners[2] = [xx+g_t.pos[p, 4], yy+g_t.pos[p, 5]]
            corners[3] = [xx+g_t.pos[p, 4], yy-g_t.pos[p, 5]]

            pts = np.array([corners[0],corners[1],corners[2],corners[3]], np.int32)
            pts = pts.reshape((-1,1,2))
            if g_t.class_part == cc and p == g_t.ind_lasts[g_t.class_part]:
                dashed_line(image, corners[0], corners[1], 2, colour_selected, 2)
                dashed_line(image, corners[1], corners[2], 2, colour_selected, 2)
                dashed_line(image, corners[2], corners[3], 2, colour_selected, 2)
                dashed_line(image, corners[3], corners[0], 2, colour_selected, 2)
            else:
                cv2.polylines(image, [pts], True, colour_unselected)

        else:
            xx = g_t.pos_tmp_fast[p, 2]*g_t.zoomSize
            yy = g_t.pos_tmp_fast[p, 3]*g_t.zoomSize
            cc = g_t.pos_tmp_fast[p, 0]

            # If it is an object of this kind, green. Otherwise, red
            if g_t.pos_tmp_fast[p, 0] == g_t.class_part:
                colour_unselected = (0,255,0)
            else:
                colour_unselected = (0,0,255)

            if g_t.class_part == cc and p == g_t.ind_lasts[g_t.class_part]:
                cv2.circle(image, (xx, yy), 4, colour_selected)
            else:
                cv2.circle(image, (xx, yy), 4, colour_unselected)

        if g_t.num_info == True:
            cv2.putText(image,str(p), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 1, colour_unselected, 1)

    return image[int(g_t.start_old[1]*g_t.zoomSize):int(g_t.start_old[1]*g_t.zoomSize+g_t.rows), int(g_t.start_old[0]*g_t.zoomSize):int(g_t.start_old[0]*g_t.zoomSize+g_t.cols)]

#------------------------------------------------------------------------#
#                                                                        #
#				          Read number of frames                          #
#                                                                        #
#   This is a fast way of calculating I thought. You could use           #
#   CV_CAP_PROP_FRAME_COUNT instead but sometimes this value is not      #
#   the real one...                                                      #
#------------------------------------------------------------------------#
def read_num_frames():
    g_t.cap.set(1,0) # Start in the initial frame

    print ('Reading number of frames for this video...')
    num_passed = 1000
    while(True):
        g_t.num_frames = g_t.num_frames + num_passed
        g_t.cap.set(1,g_t.num_frames)
        print("pass "+str(num_passed)+" num_frames "+str(g_t.num_frames))
        if g_t.cap.grab() == False:
            print("passsssss "+str(num_passed)+" num_frames "+str(g_t.num_frames))
            g_t.num_frames -= num_passed
            g_t.cap.set(1,g_t.num_frames-num_passed)
            if num_passed == 1:
                g_t.cap.set(1,0) # Start in the initial frame
                ret, g_t.frame = g_t.cap.read()
                break
            else:
                num_passed /= 10

    print ('Frames read. There are '+str(g_t.num_frames)+" frames.\n")

#------------------------------------------------------------------------#
#                                                                        #
#				          Read number of frames                          #
#                                                                        #
#   This is a fast way of calculating I thought. You could use           #
#   CV_CAP_PROP_FRAME_COUNT instead but sometimes this value is not      #
#   the real one...                                                      #
#------------------------------------------------------------------------#
def read_num_frames_slow(Test=False):
    g_t.cap.set(1,0) # Start in the initial frame

    limit = 0
    print ('Reading number of frames for this video...')

    if Test == False:
        while(True):
            ret, g_t.frame = g_t.cap.read()
            if hasattr(g_t.frame, 'shape') == False:
                break
            limit = limit +1
    else:
        limit=7200

    g_t.cap.set(1,0) # Start in the initial frame
    ret, g_t.frame = g_t.cap.read()
    g_t.num_frames = limit

    print ('Frames read. There are '+str(g_t.num_frames)+" frames.\n")

#------------------------------------------------------------------------#
#                                                                        #
#				Keyboard control while frames are showed                 #
#                                                                        #
#   @Inputs:                                                             #
#     ms (int). Time in ms for cv2.waitKey function. It delays this      #
#         quantity between frames. I think the minimum is 1...           #
#     auto (True/False). If False it stops the images until user resumes #
#         with an intro                                                  #
#     show (True/False). Show the window if the user asks for it         #
#                                                                        #
#------------------------------------------------------------------------#
def keyboard_control(ms, auto, show):

    # Init variables
    finish = True
    dst_tmp = g_t.dst.copy()

    # Do always
    while(finish):
        # Show frame if asked
        if show == True:
            cv2.imshow("Main_window",g_t.dst)

        k = cv2.waitKey(ms) & 0xFF
        #print('You pressed %d (0x%x), LSB: %d (%s)' % (k, k, k % 256, repr(chr(k%256)) if k%256 < 128 else '?'))
        if auto == True and g_t.stopped == False:
            finish = False

        if k == ord('q') or k == ord('Q'):			#Quit frame
            g_t.dst = dst_tmp
            return 0
        elif k == ord('\r'):						#Stop/Resume
            if g_t.stopped == False:
                g_t.stopped = True
                finish = True
            else:
                g_t.stopped = False
                #finish = False
                g_t.dst = dst_tmp
                return 1
        elif k == ord('c'):							#Ask for class to paint
            g_t.class_part = simpledialog_t.askinteger("Class", "Which object class do you want to paint?",
                                 parent=g_t.root,
                                 minvalue=0, maxvalue=g_t.classes_to_recognize-1)
        elif k == ord('a') or k == ord('A'):		#ASWD: Typical keys to move the position of the particle
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos[g_t.ind_lasts[g_t.class_part], 2] -= 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('d') or k == ord('D'):
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos[g_t.ind_lasts[g_t.class_part], 2] += 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('w') or k == ord('W'):
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos[g_t.ind_lasts[g_t.class_part], 3] -= 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('s') or k == ord('S'):
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos[g_t.ind_lasts[g_t.class_part], 3] += 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('r') or k == ord('R'):		#Remove this particle from dataset
            remove_particle(g_t.ind_lasts[g_t.class_part])
            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('z') or k == ord('Z'):		#Move to the next or previous particle in this class
            if g_t.ind_lasts[g_t.class_part] > 0:
                particles_this_class = np.where(g_t.pos[:, 0] == g_t.class_part)
                particles_this_class_now = np.where(particles_this_class[0] == g_t.ind_lasts[g_t.class_part])
                if particles_this_class_now[0][0] > 0:
                    entered = False
                    print(particles_this_class_now[0])
                    for i in range(particles_this_class_now[0][0]-1, -1, -1):
                        if g_t.pos[particles_this_class[0][i], 1] > 0:
                            g_t.ind_lasts[g_t.class_part] = particles_this_class[0][i]
                            entered = True
                            break
                    if entered == False:
                        print("EVento -1 entered")
                        g_t.ind_lasts[g_t.class_part] = -1
                else:
                    print("EVento -1 else")
                    g_t.ind_lasts[g_t.class_part] = -1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('x') or k == ord('X'):
            particles_this_class = np.where(g_t.pos[:, 0] == g_t.class_part)
            particles_this_class_now = np.where(particles_this_class[0] == g_t.ind_lasts[g_t.class_part])
            print(g_t.ind_lasts[g_t.class_part])
            print(g_t.class_part)
            print(particles_this_class_now)
            if len(particles_this_class_now[0]) > 0: #There is a bug here, but I cannot see where it comes from
                if particles_this_class_now[0][0]+1 < len(particles_this_class[0]):
                    for i in range(particles_this_class_now[0][0]+1, len(particles_this_class[0])):
                        if g_t.pos[particles_this_class[0][i], 1] > 0:
                            g_t.ind_lasts[g_t.class_part] = particles_this_class[0][i]
                            break

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)
        elif k == ord('i') or k == ord('I'):
            if g_t.num_info == True:
                g_t.num_info = False
            else:
                g_t.num_info = True

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)

#------------------------------------------------------------------------#
#                                                                        #
#				Keyboard control while frames are showed                 #
#                                                                        #
#   @Inputs:                                                             #
#     ms (int). Time in ms for cv2.waitKey function. It delays this      #
#         quantity between frames. I think the minimum is 1...           #
#     auto (True/False). If False it stops the images until user resumes #
#         with an intro                                                  #
#     show (True/False). Show the window if the user asks for it         #
#                                                                        #
#------------------------------------------------------------------------#
def keyboard_control_fast(ms, auto, show):

    # Init variables
    finish = True
    dst_tmp = g_t.dst.copy()

    # Do always
    while(finish):
        # Show frame if asked
        if show == True:
            cv2.imshow("Main_window",g_t.dst)

        k = cv2.waitKey(ms) & 0xFF
        #print('You pressed %d (0x%x), LSB: %d (%s)' % (k, k, k % 256, repr(chr(k%256)) if k%256 < 128 else '?'))
        if auto == True and g_t.stopped == False:
            finish = False

        if k == ord('q') or k == ord('Q'):			#Quit frame
            g_t.dst = dst_tmp
            return 0
        elif k == ord('\r'):						#Stop/Resume
            if g_t.stopped == False:
                g_t.stopped = True
                finish = True
            else:
                g_t.stopped = False
                #finish = False
                g_t.dst = dst_tmp
                return 1
        elif k == ord('c'):							#Ask for class to paint
            g_t.class_part = simpledialog_t.askinteger("Class", "Which object class do you want to paint?",
                                 parent=g_t.root,
                                 minvalue=0, maxvalue=g_t.classes_to_recognize-1)
        elif k == ord('a') or k == ord('A'):		#ASWD: Typical keys to move the position of the particle
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos_tmp_fast[g_t.ind_lasts[g_t.class_part], 2] -= 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, True)
        elif k == ord('d') or k == ord('D'):
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos_tmp_fast[g_t.ind_lasts[g_t.class_part], 2] += 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, True)
        elif k == ord('w') or k == ord('W'):
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos_tmp_fast[g_t.ind_lasts[g_t.class_part], 3] -= 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, True)
        elif k == ord('s') or k == ord('S'):
            if g_t.ind_lasts[g_t.class_part] > -1:
                g_t.pos_tmp_fast[g_t.ind_lasts[g_t.class_part], 3] += 1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, True)
        elif k == ord('n') or k == ord('N'):		#Next frame
            change_frame(1,0,True)
        elif k == ord('p') or k == ord('P'):		#Previous frame
            change_frame(-1,0,True)
        elif k == ord('f') or k == ord('F'):		#Frame you want to go. Should I remove it???
            frame_to_go = simpledialog_t.askinteger("Class", "To which frame do you want to move?",
                                 parent=g_t.root,
                                 minvalue=0, maxvalue=g_t.num_frames-1)
            change_frame(0,int(frame_to_go),True)
        elif k == ord('r') or k == ord('R'):		#Remove this particle from dataset
            remove_particle_fast(g_t.ind_lasts[g_t.class_part])
            update_image(g_t.dst, False)
        elif k == ord('z') or k == ord('Z'):		#Move to the next or previous particle in this class
            if g_t.ind_lasts[g_t.class_part] > 0:
                particles_this_class = np.where(g_t.pos_tmp_fast[:, 0] == g_t.class_part)
                particles_this_class_now = np.where(particles_this_class[0] == g_t.ind_lasts[g_t.class_part])
                if particles_this_class_now[0][0] > 0:
                    entered = False
                    for i in range(particles_this_class_now[0][0]-1, -1, -1):
                        if g_t.pos_tmp_fast[particles_this_class[0][i], 1] > 0:
                            g_t.ind_lasts[g_t.class_part] = particles_this_class[0][i]
                            entered = True
                            break
                    if entered == False:
                        g_t.ind_lasts[g_t.class_part] = -1
                else:
                    g_t.ind_lasts[g_t.class_part] = -1

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, True)
        elif k == ord('x') or k == ord('X'):
            particles_this_class = np.where(g_t.pos_tmp_fast[:, 0] == g_t.class_part)
            particles_this_class_now = np.where(particles_this_class[0] == g_t.ind_lasts[g_t.class_part])
            if particles_this_class_now[0][0]+1 < len(particles_this_class[0]):
                for i in range(particles_this_class_now[0][0]+1, len(particles_this_class[0])):
                    if g_t.pos_tmp_fast[particles_this_class[0][i], 1] > 0:
                        g_t.ind_lasts[g_t.class_part] = particles_this_class[0][i]
                        break

            g_t.dst = g_t.frame.copy()
            g_t.dst = update_image(g_t.dst, False)

#------------------------------------------------------------------------#
#                                                                        #
#                     Remove particle from the dataset                   #
#                                                                        #
#   @Inputs:                                                             #
#     p (int). Particle number                                           #
#                                                                        #
#------------------------------------------------------------------------#
def remove_particle(p):
    if p < g_t.ind_num_part[0]:
        g_t.pos[p, 1] = -1
        print(g_t.ind_lasts[g_t.class_part])
        particles_this_class = np.where(g_t.pos[:, 0] == g_t.class_part)
        particles_this_class_now = np.where(particles_this_class[0] == g_t.ind_lasts[g_t.class_part])

        entered = False
        if particles_this_class_now[0][0] > 0:
            for i in range(particles_this_class_now[0][0]-1, -1, -1):
                if g_t.pos[particles_this_class[0][i], 1] > 0:
                    g_t.ind_lasts[g_t.class_part] = particles_this_class[0][i]
                    entered = True
                    break

        if (entered == False) and (particles_this_class_now[0][0]+1 < len(particles_this_class[0])):
            for i in range(particles_this_class_now[0][0]+1, len(particles_this_class[0])):
                if g_t.pos[particles_this_class[0][i], 1] > 0:
                    g_t.ind_lasts[g_t.class_part] = particles_this_class[0][i]
                    entered = True
                    break
                    
        if entered == False:
            g_t.ind_lasts[g_t.class_part] = -1

        print(g_t.ind_lasts[g_t.class_part])

#------------------------------------------------------------------------#
#                                                                        #
#                 Remove particle from the fast dataset                  #
#                                                                        #
#   @Inputs:                                                             #
#     p (int). Particle number                                           #
#                                                                        #
#------------------------------------------------------------------------#
def remove_particle_fast(p):
    if p < g_t.ind_num_part[0]:
        c = g_t.pos_tmp_fast[p, 0]

        particles_this_class = np.where(g_t.pos_tmp_fast[:, 0] == c)
        particles_this_class_now = np.where(particles_this_class[0] == g_t.ind_lasts[c])

        if particles_this_class_now[0][0] > 0:
            entered = False
            for i in range(particles_this_class_now[0][0]-1, -1, -1):
                if g_t.pos_tmp_fast[particles_this_class[0][i], 1] > 0:
                    g_t.ind_lasts[g_t.pos_tmp_fast[p, 0]] = particles_this_class[0][i]
                    entered = True
                    break
            if entered == False:
                g_t.ind_lasts[g_t.pos_tmp_fast[p, 0]] = -1
        else:
            g_t.ind_lasts[g_t.pos_tmp_fast[p, 0]] = -1

        g_t.ind_lasts[g_t.pos_tmp_fast[p, 0]] = index
        #in numpy the selection ends 1 before the end. eg_t. [A:B] is from A to B-1 !!!
        g_t.pos_tmp_fast[p:g_t.ind_num_part[0]-1, :] = g_t.pos_tmp_fast[p+1:g_t.ind_num_part[0], :]
        g_t.ind_num_part[0] -= 1
        g_t.ind_num_part[c+1] -= 1
