# -*- coding: utf-8 -*-
"""
# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0
"""

import cmapy
import csv
from copy import copy, deepcopy
import cv2
import imutils
import math
import numpy as np
import copy
from lxml import etree

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

unit_distance = "px"
unit_time = "frames"

#----------------------------------------------------------------------------#
#                                                                            #
#                                 Save images                                #
#                                                                            #
# This function was introduced in case in the future someone wants to add    #
# different options to save the image, e.g. PascalVOC                        #
#                                                                            #
#----------------------------------------------------------------------------#
def save_images():
    save_images_YOLOv3()

#----------------------------------------------------------------------------#
#                                                                            #
#                        Save images in YOLOv3 format                        #
#                                                                            #
# This function was introduced in case in the future someone wants to add    #
# different options to save the image, e.g. PascalVOC                        #
#                                                                            #
# We will save data and images as it is, and also rotated versions           #
#   - As it is: we will create rectangular regions parallel to the X axis    #
#               with the minimum size to fit the regions we have.            #
#   - Rotated: for each object, we create a new image where we rotate all the#
#              image to put each object parallel to the X axis. If there are #
#              more than 1 object per image, this trick will be bad...       #
#  @Input:                                                                   #
#    new_line (optional, boolean). If true, we will put each particle per    #
#              line. If false, all particles will go in the same line (after #
#              splitting by class                                            #
#                                                                            #
#----------------------------------------------------------------------------#
def save_images_YOLOv3(new_line=False):

    # Create folders, in case they don't exist
    if not os.path.exists(g.path_out+"annotations"+os.path.sep):	
        os.makedirs(g.path_out+"annotations"+os.path.sep)
    if not os.path.exists(g.path_out+"particles"+os.path.sep):
        os.makedirs(g.path_out+"particles"+os.path.sep)
    if not os.path.exists(g.path_out+"particles_with_border"+os.path.sep):
        os.makedirs(g.path_out+"particles_with_border"+os.path.sep)
    if not os.path.exists(g.path_out+"annotations_not_rotated_full"+os.path.sep):	
        os.makedirs(g.path_out+"annotations_not_rotated_full"+os.path.sep)
    if not os.path.exists(g.path_out+"particles_not_rotated_full"+os.path.sep):
        os.makedirs(g.path_out+"particles_not_rotated_full"+os.path.sep)
    if not os.path.exists(g.path_out+"particles_with_border_not_rotated_full"+os.path.sep):
        os.makedirs(g.path_out+"particles_with_border_not_rotated_full"+os.path.sep)

    first_time = True
    corners_memory = []

    total_particles = 0

    # For each kind of particle
    for c in range(g.classes_to_recognize):
        total_particles = total_particles + len(g.particles[c])
        if len(g.particles[c]) > 0:
            # We prepare the text. After the first class we have to put a break line!
            if first_time:
                total_text = str(c)
                total_text_not_rotated = str(c)
                first_time = False
            else:
                total_text = "\n"+str(c)
                total_text_not_rotated = "\n"+str(c)

            corners_yolo = []

            # For each particle
            for p in range(len(g.particles[c])):
                #yolov3 needs a rectangle without tilting, so I will rotate both the image and the rectangle :)
                corners = iml.create_rectangle(c, p, False, -1, -1, -1)
                corners_memory.append(deepcopy(corners))

                # Take the corner with lower X
                index_min_x = np.argmin(corners, axis=0)[0]

                # Take the closest point to the previous one in X
                # Otherwise is pointing up
                corners_with_out_Xmin = np.delete(deepcopy(corners), index_min_x, 0)
                index_min_x2 = np.argmin(corners_with_out_Xmin, axis=0)[0]

                # The angle is always 90 - acos(1/sqrt(1+(dy/dx)**2))
                # Except if there are 2 points with the same y. Then it's 0!
                if len(np.unique(corners[:, 1])) > 3:
                    delta_x = corners_with_out_Xmin[index_min_x2, 0]-corners[index_min_x, 0]
                    delta_y = corners_with_out_Xmin[index_min_x2, 1]-corners[index_min_x, 1]
                    angle = np.pi - np.arccos(1./np.sqrt(1.+(delta_y/delta_x)**2))

                    # But the rectangle can point up or down. If it's down, we need to multiply per -1
                    if delta_y < 0: angle *= -1.

                else:
                    angle = 0

                rotated_corners = iml.Rotate_pos_time(corners, angle)
                # Rotates around center #rotate_bound is positive, but i think rotate needs negative angle
                rotated_image = imutils.rotate_bound(g.frame.copy(), angle*180/np.pi)	

                rotated_corners[:,0] += int((rotated_image.shape[0]-g.frame.shape[0])/2)
                rotated_corners[:,1] += int((rotated_image.shape[1]-g.frame.shape[1])/2)

                width = abs(rotated_corners[rotated_corners.argmax(axis=0)[0], 0]-rotated_corners[rotated_corners.argmin(axis=0)[0], 0])
                height = abs(rotated_corners[rotated_corners.argmax(axis=0)[1], 1]-rotated_corners[rotated_corners.argmin(axis=0)[1], 1])

                x = rotated_corners[rotated_corners.argmin(axis=0)[0], 0] + int(width/2.)
                y = rotated_corners[rotated_corners.argmin(axis=0)[1], 1] + int(height/2.)

                g.rows_rot, g.cols_rot, d = rotated_image.shape

                if new_line:
                    total_text = total_text + "\n" + str(c) + " " + str(x/g.cols_rot) + " "+str(y/g.rows_rot) + " "+str(width/g.cols_rot) + " "+str(height/g.rows_rot)
                else:
                    total_text = total_text + " "+str(x/g.cols_rot) + " "+str(y/g.rows_rot) + " "+str(width/g.cols_rot) + " "+str(height/g.rows_rot)
 

                # Now we do the operations for non-rotating system
                # First take the max and min points of the bounding box
                min_x = corners[corners.argmin(axis=0)[0], 0]
                min_y = corners[corners.argmin(axis=0)[1], 1]
                max_x = corners[corners.argmax(axis=0)[0], 0]
                max_y = corners[corners.argmax(axis=0)[1], 1]

                if min_x < 0:
                    min_x = 0
                if min_y < 0:
                    min_y = 0
                if max_x > g.cols-1:
                    max_x = g.cols-1
                if max_y > g.rows-1:
                    max_y = g.rows-1

                #width = abs(corners[corners.argmax(axis=0)[0], 0]-corners[corners.argmin(axis=0)[0], 0])
                #height = abs(corners[corners.argmax(axis=0)[1], 1]-corners[corners.argmin(axis=0)[1], 1])
                width = abs(max_x-min_x)
                height = abs(max_y-min_y)

                #x = corners[corners.argmin(axis=0)[0], 0] + int(width/2.)
                #y = corners[corners.argmin(axis=0)[1], 1] + int(height/2.)
                x = min_x + int(width/2.)
                y = min_y + int(height/2.)

                corners_yolo.append([[x-int(width/2.),y-int(height/2.)],[x-int(width/2.),y+int(height/2.)],[x+int(width/2.),y+int(height/2.)],[x+int(width/2.),y-int(height/2.)]])

                if new_line:
                    total_text_not_rotated = total_text_not_rotated + "\n" + str(c) + " " + str(x/g.cols) + " "+str(y/g.rows) + " "+str(width/g.cols) + " "+str(height/g.rows)
                else:                         
                    total_text_not_rotated = total_text_not_rotated + " "+str(x/g.cols) + " "+str(y/g.rows) + " "+str(width/g.cols) + " "+str(height/g.rows)

                file1 = open(g.path_out+"annotations"+os.path.sep+g.filename[g.filename.rfind(os.path.sep)+1:g.filename.rfind(".")]+"_f"+str(g.at_frame)+"_c"+str(c)+"_p"+str(p)+".txt","w")
                file1.write(total_text)
                file1.close()

                cv2.imwrite(g.path_out+"particles"+os.path.sep+g.filename[g.filename.rfind(os.path.sep)+1:g.filename.rfind(".")]+"_f"+str(g.at_frame)+"_c"+str(c)+"_p"+str(p)+".png", rotated_image)

                cv2.drawContours(rotated_image,[rotated_corners],0,(255,100,0),2)
                cv2.imwrite(g.path_out+"particles_with_border"+os.path.sep+g.filename[g.filename.rfind(os.path.sep)+1:g.filename.rfind(".")]+"_f"+str(g.at_frame)+"_c"+str(c)+"_p"+str(p)+".png", rotated_image)

    if total_particles > 0:
        file1 = open(g.path_out+"annotations_not_rotated_full"+os.path.sep+g.filename[g.filename.rfind(os.path.sep)+1:g.filename.rfind(".")]+"_f"+str(g.at_frame)+".txt","w")
        file1.write(total_text_not_rotated)
        file1.close()

        cv2.imwrite(g.path_out+"particles_not_rotated_full"+os.path.sep+g.filename[g.filename.rfind(os.path.sep)+1:g.filename.rfind(".")]+"_f"+str(g.at_frame)+".png", g.frame)

        image_copy = g.frame.copy()

        for i in corners_memory:
            cv2.drawContours(image_copy,[i],0,(255,100,0),2)
        for i in np.array(corners_yolo):
            cv2.drawContours(image_copy,[i],0,(100,100,0),2)
        cv2.imwrite(g.path_out+"particles_with_border_not_rotated_full"+os.path.sep+g.filename[g.filename.rfind(os.path.sep)+1:g.filename.rfind(".")]+"_f"+str(g.at_frame)+".png", image_copy)


'''
def save_images_mini():
    for i in range(2):
        if i == 0:
            m = xp
            n = yp
            folder = mypath[:mypath.rfind(os.path.sep)+1]+"yes"
        else:
            m = xn
            n = yn
            folder = mypath[:mypath.rfind(os.path.sep)+1]+"no"
				
        if not os.path.exists(folder):
            os.makedirs(folder)
				
        list = os.listdir(folder) 
        counter = len(list)	
				
        corners = [[0,0],[0,0],[0,0],[0,0]]
        C = int(R)
        for j in range(len(m)):
            xx =  m[j]
            yy = n[j]
            corners[0] = [xx-C, yy-C]
            corners[2] = [xx+C, yy+C]
					
            cv2.imwrite(folder+os.path.sep+str(counter)+".png", frame[corners[0][1]:corners[2][1],corners[0][0]:corners[2][0]])
            counter = counter +1

def save_images_PascalVOC():
    global cols, rows, d,filename,lastf
	
    if len(xp) == 0 and len(xn) == 0:
        return 0
	
    root = etree.Element("annotation")
    c_name = etree.SubElement(root, "filename")
    c_name.text = mypath[:mypath.rfind(os.path.sep)]+"particles"+os.path.sep+filename[filename.rfind(os.path.sep)+1:filename.rfind(".")]+"_f"+str(lastf)+".png"
    c_path = etree.SubElement(root, "path")
    c_path.text = filename

    c_size = etree.SubElement(root, "size")
    c_width = etree.SubElement(c_size, "width")
    c_width.text = str(cols)
    c_height = etree.SubElement(c_size, "height")
    c_height.text = str(rows)
    c_depth = etree.SubElement(c_size, "depth")
    c_depth.text = str(d)
    c_segmented = etree.SubElement(root, "segmented")
    c_segmented.text = "0"
	
    c_object = []

    conta = -1
    for i in range(2):
        if i == 0:
            m = xp
            n = yp
        else:
            m = xn
            n = yn
					
        corners = [[0,0],[0,0],[0,0],[0,0]]
        C = int(R)
        for j in range(len(m)):
            xx =  m[j]
            yy = n[j]
            conta = conta + 1
            corners[0] = [xx-C, yy-C]
            corners[2] = [xx+C, yy+C]
            c_object.append(etree.SubElement(root, "object"))
            if (i==0):
                etree.SubElement(c_object[conta], "name").text = "Particle"			
            else:
                etree.SubElement(c_object[conta], "name").text = "Particle_no"
			
            c_box = etree.SubElement(c_object[conta], "bndbox")
            c_xmin = etree.SubElement(c_box, "xmin")	
            c_xmin.text = str(corners[0][0])
            c_ymin = etree.SubElement(c_box, "ymin")	
            c_ymin.text = str(corners[0][1])
            c_xmax = etree.SubElement(c_box, "xmax")	
            c_xmax.text = str(corners[2][0])
            c_ymax = etree.SubElement(c_box, "ymax")	
            c_ymax.text = str(corners[2][1])
			
    tree = etree.ElementTree(root)
    if not os.path.exists(mypath[:mypath.rfind(os.path.sep)+1]+"annotations"+os.path.sep):	
        os.makedirs(mypath[:mypath.rfind(os.path.sep)+1]+"annotations"+os.path.sep)
    if not os.path.exists(mypath[:mypath.rfind(os.path.sep)+1]+"particles"+os.path.sep):
        os.makedirs(mypath[:mypath.rfind(os.path.sep)+1]+"particles"+os.path.sep)
	
    tree.write(mypath[:mypath.rfind(os.path.sep)+1]+"annotations"+os.path.sep+filename[filename.rfind(os.path.sep)+1:filename.rfind(".")]+"_f"+str(lastf)+".xml")
    cv2.imwrite(mypath[:mypath.rfind(os.path.sep)+1]+"particles"+os.path.sep+filename[filename.rfind(os.path.sep)+1:filename.rfind(".")]+"_f"+str(lastf)+".png", frame)
'''

