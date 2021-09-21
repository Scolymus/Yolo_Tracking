# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 2019

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

"""

import cmapy
import csv
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

import Image_API_t as iml_t
import math
import numpy as np
import os
import settings_t as g_t
import copy

unit_distance = "px"
unit_time = "frames"

def save_data(filename):
    # We start saving the video processed
    if not os.path.exists(filename[:filename.rfind(os.path.sep)+1]+"csv"+os.path.sep):
        os.makedirs(filename[:filename.rfind(os.path.sep)+1]+"csv"+os.path.sep)

    print("We are writing data for video "+filename)
    save_general_info(filename)

    for c in range(g_t.classes_to_recognize):
        filename_csv = filename[:filename.rfind(os.path.sep)+1]+"csv"+os.path.sep+filename[filename.rfind(os.path.sep)+1:-4]+"_c_"+str(c)+".csv"
        print("We are writing class "+str(c+1)+" out of "+str(g_t.classes_to_recognize))
        print("In file "+filename_csv)
        with open(filename_csv,'w') as csvfile:	#in python2 is wb mode!
            writer = csv.writer(csvfile, delimiter=';')
            for p in range(g_t.ind_num_part[0]):
                if g_t.pos[p, 0] == c:
                    writer.writerow(['Particle', str(int(p))])
                    writer.writerow(['X ('+unit_distance+')']+list( g_t.pos_time[p][0][g_t.pos_time[p][0] > 0] ))
                    writer.writerow(['Y ('+unit_distance+')']+list( g_t.pos_time[p][1][g_t.pos_time[p][0] > 0] ))
                    writer.writerow(['W/2 ('+unit_distance+')']+list( g_t.pos_time[p][2][g_t.pos_time[p][0] > 0] ))
                    writer.writerow(['H/2 ('+unit_distance+')']+list( g_t.pos_time[p][3][g_t.pos_time[p][0] > 0] ))
                    writer.writerow(['t ('+unit_time+')']+list( np.where( g_t.pos_time[p][0] > 0 )[0] ))
                    writer.writerow('')
    print("File written")

def save_data_correct_disaster(filename):
    # We start saving the video processed
    if not os.path.exists(filename[:filename.rfind(os.path.sep)+1]+"csv_corrected"+os.path.sep):
        os.makedirs(filename[:filename.rfind(os.path.sep)+1]+"csv_corrected"+os.path.sep)

    g_t.at_frame = g_t.num_frames

    print("We are writing data for video "+filename)
    save_general_info(filename)

    for c in range(g_t.classes_to_recognize):
        filename_csv = filename[:filename.rfind(os.path.sep)+1]+"csv_corrected"+os.path.sep+filename[filename.rfind(os.path.sep)+1:-4]+"_c_"+str(c)+".csv"
        print("We are writing class "+str(c+1)+" out of "+str(g_t.classes_to_recognize))
        print("In file "+filename_csv)
        with open(filename_csv,'w') as csvfile:	#in python2 is wb mode!
            writer = csv.writer(csvfile, delimiter=';')
            for p in range(len(g_t.pos_time[c])):
                writer.writerow(['Particle', str(int(p))])
                writer.writerow(['X ('+unit_distance+')']+list(g_t.pos_time[c][p][1]))
                writer.writerow(['Y ('+unit_distance+')']+list(g_t.pos_time[c][p][2]))
                writer.writerow(['W/2 ('+unit_distance+')']+list( g_t.pos_time[p][2][g_t.pos_time[p][0] > 0] ))
                writer.writerow(['H/2 ('+unit_distance+')']+list( g_t.pos_time[p][3][g_t.pos_time[p][0] > 0] ))
                writer.writerow(['t ('+unit_time+')']+list(g_t.pos_time[c][p][0]))
                writer.writerow('')
    print("File written")

def save_general_info(filename):
    filename_csv = filename[:filename.rfind(os.path.sep)+1]+"csv"+os.path.sep+filename[filename.rfind(os.path.sep)+1:-4]+"_c_general.csv"
    with open(filename_csv,'w') as csvfile:	#in python2 is wb mode!
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Classes', str(g_t.classes_to_recognize)])
        writer.writerow(['Frames (starting index==0)', str(g_t.at_frame)])
        for c in range(g_t.classes_to_recognize):
            writer.writerow(['Num particles in class', str(c), str(g_t.ind_num_part[c+1])])

def open_data(filename, use_original_video=False, use_rotated_data=False):
    print("We are reading data for video "+filename)
    open_general_info(filename, use_original_video, use_rotated_data)

    for c in range(g_t.classes_to_recognize):
        if use_original_video == False:
            filename_csv = filename[:filename.rfind(os.path.sep)]   #/tracked
        else:
            filename_csv = filename

        if use_rotated_data == False:
            filename_csv = filename_csv[:filename_csv.rfind(os.path.sep)+1]+"csv"+os.path.sep   #/csv/
        else:
            filename_csv = filename_csv[:filename_csv.rfind(os.path.sep)+1]+"csv_rotated_clean"+os.path.sep   #/csv/

        filename_csv = filename_csv+filename[filename.rfind(os.path.sep)+1:-4]+"_c_"+str(c)+".csv"

        print("We are reading class "+str(c+1)+" out of "+str(g_t.classes_to_recognize))
        print("In file "+filename_csv)

        with open(filename_csv,'r') as csvfile:	#in python2 is rb mode!
            reader = csv.reader(csvfile, delimiter=';')

            for row in reader:
                if row == []:
                    continue
                elif row[0] == "Particle":
                    p = int(row[1])
                    #print("p is "+str(p-1)+" " +str(len(g_t.pos_time[c][p-1][0])))
                    #if len(g_t.pos_time[c][p-1][0]) > g_t.num_frames+1:
                    #    print("p is "+str(p-1)+" " +str(len(g_t.pos_time[c][p-1][0])))
                elif row[0].startswith("X ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][1].append(int(num))
                elif row[0].startswith("Y ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][2].append(int(num))
                elif row[0].startswith("W/2 ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][3].append(int(num))
                elif row[0].startswith("H/2 ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][4].append(int(num))
                elif row[0].startswith("t ("):
                    positions = row[1:]
                    #print("p is "+str(p)+" " +str(len(positions)))
                    #if p == 162:
                    #    print(positions)
                    for num in positions:
                        g_t.pos_time[c][p][0].append(int(num))

    print("File read")

def open_data_correct_disaster(filename, use_original_video=False, use_rotated_data=False):
    print("We are reading data for video "+filename)
    open_general_info(filename, use_original_video, use_rotated_data)

    for c in range(g_t.classes_to_recognize):
        if use_original_video == False:
            filename_csv = filename[:filename.rfind(os.path.sep)]   #/tracked
        else:
            filename_csv = filename

        if use_rotated_data == False:
            filename_csv = filename_csv[:filename_csv.rfind(os.path.sep)+1]+"csv"+os.path.sep   #/csv/
        else:
            filename_csv = filename_csv[:filename_csv.rfind(os.path.sep)+1]+"csv_rotated_clean"+os.path.sep   #/csv/

        filename_csv = filename_csv+filename[filename.rfind(os.path.sep)+1:-4]+"_c_"+str(c)+".csv"

        print("We are reading class "+str(c+1)+" out of "+str(g_t.classes_to_recognize))
        print("In file "+filename_csv)

        with open(filename_csv,'r') as csvfile:	#in python2 is rb mode!
            reader = csv.reader(csvfile, delimiter=';')

            for row in reader:
                if row == []:
                    continue
                elif row[0] == "Particle":
                    p = int(row[1])
                    #print("p is "+str(p-1)+" " +str(len(g_t.pos_time[c][p-1][0])))
                    #if len(g_t.pos_time[c][p-1][0]) > g_t.num_frames+1:
                    #    print("p is "+str(p-1)+" " +str(len(g_t.pos_time[c][p-1][0])))
                elif row[0].startswith("X ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][1].append(int(num))
                elif row[0].startswith("Y ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][2].append(int(num))
                elif row[0].startswith("W/2 ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][3].append(int(num))
                elif row[0].startswith("H/2 ("):
                    positions = row[1:]
                    for num in positions:
                        g_t.pos_time[c][p][4].append(int(num))
                elif row[0].startswith("t ("):
                    positions = row[1:]
                    #print("p is "+str(p)+" " +str(len(positions)))
                    #if p == 162:
                    #    print(positions)
                    if positions[0].startswith("["):
                        positions = positions[0][1:-1].split(" ")
                        for num in positions:
                            xxx = num.strip()                        
                            if xxx != "":
                                g_t.pos_time[c][p][0].append(int(xxx))
                    else:
                        for num in positions:
                            g_t.pos_time[c][p][0].append(int(num))

    print("File read")

def open_general_info(filename, use_original_video, use_rotated_data):
    if use_original_video == False:
        filename_csv = filename[:filename.rfind(os.path.sep)]   #/tracked
    else:
        filename_csv = filename
    #print(filename)
    if use_rotated_data == False:
        filename_csv = filename_csv[:filename_csv.rfind(os.path.sep)+1]+"csv"+os.path.sep   #/csv/
    else:
        filename_csv = filename_csv[:filename_csv.rfind(os.path.sep)+1]+"csv_rotated_clean"+os.path.sep   #/csv/

    filename_csv = filename_csv+filename[filename.rfind(os.path.sep)+1:-4]+"_c_general.csv"

    with open(filename_csv,'r') as csvfile:	#in python2 is rb mode!
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                if row == []:
                    continue
                elif row[0].startswith("Classes"):
                    g_t.classes_to_recognize = int(row[1])
                    g_t.ind_num_part = np.zeros(g_t.classes_to_recognize+1).astype(int)
                elif row[0].startswith("Frames"):
                    g_t.num_frames = int(row[1])
                elif row[0].startswith("Num particles in class"):
                    g_t.ind_num_part[int(row[1])+1] = int(row[2])
                    g_t.ind_num_part[0] += int(row[2])
                elif row[0].startswith("Angle rotation"):
                    g_t.angle = float(row[1])*180/math.radians(180)
                elif row[0].startswith("Reference point"):
                    g_t.angle_pos[0] = int(row[1])
                    g_t.angle_pos[1] = int(row[2])
                elif row[0].startswith("Extra rotation"):
                    g_t.angle_extra = eval(row[1])
                elif row[0].startswith("Extra flipped"):
                    g_t.flipped_extra = eval(row[1])

    g_t.pos_time = [[[[] for k in range(5)] for j in range(int(g_t.ind_num_part[i+1]))] for i in range(g_t.classes_to_recognize)]

    print("We found "+str(g_t.classes_to_recognize)+" classes")
