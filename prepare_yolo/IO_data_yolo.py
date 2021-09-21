# -*- coding: utf-8 -*-

# Scolymus 2021.
# https://github.com/Scolymus/Yolo_Tracking
# License CC BY-NC-SA 4.0

import cv2
import numpy as np
import shutil
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
import random
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
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"prepare_yolo")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import settings_yolo as g_l

#----------------------------------------------------------------------------#
#                                                                            #
#                              YOLOv3_to_centers                             #
#                                                                            #                                              
#   Loads the txt file in YOLOv3 format and stores it as the position of the #
#   particle.                                                                #
#                                                                            #
#   @Inputs:                                                                 #
#     filename (string). File to load                                        #
#     rows (int). Number of rows of the image to load                        #
#     cols (int). Number of cols of the image to load                        #
#                                                                            #
#----------------------------------------------------------------------------#
def YOLOv3_to_centers(filename,rows,cols):
    # Load file
    f = open(filename,"r")
    if f.mode == 'r':	
        f1=f.readlines()

    # Init variables
    # First dimension is class. Second is [x,y,width,height]
    g_l.particles = [[] for j in range(5)]

    # for each line
    for x in f1:
        chunks = x.split(" ")	# Split every space
        # line is: class_num x/cols y/rows width/cols height/rows
        # a line can have more than 1 particle, and hence the last 4 elements can
        # appear more than once! (class_num only appears one per line)
        for i in range(len(chunks[1::4])):
            g_l.particles[0].append(int(x[0]))
            g_l.particles[1].append(float(chunks[1::4][i])*cols)
            g_l.particles[2].append(float(chunks[2::4][i])*rows)
            g_l.particles[3].append(float(chunks[3::4][i])*cols)
            g_l.particles[4].append(float(chunks[4::4][i])*rows)

    f.close()


#----------------------------------------------------------------------------#
#                                                                            #
#                              Particle to image                             #
#                                                                            #                                              
#   Splices the image at filename adress using the particles coordinates     #
#   previously loaded. This function is the old "sub_frame_particles2"       #
#   modified appropiately for non uniform Radius                             #
#   @Inputs:                                                                 #
#     filename (string). File to load                                        #
#     num_trials (int). For each particle, we create an "empty image" (an    #
#           image with the same size as the particle but without any) to     #
#           improve the dataset. The amount of random images we try until we #
#           obtain one without particle is this parameter.                   #
#     isgray (boolean). True: B&W image. False: Colour image                 #
#----------------------------------------------------------------------------#
def Particle2image(filename, num_trials, isgray):
    corners = [[0,0],[0,0],[0,0],[0,0]]
    images = []
    height_image = g_l.rows
    width_image = g_l.cols

    # Create folder where to save the new dataset
    folder = filename[:filename.rfind(os.path.sep)+1]+"cut_images"+os.path.sep
    filee = filename[filename.rfind(os.path.sep)+1:]

    # First we will create polygons using the bounding boxes of our particles
    truth_bboxes = []
    for p in range(len(g_l.particles[0])):
        corners[0] = [int(g_l.particles[1][p]-g_l.particles[3][p]/2), int(g_l.particles[2][p]-g_l.particles[4][p]/2)]
        corners[1] = [int(g_l.particles[1][p]-g_l.particles[3][p]/2), int(g_l.particles[2][p]+g_l.particles[4][p]/2)]
        corners[2] = [int(g_l.particles[1][p]+g_l.particles[3][p]/2), int(g_l.particles[2][p]+g_l.particles[4][p]/2)]
        corners[3] = [int(g_l.particles[1][p]+g_l.particles[3][p]/2), int(g_l.particles[2][p]-g_l.particles[4][p]/2)]
                
        truth_bboxes.append(Polygon(corners)) 

    # Second, we will draw empty regions and cut images
    # This improves our dataset by including areas with no particles
    for p in range(len(g_l.particles[0])):
        # Create folder where to save empty images
        if not os.path.exists(folder+"images_void"+os.path.sep):
            os.makedirs(folder+"images_void"+os.path.sep)
        if not os.path.exists(folder+"annotations_void"+os.path.sep):
            os.makedirs(folder+"annotations_void"+os.path.sep)
        
        # Build size of new cutted images depending on the size of this particle
        # We use the larger axis for cutting the image, but consider the smaller for the radius
        if g_l.particles[3][p] > g_l.particles[4][p]:
            radius = int(g_l.particles[4][p]/2.)
            length = int(g_l.particles[3][p]*g_l.cut_length/2.)            
        else:
            radius = int(g_l.particles[3][p]/2.)
            length = int(g_l.particles[4][p]*g_l.cut_length/2.)

        # We will look for random regions, and check if no particle is inside. 
        # If we can, then we will print these regions
        #escape = True
        intersects = False    

        # We limit the (x,y). It is like removing a border of D size from the original image
        # Remember that in CV the top border of the image is y=0!
        limx_left = length
        limx_right = width_image-length
        limy_top = length
        limy_down = height_image-length
        
        make_void = True
        if limx_left > limx_right or limx_right < 0 or limx_left > width_image or limy_top > limy_down or limy_down < 0 or limy_top > height_image:
            make_void = False
        #while(escape):
        # For each particle we will try to create a random void area for up to num_trials times
        # If we obtain it, we stop. If we don't, we will not have any for this particle
        if make_void:
            for trials in range(num_trials):	
                # Create a random center
                xr = random.randint(limx_left, limx_right)
                yr = random.randint(limy_top, limy_down)       
             
                # Create a region around this random center
                limx_left_random = xr-length
                limx_right_random = xr+length
                limy_top_random = yr-length
                limy_down_random = yr+length

                if limx_left_random < 0:
                    limx_left_random = 0
                elif limx_left_random > width_image:
                    intersects = False   
                    break

                if limx_right_random > width_image:
                    limx_right_random = width_image

                if limy_top_random < 0:
                    limy_top_random = 0
                elif limy_top_random > height_image:
                    intersects = False   
                    break

                if limy_down_random > height_image:
                    limy_down_random = height_image			

                corners[0] = [limx_left_random, limy_top_random]
                corners[1] = [limx_left_random, limy_down_random]
                corners[2] = [limx_right_random, limy_down_random]
                corners[3] = [limx_right_random, limy_top_random]   

                if  limx_right_random < limx_left_random or limy_down_random < limy_top_random:
                    intersects = False   
                    break
                
                void_surface = Polygon(corners)
                
                for pp in range(len(g_l.particles[0])):
                    if truth_bboxes[pp].intersects(void_surface) == True:
                        intersects = True
                        break

                if intersects == False:
                    #print("len "+str(g_l.frame.shape)+" xl :"+str(limx_left_random)+" xr :"+str(limx_right_random)+" yt :"+str(limy_top_random)+" yd :"+str(limy_down_random))
                    cv2.imwrite(folder+"images_void"+os.path.sep+filee.replace(".png","_empty_c_"+str(g_l.particles[0][p])+"_p_"+str(p)+".png"),g_l.frame[limy_top_random:limy_down_random,limx_left_random:limx_right_random])                                                                
                    open(folder+"annotations_void"+os.path.sep+filee.replace(".png","_empty_c_"+str(g_l.particles[0][p])+"_p_"+str(p)+".txt"), 'a').close()
                    #escape = False
                    break
                else:
                    intersects = False   
                
        # Now we cut the particle from the whole image
        # limxy_zzz_bp is the border of the new image; xbp,ybp is the center of this image
        # limxy_zzz_p is the bounding box of this particle in this new image
        # However, we shift the center of the image a bit from the center of the particle
        # to avoid having always a particle in the center
        # We will correct the boundings from being outside the image borders

        xbp = g_l.particles[1][p]+random.uniform(-1, 1)*radius
        ybp = g_l.particles[2][p]+random.uniform(-1, 1)*radius

        limx_left_bp = xbp-length
        limx_right_bp = xbp+length
        limy_top_bp = ybp-length
        limy_down_bp = ybp+length
            
        if limx_left_bp < 0:
            limx_right_bp += -limx_left_bp
            limx_left_bp = 0
        if limx_right_bp > width_image:
            limx_left_bp -= limx_right_bp-width_image
            limx_right_bp = width_image
        if limy_top_bp < 0:
            limy_down_bp += -limy_top_bp
            limy_top_bp = 0
        if limy_down_bp > height_image:
            limy_top_bp -= limy_down_bp-height_image
            limy_down_bp = height_image                

        width_new_image = limx_right_bp-limx_left_bp
        height_new_image = limy_down_bp-limy_top_bp

        # Now the particle
        xp = g_l.particles[1][p]
        yp = g_l.particles[2][p]
        wp = int(g_l.particles[3][p]/2.)
        hp = int(g_l.particles[4][p]/2.)

        # Now this is with respect the new image axis
        limx_left_p = xp-wp  - limx_left_bp
        limx_right_p = xp+wp - limx_left_bp
        limy_top_p = yp-hp   - limy_top_bp
        limy_down_p = yp+hp  - limy_top_bp
            
        if limx_left_p < 0:
            limx_left_p = 0
        if limx_right_p > width_new_image:
            limx_right_p = width_new_image
        if limy_top_p < 0:
            limy_top_p = 0
        if limy_down_p > height_new_image:
            limy_down_p = height_new_image    
              
        # Just in case, everything to integer!
        limx_left_bp = int(limx_left_bp)
        limx_right_bp = int(limx_right_bp)
        limy_top_bp = int(limy_top_bp)
        limy_down_bp = int(limy_down_bp)
        limx_left_p = int(limx_left_p)
        limx_right_p = int(limx_right_p)
        limy_top_p = int(limy_top_p)
        limy_down_p = int(limy_down_p)
                
        # We create the bounding box of the particle
        corners[0] = [limx_left_p, limy_top_p]
        corners[1] = [limx_left_p, limy_down_p]
        corners[2] = [limx_right_p, limy_down_p]
        corners[3] = [limx_right_p, limy_top_p]

        # We draw this particle in a temporal image
        temp = g_l.frame.copy()
        temp = temp[limy_top_bp:limy_down_bp, limx_left_bp:limx_right_bp] #in cv, first Y, then X axis
        pts = np.array([corners[0],corners[1],corners[2],corners[3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(temp,[pts],True,(0,0,255))             

        # We also create a polygon using the size of the new image
        corners[0] = [limx_left_bp, limy_top_bp]
        corners[1] = [limx_left_bp, limy_down_bp]
        corners[2] = [limx_right_bp, limy_down_bp]
        corners[3] = [limx_right_bp, limy_top_bp]   
  
        new_image_polygon = Polygon(corners)
        
        # Now, we will also add those particles that are in the new image. Only if their center is inside the image I will count them
        final_particles = [[] for j in range(5)]

        for pp in range(len(g_l.particles[0])):
            # if this particle is inside the new image
            if truth_bboxes[pp].intersects(new_image_polygon) == True:
                # and its center of mass is also inside
                if g_l.particles[1][pp]-limx_left_bp > 0 and g_l.particles[2][pp]-limy_top_bp > 0 and limx_right_bp-g_l.particles[1][pp] > 0 and limy_down_bp-g_l.particles[2][pp] > 0:
                    # take borders of the image: (x,y)+-(w,h)-new_origin
                    limx_left_pp = g_l.particles[1][pp]-int(g_l.particles[3][pp]/2.)  - limx_left_bp
                    limx_right_pp = g_l.particles[1][pp]+int(g_l.particles[3][pp]/2.) - limx_left_bp
                    limy_top_pp = g_l.particles[2][pp]-int(g_l.particles[4][pp]/2.)   - limy_top_bp
                    limy_down_pp = g_l.particles[2][pp]+int(g_l.particles[4][pp]/2.)  - limy_top_bp
            
                    if limx_left_pp < 0:
                        limx_left_pp = 0
                    if limx_right_pp > width_new_image:
                        limx_right_pp = width_new_image
                    if limy_top_pp < 0:
                        limy_top_pp = 0
                    if limy_down_pp > height_new_image:
                        limy_down_pp = height_new_image  
 
                    limx_left_pp = int(limx_left_pp)
                    limx_right_pp = int(limx_right_pp)
                    limy_top_pp = int(limy_top_pp)
                    limy_down_pp = int(limy_down_pp)

                    corners[0] = [limx_left_pp, limy_top_pp]
                    corners[1] = [limx_left_pp, limy_down_pp]
                    corners[2] = [limx_right_pp, limy_down_pp]
                    corners[3] = [limx_right_pp, limy_top_pp]    

                    # Paint particle in frame
                    pts = np.array([corners[0],corners[1],corners[2],corners[3]], np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(temp,[pts],True,(255,0,0))    

                    # Save particle
                    final_particles[0].append(g_l.particles[0][pp])
                    final_particles[1].append(g_l.particles[1][pp]-limx_left_bp)
                    final_particles[2].append(g_l.particles[2][pp]-limy_top_bp)
                    final_particles[3].append(limx_right_pp-limx_left_pp)
                    final_particles[4].append(limy_down_pp-limy_top_pp)

        # We store this image to return it with this function
        images.append(temp)

        # Create folders if necessary
        if not os.path.exists(folder+"images_show"+os.path.sep):
            os.makedirs(folder+"images_show"+os.path.sep)
        if not os.path.exists(folder+"images"+os.path.sep):
            os.makedirs(folder+"images"+os.path.sep)
        if not os.path.exists(folder+"annotations"+os.path.sep):
            os.makedirs(folder+"annotations"+os.path.sep)

        # Saves images
        cv2.imwrite(folder+"images_show"+os.path.sep+filee.replace(".png","_c_"+str(g_l.particles[0][p])+"_p_"+str(p)+"_show.png"),temp)
        if isgray:
            cv2.imwrite(folder+"images"+os.path.sep+filee.replace(".png","_c_"+str(g_l.particles[0][p])+"_p_"+str(p)+".png"),cv2.cvtColor(g_l.frame[limy_top_bp:limy_down_bp, limx_left_bp:limx_right_bp], cv2.COLOR_BGR2GRAY))
        else:
            cv2.imwrite(folder+"images"+os.path.sep+filee.replace(".png","_c_"+str(g_l.particles[0][p])+"_p_"+str(p)+".png"),g_l.frame[limy_top_bp:limy_down_bp, limx_left_bp:limx_right_bp])

        # Save annotations
        res = particles2YOLO(images[len(images)-1].shape[1], images[len(images)-1].shape[0], isgray, folder+"annotations"+os.path.sep+filee.replace(".png","_c_"+str(g_l.particles[0][p])+"_p_"+str(p)+".txt"), final_particles)
				
    return images

#----------------------------------------------------------------------------#
#                                                                            #
#                              Particle to image                             #
#                                                                            #                                              
#   Saves the input as a YOLOv3 annotation file                              #
#   This function is the old "save_images_YOLOv3_cut_images2"                #
#   @Inputs:                                                                 #
#     cols (int). Columns in the image                                       #
#     rows (int). Rows in the image                                          #
#     isgray (boolean). True: B&W. False: Colour                             #
#     filename (string). File to load                                        #
#     particles (as g_l.particles format). Particles list to save            #
#     new_line (boolean), false by default. If true, each particle is printed#
#                in a new line                                               #
#----------------------------------------------------------------------------#
def particles2YOLO(cols, rows, isgray, filename, particles, new_line=False):  

    particles = np.array(particles).astype(int)
    first = True
    total_text = ""

    # For each of the classes we have
    for c in np.unique(particles[0]):
        if new_line == False:
            if first:
                total_text = str(c)
                first = False
            else:
                total_text = total_text + "\n" + str(c)

        mask = particles[0, :] == c
        p_c = particles[:,mask]

        for p in range(len(p_c[0])):
            x = p_c[1][p]
            y = p_c[2][p]
            w = p_c[3][p]
            h = p_c[4][p]
            if new_line == False:
                total_text = total_text + " "+str('{:.15f}'.format(x/cols)) + " "+str('{:.15f}'.format(y/rows)) + " "+str('{:.15f}'.format(w/cols)) + " "+str('{:.15f}'.format(h/rows))
            else:
                total_text = total_text + str(c) + " "+str('{:.15f}'.format(x/cols)) + " "+str('{:.15f}'.format(y/rows)) + " "+str('{:.15f}'.format(w/cols)) + " "+str('{:.15f}'.format(h/rows))+"\n"

    if new_line:
        total_text = rstrip(total_text)

    file1 = open(filename,"w")
    file1.write(total_text)
    file1.close()

#----------------------------------------------------------------------------#
#                                                                            #
#                             Split data set                                 #
#                                                                            #                                              
#   Splits x% of images at image_dir as training, and 100-x% as test         #
#   Needs the dir of the dataset and its name                                #
#----------------------------------------------------------------------------#
def split_data_set(yolo_dir, dataset_name, x, image_dir):

    f_val = open(yolo_dir+str(dataset_name)+"_test.txt", 'w')
    f_train = open(yolo_dir+os.path.sep+str(dataset_name)+"_train.txt", 'w')

    print(image_dir)
    
    path, dirs, files = next(os.walk(image_dir))
    data_size = len(files)

    ind = 0
    data_test_size = int(x * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)
    
    for f in os.listdir(image_dir):
        if(f.split(".")[1] == "png"):
            ind += 1
            
            if ind in test_array:
                f_val.write(image_dir+f+'\n')
            else:
                f_train.write(image_dir+f+'\n')

    return yolo_dir+str(dataset_name)+"_train.txt", yolo_dir+str(dataset_name)+"_test.txt"

#----------------------------------------------------------------------------#
#                                                                            #
#                                  Copytree                                  #
#                                                                            #                                              
#   Copy all files from src folder to dst folder                             #
#----------------------------------------------------------------------------#
#https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
