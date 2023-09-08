#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 08:37:22 2023

@author: plopezmaggi
"""

from __future__ import print_function
import cv2 as cv
import numpy as np

#%%

# Partir en frames
def video2images(video_path, images_path):
    """
    video_path : ruta al video
    images_path : ruta para guardar las imágenes
    """
    
    # Abrir dirección del video
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(video_path))
    
    
    # Leer cada cuadro del video y guardarlo en la carpeta images_path
    i = 0

    if not capture.isOpened():
        raise(Exception("Unable to open file"))
    while True:
        ret, frame = capture.read() ### Reads each frame of the video
        if frame is None:
            break
        if i < 10:
            imagen_numero = f"000{i}.jpg"
        elif 100 > i >= 10:
            imagen_numero = f"00{i}.jpg"
        elif 1000 > i >= 100:
            imagen_numero = f"0{i}.jpg"
        else:
            imagen_numero = f"{i}.jpg"
        cv.imwrite(images_path+imagen_numero,frame)
        i += 1

#%%
""" Esto hace lo mismo (partir en frames) pero además procesa las imágenes para mejorar el contraste """
     
def pre_process(dir: str, file_name: str, start_at: int, stop_at: int, folder="", method="KNN", filter_color=False):
    """ Process each individual frame of a video, generating a high contrast set of frames.
    It removes the background and light reflections on the surface of the fluid, with an optional
    feature of removing the color.
    It's configured to filter all non-blue color by default. To remove another, modify the lower_color and upper_color variables.

    Attributes
    ----------
    dir, file_name : str
        directory and file-name for the video you'd like to process.
    stop_at : int
        Frame number at wich to stop.
    folder (optional) : str
        name of the folder in wich the images will be saved, on the same location as the video.
    method : str
        method used for the background subtraction. Either MOG2 of KNN. Default: KNN
    filter_color : boolian
        whether to apply a color filter on top of the background subtraction. Default: False.
        NOTE: In order for the color filter to work, you have to adjust the color parameters (lower_color, upper_color).
    """

    ### Choosing method selected for the background substractor object
    if method == "MOG2":
        backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    if method == "KNN":
        backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
    else:
        raise(Exception("Invalid method name"))

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(dir+"/"+file_name))

    i = 0 ### Initialazing iterable variable

    Masks = [] ### Initialazing mask list for saving each mask (i.e. Image matrix comprised of 0s and 255s, black and white.)
    if not capture.isOpened():
        raise(Exception("Unable to open file"))
    while True:
        ret, frame = capture.read() ### Reads each frame of the video
        if frame is None:
            break
        
        if i >= start_at:
            fgMask = backSub.apply(frame) ### Substract background
    
            cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    
            res1 = cv.bitwise_and(frame, frame, mask=fgMask)
    
            gray = cv.cvtColor(res1, cv.COLOR_BGR2GRAY)
    
            # threshold grayscale image to extract glare
            mask = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)[1]
            res2 = cv.inpaint(res1, mask, 21, cv.INPAINT_TELEA)
    
            if i < 10:
                imagen_numero = f"000{i}.jpg"
            elif 100 > i >= 10:
                imagen_numero = f"00{i}.jpg"
            elif 1000 > i >= 100:
                imagen_numero = f"0{i}.jpg"
            else:
                imagen_numero = f"{i}.jpg"
    
            if filter_color:
    
                ### convert the BGR image to HSV colour space
                hsv = cv.cvtColor(res2, cv.COLOR_BGR2HSV)
    
                ### set the lower and upper bounds for the color hue
                lower_color = np.array([60, 35, 140])
                upper_color = np.array([180, 255, 255])
    
                ### create a mask for color colour using inRange function
                mask = cv.inRange(hsv, lower_color, upper_color)
    
                ### perform bitwise and on the original image arrays using the mask
                res3 = cv.bitwise_and(res2, res2, mask=mask)
    
                ### display original frame, and filtered black and white frame.
                cv.imshow('Frame', frame)
                cv.imshow('FG Mask', mask)
    
                ### Save file
                cv.imwrite(dir+folder+"/"+imagen_numero,res3)
                i += 1
    
            else:
                ### display original frame, and filtered frame.
                cv.imshow('Frame', frame)
                cv.imshow('FG Mask', res2)
    
                ### Save file
                cv.imwrite(dir+"/"+folder+"/"+imagen_numero,res2)
                i += 1
    
            Masks.append(mask) ### Add mask to mask list
    
            ### Display progress
            if i%20 == 0:
                print(i, end="\r")
    
            keyboard = cv.waitKey(30)
        if i == stop_at:
            break

    ### Close all windows

    capture.release()
    # video.release()
    cv.destroyAllWindows()
    cv.waitKey(1)
    return Masks

#%%

