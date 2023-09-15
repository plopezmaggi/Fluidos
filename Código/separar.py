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
     
def pre_process(dir: str, file_name: str, start_at: int, stop_at: int, folder="", method="KNN", filter_color=False, circulo=None, rectangulo=None):
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
        
    circulo
        Para recortar con forma de círculo, poner pasarle una tupla ((centro_x, centro_y), radio)
    
    rectangulo
        Para recortar con forma de rectángulo, pasarle una tupla ((esquina_superior_izquierda_x, esquina_superior_izquierda_y), lado_horizontal, lado_vertical)
    """
    
    ### Choosing method selected for the background substractor object
    
    # Esta parte del código elimina el fondo de las imágenes. Hay 2 métodos, MOG y KNN
    #KNN es un poco más rápido computacionalmente pero si el fondo es poco estable es preferible elegir MOG2. Por default si no ponemos nada va KNN. 
    if method == "MOG2":
        backSub = cv.createBackgroundSubtractorMOG2(detectShadows=False)
    elif method == "KNN":
        backSub = cv.createBackgroundSubtractorKNN(detectShadows=False)
    else:
        raise(Exception("Invalid method name"))
    
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(dir+"/"+file_name)) #Abre el video

    i = 0 ### Initialazing iterable variable // Cada i es un fotograma
     
    Masks = [] ### Initialazing mask list for saving each mask (i.e. Image matrix comprised of 0s and 255s, black and white.)
    # Acá abrió una lista vacía Masks, donde va a ir guardando las máscaras resultantes después de haberles sacado el fondo
    
    if not capture.isOpened():
        raise(Exception("Unable to open file"))
    while True: #Entra al While True, empieza a procesar los fotogramas uno por uno
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
    
            # threshold grayscale image to extract glare - Lo pasa a escala de grises, extrae el resplandor
            mask = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)[1]
            res2 = cv.inpaint(res1, mask, 21, cv.INPAINT_TELEA) 
    
            # Genera el nombre del archivo para la imagen procesada
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
                lower_color = np.array([0, 0, 0])
                upper_color = np.array([180, 255, 255])
     
                ### create a mask for color colour using inRange function
                mask = cv.inRange(hsv, lower_color, upper_color)
    
                ### perform bitwise and on the original image arrays using the mask
                res3 = cv.bitwise_and(res2, res2, mask=mask)
                
                ### display original frame, and filtered black and white frame. 
                #cv.imshow('Frame', frame)
                #cv.imshow('FG Mask', mask)
                
                mask = np.zeros_like(res3)
                
                cropped_image = cv.bitwise_and(res3, mask)
                
                # Recorte con círculo
                if circulo is not None:
                    centro, radius = circulo
                    center_x, center_y = centro
                    
                    # Recorte circular
                    center_x = 932  # coordenada x del centro en px
                    center_y = 526  # coordenada y del centro en px
                    radius = 393   # radio
                    x1 = center_x - radius
                    y1 = center_y - radius
                    x2 = center_x + radius
                    y2 = center_y + radius
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    
                    cv.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=-1)
                    cropped_image = cv.bitwise_and(res2, mask)
                    cropped_image = cropped_image[y1:y2, x1:x2]
                
                # Recorte con rectángulo
                elif rectangulo is not None:
                    esquina, horizontal, vertical = rectangulo
                    x, y = esquina
                    
                    x1, x2 = x, x + horizontal
                    y1, y2 = y - vertical, y
                    
                    cv.rectangle(mask, esquina, (x2, y2), (255, 255, 255), thickness=-1)
                    cropped_image = cv.bitwise_and(res2, mask)
                    cropped_image = cropped_image[y1:y2, x1:x2]
                    
                ### Save file
                cv.imwrite(dir+"/"+folder+"/"+imagen_numero,cropped_image)
                    
    
            else:
                ### display original frame, and filtered frame. 
                # Acá me muestra la imagen original y la procesada
                #cv.imshow('Frame', frame)
                #cv.imshow('FG Mask', res2)
                
                
                mask = np.zeros_like(res2)
                
                cropped_image = cv.bitwise_and(res2, mask)
                
                
                # Recorte con círculo
                if circulo is not None:
                    centro, radius = circulo
                    center_x, center_y = centro
                    
                    # Recorte circular
                    center_x = 541  # coordenada x del centro en px
                    center_y = 971  # coordenada y del centro en px
                    radius = 440   # radio
                    x1 = center_x - radius
                    y1 = center_y - radius
                    x2 = center_x + radius
                    y2 = center_y + radius
                    x1 = max(x1, 0)
                    y1 = max(y1, 0)
                    cv.circle(mask, (center_x, center_y), radius, (255, 255, 255), thickness=-1)
                    cropped_image = cv.bitwise_and(res2, mask)
                    cropped_image = cropped_image[y1:y2, x1:x2]
        
                
                # Recorte con rectángulo
                elif rectangulo is not None:
                    esquina, horizontal, vertical = rectangulo
                    x, y = esquina
                    
                    x1, x2 = x, x + horizontal
                    y1, y2 = y, y + vertical

                    
                    cv.rectangle(mask, esquina, (x2, y2), (255, 255, 255), thickness=-1)
                    cropped_image = cv.bitwise_and(res2, mask)
                    cropped_image = cropped_image[y1:y2, x1:x2]
                    
                ### Save file
                cv.imwrite(dir+"/"+folder+"/"+imagen_numero,cropped_image)
                
            
            Masks.append(mask) ### Add mask to mask list
    
            ### Display progress
            if i%20 == 0:
                print(i, end="\r")
            
            keyboard = cv.waitKey(30)
        
        i += 1
        
            
        if i == stop_at:
            break
    
    ### Close all windows

    capture.release()
    # video.release()
    cv.destroyAllWindows()
    cv.waitKey(1)
    return Masks

#%%

