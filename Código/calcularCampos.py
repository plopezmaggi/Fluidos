#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:15:44 2023

@author: plopezmaggi
"""

from openpiv import tools, pyprocess, validation, filters, scaling

from tqdm import tqdm
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

def remove_velocity_outliers(u, v):
    """
    Cambiar por NaN las velocidades alejadas en módulo de la media.
    
    u, v : arrays 2d con las componentes del campo de velocidades
    """
    # Módulo cuadrado
    Velocity = np.sqrt(u**2 + v**2)
    
    # Media y std de la velocidad (ignora las que no están definidas)
    mean_velocity = np.mean(Velocity[~np.isnan(Velocity)])
    sigma = np.std(Velocity[~np.isnan(Velocity)])
    
    # Mask para velocidades alejadas
    Velocity_filtered = np.where(Velocity > mean_velocity + 3*sigma)
    
    # Cambiar velocidades alejadas por NaN
    u[Velocity_filtered], v[Velocity_filtered] = np.nan, np.nan
    
    return u, v

def get_velocity_field(start, stop, path, fps, pixel2cm, step=1, winsize=32, 
                      searchsize=32, overlap=16, threshold=2, replace_outliers=False):
    '''
    Attributes
    ----------
    start // stop // step: int
        defines the frame in which to start // end the field calculation,
        and the step between each frame (defaults to 1).
    fps : int
        frames per second of the video from wich the images were extracted.
    pixel2cm : int // float
        scaling factor in micron/pixel
    winsize : int
        pixels, interrogation window size in frame A
    searchsize : int
        pixels, search in image B. Cannot be smaller than winsize!
    threshold : int
        threshold for the sig2noice validation. The validation then eliminates the vector for wich the sig2noice value surpasses the threshold.
    overlap : int:
        pixels, overlap between two windows (defines point density)
    
    '''
    
    U, V = [], []

    # Loopear sobre todos los frames para compararlos de a pares
    for frame_idx in tqdm(range(start, stop, step)):
        # Definir nombre del primer frame (según cómo se lo puso cuando partió el video)
        if frame_idx < 10:
            imagen_numero_1 = f"000{frame_idx}.jpg"
        elif 100 > frame_idx >= 10:
            imagen_numero_1 = f"00{frame_idx}.jpg"
        elif 1000 > frame_idx >= 100:
            imagen_numero_1 = f"0{frame_idx}.jpg"
        else:
            imagen_numero_1 = f"{frame_idx}.jpg"
        
        # El segundo frame es el siguiente al primero
        frame_idx_2 = frame_idx + 1
        
        # Definir nombre del primer frame (según cómo se lo puso cuando partió el video)
        if frame_idx_2 < 10:
            imagen_numero_2 = f"000{frame_idx_2}.jpg"
        elif 100 > frame_idx_2 >= 10:
            imagen_numero_2 = f"00{frame_idx_2}.jpg"
        elif 1000 > frame_idx_2 >= 100:
            imagen_numero_2 = f"0{frame_idx_2}.jpg"
        else:
            imagen_numero_2 = f"{frame_idx_2}.jpg"
        
        print(path+imagen_numero_1)
        
        # Abrir los dos frames (y printear un mensaje de error si no puede)
        try:
            frame_a  = tools.imread(path+imagen_numero_1)
            frame_b  = tools.imread(path+imagen_numero_2)
        except:
            raise(Exception("frame index out of range; Check: \n that the stop argument is lower than the total number of frames or \n \t that the name format for the image is the right one."))

        dt = 1 / fps # sec, time interval between frames
        
        # Comparar los dos frames para calcular el campo de velocidades
        u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                            frame_b.astype(np.int32), 
                                                            window_size=winsize, 
                                                            overlap=overlap, 
                                                            dt=dt, 
                                                            search_area_size=searchsize, 
                                                            sig2noise_method='peak2peak')

        # Coordenadas de las ventanas
        x, y = pyprocess.get_coordinates(image_size=frame_a.shape, 
                                        search_area_size=searchsize, 
                                        overlap=overlap )
        
        # Reemplaza por NaN a los vectores con demasiado ruido
        u1, v1 = u0.copy(), v0.copy()

        u1, v1, mask = validation.sig2noise_val(u1, v1, 
                                                sig2noise, 
                                                threshold = threshold)
        
        # Reemplaza por NaN a los vectores muy alejados en módulo de la media
        u1, v1 = remove_velocity_outliers(u1, v1)

        # filter out outliers that are very different from the neighbours
        if replace_outliers:
            u2, v2 = filters.replace_outliers( u1, v1, 
                                            method='localmean', 
                                            max_iter=3, 
                                            kernel_size=3)
        
        # Si no, nos quedamos con todos
        else:
            u2, v2 = u1.copy(), v1.copy()



        # convert x,y to cm
        # convert u,v to cm/sec
        x3, y3, u3, v3 = scaling.uniform(x, y, u2, v2, 
                                    scaling_factor=pixel2cm)

        ### 0,0 shall be bottom left, positive rotation rate is counterclockwise
        ### Note: Imread reads each frame inverted on the y-axis (dunno why),
        ### so we have to adjust for that reflection (View plots)
        x3, y3, u3, v3 = tools.transform_coordinates(x3, y3, u3, v3)

        U.append(u3)
        V.append(v3)
        
    return x3, y3, U, V

def mean_velocity_field(F):
    """
    Promediar campo de velocidades entre todos los frames.
    """
    F_total = np.zeros_like(F[0])
    F_total_err = np.zeros_like(F[0])
    for i in tqdm(range(F[0].shape[0])):
        for j in range(F[0].shape[1]):
            counter = 0
            F_list = []
            for k in range(len(F)):
                if ~np.isnan(F[k][i, j]):
                    F_total[i, j] += F[k][i, j]
                    F_list.append(F[k][i, j])
                    counter += 1
            if counter != 0:
                F_total[i, j] = F_total[i, j]/counter
                F_total_err[i, j] = np.std(F_list)/np.sqrt(counter)
    return F_total, F_total_err


path_images = "cuadros"

frames = [int(nombre[:-4]) for nombre in os.listdir(path_images)]
start = min(frames)
stop = max(frames)

video = cv2.VideoCapture('video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)

diametro_px = 882 # <----- CAMBIEN ESTO!!!!!!
diametro_cm = 12.7

pixel2cm = diametro_px / diametro_cm ### scale in pixel/cm

ws = 32
ss = 32
ol = 20
threshold = 1.2 ### After a few iterations with different thresholds, this is the one we landed on.

x, y, U, V = get_velocity_field(start=start, stop=stop, path=path_images+"/", fps=fps,
                                pixel2cm=pixel2cm, winsize=ws, searchsize=ss,
                                overlap=ol, threshold=threshold, replace_outliers=False)

# Calcular campos de velocidades promediados
U_total, U_total_err = mean_velocity_field(U)
V_total, V_total_err = mean_velocity_field(V)

# Guardar campos promediados como txt
mask = np.zeros(U[0].shape, dtype=bool) ### Define the mask as an all-true matrix, bc we wont take into account the error in calculating the velocities for each frame.
tools.save(x, y, U_total, V_total, mask, 'pos+promedio.txt')


x = x.reshape(x.shape[0] * x.shape[1])
y = y.reshape(y.shape[0] * y.shape[1])
U_total = U_total.reshape(U_total.shape[0] * U_total.shape[1])
V_total = V_total.reshape(V_total.shape[0] * V_total.shape[1])
U_total_err = U_total_err.reshape(U_total_err.shape[0] * U_total_err.shape[1])
V_total_err = V_total_err.reshape(V_total_err.shape[0] * V_total_err.shape[1])


# Guardar campos promediados como archivo comprimido de numpy
np.savez_compressed('promedio-vel', U=U_total, V=V_total, Uerr=U_total_err, Verr=V_total_err)

# Guardar posiciones como archivo comprimido de numpy
np.savez_compressed('posiciones', x=x, y=y)

# Guardar campos por frame como archivo comprimido de numpy
np.savez_compressed('velocidades', Ulist=U, Vlist=V)