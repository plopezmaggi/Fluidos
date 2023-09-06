#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:15:44 2023

@author: plopezmaggi
"""

from openpiv import tools, pyprocess, validation, filters, scaling, smoothn

import imageio
from os import remove
from tqdm import tqdm
import seaborn as sns

import numpy as np
import matplotlib .pyplot as plt

#%%
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

#%%

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
            imagen_numero_1 = f"Cuadros000{frame_idx}.jpg"
        elif 100 > frame_idx >= 10:
            imagen_numero_1 = f"Cuadros00{frame_idx}.jpg"
        elif 1000 > frame_idx >= 100:
            imagen_numero_1 = f"Cuadros0{frame_idx}.jpg"
        else:
            imagen_numero_1 = f"Cuadros{frame_idx}.jpg"
        
        # El segundo frame es el siguiente al primero
        frame_idx_2 = frame_idx + 1
        
        # Definir nombre del primer frame (según cómo se lo puso cuando partió el video)
        if frame_idx_2 < 10:
            imagen_numero_2 = f"Cuadros000{frame_idx_2}.jpg"
        elif 100 > frame_idx_2 >= 10:
            imagen_numero_2 = f"Cuadros00{frame_idx_2}.jpg"
        elif 1000 > frame_idx_2 >= 100:
            imagen_numero_2 = f"Cuadros0{frame_idx_2}.jpg"
        else:
            imagen_numero_2 = f"Cuadros{frame_idx_2}.jpg"
        
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

        ### Graficar cuadros 5, 15 y 25
        if frame_idx in [5, 15, 25]:
            fig, axs = plt.subplots(2, 2,figsize=(20, 20))
            axs[0,0].quiver(x, y, u0, v0, 
                    #   np.sqrt(u0**2 + v0**2), ### show intensity of the field with a colour range
                      lw=.2, angles="xy", scale_units="xy", scale=10)
            axs[0,0].imshow(frame_a, cmap="Greens", origin="lower")
            axs[0,0].set_title("Calculated Velocity Field")
            # plt.show()
            axs[0,1].quiver(x, y, u1, v1, 
                    #   np.sqrt(u1**2 + v1**2), ### show intensity of the field with a colour range
                      lw=.2, angles="xy", scale_units="xy", scale=10)
            axs[0,1].imshow(frame_a, cmap="Greens", origin="lower")
            axs[0,1].set_title("Filtered Velocity Field")
            # plt.show()
            axs[1,0].quiver(x, y, u2, v2, 
                      np.sqrt(u2**2 + v2**2), ### show intensity of the field with a colour range
                      lw=.2, angles="xy", scale_units="xy", scale=10)
            axs[1,0].set_title("Smoothened Velocity Field")
            # plt.show()
            axs[1,1].quiver(x3, y3, u3, v3, 
                      np.sqrt(u3**2 + v3**2), ### show intensity of the field with a colour range
                      lw=.2, angles="xy", scale_units="xy", scale=10)
            axs[1,1].set_title("Adjusted Velocity Field")
            plt.show()
            
            ### if you need more detailed look, first create a histogram of sig2noise
            fig, axs = plt.subplots(1, 2,figsize=(20, 8))
            axs[0].hist([s2n if (s2n != 0.0) and (s2n < 1e2) else np.nan for s2n in sig2noise.flatten()], bins=250)
            axs[0].set_xlim(0, 10)
            # axs[0].set_xticks(np.arange(1, 10, .5), rotation=90)
            axs[0].grid()
            axs[0].set_title("Histogram of the sig2noice of all the velocity vectors \n(the higher sig2noice, the better)")

            s2n_filtered = [s2n if (s2n != 0.) and (s2n < 10) else np.nan for s2n in sig2noise.flatten()]
            s2n_matrix = np.array(s2n_filtered).reshape(u1.shape)
            sns.heatmap(s2n_matrix, linewidth=0.5, ax=axs[1])
            axs[1].imshow(frame_a)
            axs[1].set_title("Heat map for signal to noice ratio \n(the higher sig2noice, the better)")
            plt.show()

        U.append(u3)
        V.append(v3)
    return x3, y3, U, V

#%%

path_images = "Cuadros procesados/" ### format "folder/folder/"
fps = 59

pixel2cm = 961.338 / 14.5 ### scale in pixel/cm
ws = 32
ss = 32
ol = 20
threshold = 1.2 ### After a few iterations with different thresholds, this is the one we landed on.
x, y, U, V = get_velocity_field(start=25, stop=26, path=path_images, fps=fps,
                                pixel2cm=pixel2cm, winsize=ws, searchsize=ss,
                                overlap=ol, threshold=threshold, replace_outliers=False)

#%%

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

#%%
# Calcular campos de velocidades promediados
U_total, U_total_err = mean_velocity_field(U)
V_total, V_total_err = mean_velocity_field(V)

#%%
save_file_name = "prueba"

# Guardar campos promediados como txt
mask = np.zeros(U[0].shape, dtype=bool) ### Define the mask as an all-true matrix, bc we wont take into account the error in calculating the velocities for each frame.
tools.save(x, y, U_total, V_total, mask, path_images+save_file_name+'.txt')

# Guardar campos promediados como archivo comprimido de numpy
np.savez_compressed(path_images+save_file_name, x=x, y=y, U=U_total, V=V_total, Uerr=U_total_err, Verr=V_total_err)

# Guardar campos por frame como archivo comprimido de numpy
np.savez_compressed(path_images+save_file_name+" velocity_lists", Ulist=U, Vlist=V)

#%%

from scipy.interpolate import griddata

### Interpolate Total Field (U_total, V_total), generating a position meshgrid of (n_points x n_points) [x_mesh, y_mesh]
### Generates interpolated vector fields [U_grid, V_grid]

### create matrix of [x, y] points (example: np.array([[0, 10], [1, 10] ... [10, 10]]), [[0, 9], [1, 9] ... [10, 9]] ...)
xy_points = np.concatenate([x.reshape(x.shape[0]*x.shape[1], 1), y.reshape(y.shape[0]*y.shape[1], 1)], axis=1)
U_flattened = U_total.flatten()
V_flattened = V_total.flatten()
n_points = 200
### create high density meshgrid of x,y points
x_mesh, y_mesh = np.meshgrid(np.linspace(np.min(x), np.max(x), n_points), np.linspace(np.min(y), np.max(y), n_points))
U_grid = griddata(xy_points, U_flattened, (x_mesh, y_mesh), method="cubic")
V_grid = griddata(xy_points, V_flattened, (x_mesh, y_mesh), method="cubic")

#%%
fig, axs = plt.subplots(1, 2, figsize=(32, 16), dpi=100)
axs[0].quiver(x_mesh, y_mesh, U_grid, V_grid, np.sqrt(U_grid**2 + V_grid**2), lw=.2)
frame = plt.imread(path_images+"Cuadros0025.jpg") #Ver esta linea!!!
axs[0].imshow(frame, extent=[np.min(x_mesh), np.max(x_mesh), np.min(y_mesh), np.max(y_mesh)])
axs[0].streamplot(x_mesh, y_mesh, U_grid, V_grid, density=2)
axs[0].set_title("interpolacion y streamplot de la interpolacion para destilada lento")

# axs[1].streamplot(x_mesh, y_mesh, U_grid, V_grid, density=1.2, color="black")
axs[1].imshow(frame, extent=[np.min(x_mesh), np.max(x_mesh), np.min(y_mesh), np.max(y_mesh)])
tools.display_vector_field(path_images + save_file_name + ".txt",
                           ax=axs[1],
                           width=0.0035, # width is the thickness of the arrow
                           on_img=False, # overlay on the image
                           # image_name='__file directory for the image__'
                        )
axs[1].set_title("glicerina 50 rapido sin replace outliers (32, 32, 20, 1.2) 120fps")
plt.show()