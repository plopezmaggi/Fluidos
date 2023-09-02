# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:42:46 2023

@author: Publico
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from glob import glob
from pprint import pprint

import cv2

from utils.preprocess.extract import FrameExtractor
from utils.preprocess.enhance import Enhancer

# Parametros de preprocesamiento para cada video.
# x e y: Offset del centro en x e y.
# r: Radio del circulo que enmascara todo.
# start: Segundo en el que iniciar a extraer.
# total_secs: Segundos totales a extraer.
# scale: Conversion de px a cm.
# Crear una fila por cada nombre de archivo como en el archivo de ejemplo abajo.
filename = "preproc_data.csv"

df = pd.read_csv(filename, sep=",", engine='python').set_index("filename")
display(df)

#%% 

# Extract Frames

# Extraer videos de la carpeta para preprocesar.
folder = './videos utilizables/*'
videos = list(map(Path, glob(folder)))
pprint(videos)

for v in videos:
    start_sec = int(df.loc[v.name].loc['start'])

    # Para buscar los parametros cambiar esta linea a 0.1 por ejemplo
    total_secs = int(df.loc[v.name].loc['total_secs'])

    extractor = FrameExtractor(
                    video_file=str(v), # Nombre del archivo
                    output_dir=f"images/{v.name}", #Se guarda en una carpeta images/
                    output_shape_ratio = 1, #Achicar la resolucion final por un factor.
                    crop_image = False, # Recortar imagen para sacar region donde no está el recipiente.
                    crop_circle_radius = int(df.loc[v.name].loc["r"]), # Radio de recorte.
                    circle_offset = (int(df.loc[v.name].loc["x"]), #Offset en X e Y del centro.
                                     int(df.loc[v.name].loc["y"]))) 

    extractor.extract(start_frame = 30*start_sec, 
                      end_frame = int(30*(start_sec + total_secs)))

    print()
    
#%%
    # Ademas extraigo con el mismo codigo un solo fotograma para tomar de la regla en el video la relacion px/cm
    
df = pd.read_csv("preproc_data.csv", sep=",", engine='python').set_index("filename")

for v in videos:
    print(f"Extracting {v}.")
    start_sec = int(df.loc[v.name].loc['start'])
    total_secs = 0.1/2

    extractor = FrameExtractor(
                    video_file=str(v),
                    output_dir=f"frames_for_pxcm/{v.name}",
                    output_shape_ratio = 1,
                    crop_image = True,
                    crop_circle_radius = 1000,
                    circle_offset = (int(df.loc[v.name].loc["x"]), 
                                     int(df.loc[v.name].loc["y"])))

    extractor.extract(start_frame = 30*start_sec, 
                      end_frame = int(30*(start_sec + total_secs)))

    print()
    
    #%%
    # Enhance Frames
    
    # Mejorar fotogramas ya obtenidos para el preprocesamiento de PIV.

folder = "./images/*"
folders = list(map(Path, glob(folder)))

pprint(folders)

enhancer = Enhancer(use_threshold=False,
                    use_substract_background=True)

for f in folders:
    print(f"Enhancing {f.name}.")
    output = os.path.join("proc_images", f.name) # Se guardan en una carpeta proc_images/
    enhancer.process_frames(f, output)