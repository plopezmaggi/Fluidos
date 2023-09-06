#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Wed Sep  6 17:50:22 2023

@author: plopezmaggi

Corta un frame de un video. Asume que el video está en la carpeta que Spyder tiene abierta, y guarda ahí la imagen.
"""

import cv2

video = "50velocidad3,5estable.mp4"

video = cv2.VideoCapture(video)

for i in range(100):
    frame = video.read()[1]

cv2.imwrite("cuadro.tif", frame)