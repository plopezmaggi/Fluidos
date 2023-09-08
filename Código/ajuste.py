#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:29:28 2023

@author: plopezmaggi
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# %matplotlib nbagg
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import rcParams  # Para aumentar la resolución de los gráficos de Matplot
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
# %config InlineBackend.figure_format='retina'
rcParams['font.family'] = 'serif'
rcParams["font.size"] = 16
rcParams['figure.figsize'] = (12, 6)
rcParams['figure.dpi'] = 100
plt.style.use('seaborn-dark-palette')

#%%

### Primero cargamos los datos y metemos las coordenadas y las velocidades en una lista.
video = '37v3-5e/' # <----- ACÁ VA EL VIDEO PARA ANALIZAR

# Abrir campo de velocidades promediado (el que se guarda en procesamiento.py)
datosPosiciones = np.load(video + 'posiciones.npz')
datosVel = np.load(video + 'promedio-vel.npz')

# Definimos las coordenadas y las velocidades
x = datosPosiciones['x'].reshape(datosPosiciones['x'].shape[0] * datosPosiciones['x'].shape[1])
y = datosPosiciones['y'].reshape(datosPosiciones['y'].shape[0] * datosPosiciones['y'].shape[1])
u = datosVel['U'].reshape(datosVel['U'].shape[0] * datosVel['U'].shape[1])
v = datosVel['V'].reshape(datosVel['V'].shape[0] * datosVel['V'].shape[1])
err_u = datosVel['Uerr'].reshape(datosVel['Uerr'].shape[0] * datosVel['Uerr'].shape[1])
err_v = datosVel['Verr'].reshape(datosVel['Verr'].shape[0] * datosVel['Verr'].shape[1])

#%%

# Gráfico del campo de velocidades
color = np.hypot(u, v)
color = (color - min(color)) / (max(color) - min(color))
C = plt.cm.Blues(color)

fig, ax = plt.subplots(1,1, figsize=(8,8))
Q = ax.quiver(x,y,u,v, color=C)

# Seleccionamos los puntos que utilizaremos para calcular el centro (filtramos con un mínimo de velocidad)
v_threshold = 2.5 # <---- MIRAR QUÉ CONVIENE PONER ACÁ

idx = np.array([i for i in range(len(x)) if (u[i]**2 + v[i]**2) > v_threshold**2])

if len(idx) > 0:
    x_sel = x[idx]
    y_sel = y[idx]
    u_sel = u[idx]
    v_sel = v[idx]

else:
    x_sel, y_sel, u_sel, v_sel = np.array([]), np.array([]), np.array([]), np.array([])

ax.scatter(x_sel, y_sel, c='r', s=5)

#%%

"""
Para calcular el centro, se grafican para un conjunto de puntos las rectas perpendiculares a sus velocidades,
que se cruzarán en el centro del vórtice si las velocidades son tangenciales.
Para esto se calcula la pendiente utilizando el ángulo theta formado entre la velocidad y la dirección horizontal,
y la ordenada al origen a partir de pedir que la recta pase por el punto, de coordenadas (x,y).
Luego se buscan los puntos de intersección entre todo par de rectas y se calcula el valor medio de todos ellos,
donde se determina el centro de coordenadas.
"""

# Pasar a polares
# El campo de velocidades en cartesianas es (u, v)
theta = np.arctan(v_sel / u_sel) # ángulos que forma el vector velocidad con la horizontal

# Calculo la pendiente de la recta normal a la velocidad
m = np.tan(theta - np.pi / 2) # pendientes

# La muevo hasta la posición de la partícula
b = y_sel - m * x_sel # ordenadas al origen

# Listas para guardar las coordenadas (x, y) entre cada par de rectas
X_intercept = []
Y_intercept = []

# Calcular la intersección de cada par posible de rectas
for i in range(len(x_sel)):
    m1 = m[i]
    b1 = b[i]
    for j in range(i+1,len(x_sel)):
        m2 = m[j]
        b2 = b[j]
        x_intercept = (b2-b1)/(m1-m2)
        y_intercept = x_intercept*m1 + b1
        X_intercept.append(x_intercept)
        Y_intercept.append(y_intercept)

# Graficar todas las rectas y las intersecciones
fig, ax = plt.subplots()

X_lin = np.linspace(np.min(x), np.max(x), 10) # vector de valores de x para graficar

for i in range(len(x_sel)):
    ax.plot(X_lin, m[i]*X_lin + b[i], c='k', lw=.5, alpha=.5)
    
ax.scatter(X_intercept, Y_intercept, c='r', s=5, alpha=.4, zorder=5)
ax.set_ylim(np.min(y), np.max(y))
ax.set_xlim(np.min(x), np.max(x))
ax.quiver(x,y,u,v, color=C, edgecolor='k', linewidth=.3, zorder=3)
plt.show()

# Finalmente, buscamos el centro como el punto medio de todas las intersecciones, y lo graficamos:
x_centro = np.mean(X_intercept)
y_centro = np.mean(Y_intercept)

err_x_centro = np.std(X_intercept)
err_y_centro = np.std(Y_intercept)

fig, ax = plt.subplots(figsize=(8,8))
ax.quiver(x,y,u,v, color=C)
ax.scatter(x_centro, y_centro, c='r')
plt.show()

#%%
"""
Ajustamos la velocidad tangencial en función del radio con ambos modelos.

El threshold que elegimos más arriba corta los puntos con menor velocidad,
así que elimina la zona donde se ven los efectos de borde
(esto se nota si ponemos un threshold muy chico, el modelo deja de ajustar).
"""
plt.close('all')

# Cambio el origen
x_desplazado, y_desplazado = x_sel - x_centro, y_sel - y_centro

# Me quedo con el semiespacio de y > 0 (porque si no el cálculo de theta no es inyectivo)
x_filtrado = x_desplazado[x_desplazado != 0]
y_filtrado = y_desplazado[x_desplazado != 0]
u_filtrado = u_sel[x_desplazado != 0]
v_filtrado = v_sel[x_desplazado != 0]

# Paso a polares
r = np.sqrt(x_filtrado**2 + y_filtrado**2)
th = np.arctan(y_filtrado / x_filtrado)

# Calculo velocidad tangencial
beta = np.arccos((u_filtrado * x_filtrado + v_filtrado * y_filtrado) / (r * np.sqrt(u_filtrado**2 + v_filtrado**2)))
vt = np.sqrt(u_filtrado**2 + v_filtrado**2) * np.cos(np.pi / 2 - beta)

# Modelos de vórtice
def rankine(r, Omega, c):
    return np.piecewise(r, [r < c, r >= c], [lambda x : x * Omega, lambda x : Omega * c**2 / x])

def burgers(r, Omega, c):
    return Omega * c**2 * (1 - np.exp(-r**2 / c**2)) / r

# Ajusto con Burgers y grafico
popt, pcov = curve_fit(burgers, r, vt, absolute_sigma=True) # <---- FALTA AGREGAR SIGMA

graf = np.linspace(np.min(r), np.max(r), 1000)

plt.figure()
plt.plot(graf, burgers(graf, *popt))
plt.plot(r, vt, ".")