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
import matplotlib.image as mpimg
from matplotlib import colormaps as cm
from matplotlib import rcParams  # Para aumentar la resolución de los gráficos de Matplot
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
%config InlineBackend.figure_format='retina'
rcParams['font.family'] = 'serif'
rcParams["font.size"] = 16
rcParams['figure.figsize'] = (12, 6)
rcParams['figure.dpi'] = 100
plt.style.use('seaborn-v0_8-poster')

#%%

def calcularCentro(datos, porcentaje=1.0, GRAFICAR=False):
    print(datos.shape)
    # Tirar velocidades nulas
    datos = np .array([dato for dato in datos if dato[2]**2 + dato[3]**2 > 0])
    
    print(datos.shape)
    # Cantidad de puntos para usar
    n = int(len(datos) * porcentaje)
    
    # Ordenar según el módulo de la velocidad y quedarnos con las más altas
    ordenados = np.array(sorted(datos, key=lambda row : row[2]**2 + row[3]**2, reverse=True))[:n]
    
    print(ordenados.shape)
    x, y, u, v, err_u, err_v = ordenados.T
    
    # Pasar a polares
    # El campo de velocidades en cartesianas es (u, v)
    theta = np.arctan(v / u) # ángulos que forma el vector velocidad con la horizontal

    # Calculo la pendiente de la recta normal a la velocidad
    m = np.tan(theta - np.pi / 2) # pendientes

    # La muevo hasta la posición de la partícula
    b = y - m * x # ordenadas al origen

    # Listas para guardar las coordenadas (x, y) entre cada par de rectas
    X_intercept = []
    Y_intercept = []

    # Calcular la intersección de cada par posible de rectas
    for i in range(len(x)):
        m1 = m[i]
        b1 = b[i]
        for j in range(i+1,len(x)):
            m2 = m[j]
            b2 = b[j]
            x_intercept = (b2-b1)/(m1-m2)
            y_intercept = x_intercept*m1 + b1
            X_intercept.append(x_intercept)
            Y_intercept.append(y_intercept)
    
    if GRAFICAR:
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
    
    return (x_centro, y_centro), (err_x_centro, err_y_centro)

def cargarDatos(video):
    # Abrir posiciones
    datosPosiciones = np.load(video + 'posiciones.npz')
    
    # Abrir campo de velocidades promediado (el que se guarda en calcularCampos.py)
    datosVel = np.load(video + 'promedio-vel.npz')

    x, y, u, v, err_u, err_v = datosPosiciones['x'], datosPosiciones['y'], datosVel['U'], datosVel['V'], datosVel['Uerr'], datosVel['Verr']
    
    print(x.shape)
    datos = np.column_stack((x, y, u, v, err_u, err_v))
    
    return datos
    
def cmap(u, v, colormap):
    color = np.hypot(u, v)
    color = (color - min(color)) / (max(color) - min(color))
    return cm[colormap](color)

def burgers(r, Omega, c):
    return Omega * c**2 * (1 - np.exp(-r**2 / c**2)) / r

def velTangencial(datos, centro):
    print(datos.shape)
    x, y, u, v, err_u, err_v = datos.T
    
    # Cambio el origen
    x_desplazado, y_desplazado = x - centro[0], y - centro[1]

    # Tiro las velocidades nulas
    x_filtrado = x_desplazado[x_desplazado != 0]
    y_filtrado = y_desplazado[x_desplazado != 0]
    u_filtrado = u[x_desplazado != 0]
    v_filtrado = v[x_desplazado != 0]
    
    x_filtrado = x_filtrado[u_filtrado != 0]
    y_filtrado = y_filtrado[u_filtrado != 0]
    v_filtrado = v_filtrado[u_filtrado != 0]
    u_filtrado = u_filtrado[u_filtrado != 0]
    

    # Paso a polares
    r = np.sqrt(x_filtrado**2 + y_filtrado**2)
    th = np.arctan(y_filtrado / x_filtrado)

    # Calculo velocidad tangencial
    beta = np.arccos((u_filtrado * x_filtrado + v_filtrado * y_filtrado) / (r * np.sqrt(u_filtrado**2 + v_filtrado**2)))
    vt = np.sqrt(u_filtrado**2 + v_filtrado**2) * np.cos(np.pi / 2 - beta)

    #Todo lo que viene a partir de ahora es para que promedie los datos
    #Hasta le agregó el error, chequear!!!

    # Número de bins para dividir los datos radiales
    num_bins = 20

    # Calcular el radio para cada punto
    r_points = np.sqrt(x_filtrado**2 + y_filtrado**2)

    # Calcular el histograma radial
    hist, bin_edges = np.histogram(r_points, bins=num_bins)

    # Calcular el promedio de las velocidades tangenciales en cada bin
    vt_avg = np.zeros(num_bins)
    vt_err = np.zeros(num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    for i in range(num_bins):
        mask = (r_points >= bin_edges[i]) & (r_points < bin_edges[i + 1])
        
        seleccionados = vt[mask]
        
        print(len(seleccionados))
        
        vt_avg[i] = np.mean(seleccionados)
        vt_err[i] = np.std(seleccionados) / np.sqrt(len(seleccionados))
    
    return bin_centers, vt_avg, vt_err



#%%
# Videos
velocidades = ['3-5', '4', '4-5']
fluidos = [30, 37, 50]

#%%

### Primero cargamos los datos y metemos las coordenadas y las velocidades en una lista.
video = 'cuad3-5/' # <----- ACÁ VA EL VIDEO PARA ANALIZAR

datos = cargarDatos(video)

centroTot, c = calcularCentro(datos)
centroFiltrado, c = calcularCentro(datos, 0.05)
#%%
plt.close('all')
def burgers(r, Omega, c):
    return Omega * (1 - np.exp(-r**2 / c**2)) / r

fluidos = [37]
# Comparo todas las velocidades para un mismo fluido
for fluido in fluidos:   
    fig, ax = plt.subplots()
    ax.set_title(f"Glicerina {fluido}%")
    
    for vel in velocidades:
        video = f"{fluido}v{vel}e/"
        
        if not os.path.isdir(video):
            continue
        
        # Abrir datos
        datos = cargarDatos(video)
        
        # Calcular centro
        centro, error = calcularCentro(datos, porcentaje=0.05)
        
        # Graficar campo de velocidades
        plt.figure()
        plt.title(video)
        x, y, u, v, u_err, v_err = datos.T
        plt.quiver(x, y, u, v, color=cmap(u, v, 'plasma'))
        plt.scatter(centro[0], centro[1])
        plt.show()
        
        # Calcular velocidad tangencial
        r, vt, err_vt = velTangencial(datos, centro)
        
        # Graficar
        ax.errorbar(r, vt, yerr=err_vt, label=vel)
    
    ax.legend()
    
#%%
plt.close('all')
datos = cargarDatos('50v3/')

centro, error = calcularCentro(datos, porcentaje=0.02)

plt.figure()
x, y, u, v, u_err, v_err = datos.T
plt.quiver(x, y, u, v, color=cmap(u, v, 'plasma'))
plt.scatter(centro[0], centro[1])
plt.show()




x, y, u, v, err_u, err_v = datos.T

# Cambio el origen
x_desplazado, y_desplazado = x - centro[0], y - centro[1]

# Tiro las velocidades nulas
x_filtrado = x_desplazado[x_desplazado != 0]
y_filtrado = y_desplazado[x_desplazado != 0]
u_filtrado = u[x_desplazado != 0]
v_filtrado = v[x_desplazado != 0]

x_filtrado = x_filtrado[u_filtrado != 0]
y_filtrado = y_filtrado[u_filtrado != 0]
v_filtrado = v_filtrado[u_filtrado != 0]
u_filtrado = u_filtrado[u_filtrado != 0]


# Paso a polares
r = np.sqrt(x_filtrado**2 + y_filtrado**2)
th = np.arctan(y_filtrado / x_filtrado)

# Calculo velocidad tangencial
beta = np.arccos((u_filtrado * x_filtrado + v_filtrado * y_filtrado) / (r * np.sqrt(u_filtrado**2 + v_filtrado**2)))
vt = np.sqrt(u_filtrado**2 + v_filtrado**2) * np.cos(np.pi / 2 - beta)

#Todo lo que viene a partir de ahora es para que promedie los datos
#Hasta le agregó el error, chequear!!!

# Número de bins para dividir los datos radiales
num_bins = 40

# Calcular el radio para cada punto
r_points = np.sqrt(x_filtrado**2 + y_filtrado**2)

# Calcular el histograma radial
hist, bin_edges = np.histogram(r_points, bins=num_bins)

# Calcular el promedio de las velocidades tangenciales en cada bin
vt_avg = np.zeros(num_bins)
vt_err = np.zeros(num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

for i in range(num_bins):
    mask = (r_points >= bin_edges[i]) & (r_points < bin_edges[i + 1])
    
    seleccionados = vt[mask]
    
    print(len(seleccionados) == hist[i])
    
    vt_avg[i] = np.mean(seleccionados) if len(seleccionados) != 0 else 0
    vt_err[i] = np.std(seleccionados) / np.sqrt(len(seleccionados)) if len(seleccionados) != 0 else 0


vt = vt_avg
r = bin_centers
inicio = 0

r = r[vt > 1.5]

vt_err = vt_err[vt > 1.5]
vt = vt[vt> 1.5]

def rankine(r, Omega, c):
    return np.piecewise(r, [r < c, r >= c], [lambda x : x * Omega, lambda x : Omega * c**2 / x])

popt, pcov = curve_fit(burgers, r, vt, sigma = vt_err, absolute_sigma = True, p0=(4.33, 2.33))

graf = np.linspace(min(r), max(r), 1000)

fig, ax = plt.subplots()
# Graficar
ax.plot(r, vt, ".")
ax.plot(graf, burgers(graf, *popt))
# ax.plot(x, y, ".")
#%%
plt.close('all')
plt.figure(figsize=(8,8))
plt.imshow(plt.imread(video + 'cuadros/0430.jpg'), extent = [min(datos[:, 0]), max(datos[:, 0]), min(datos[:, 1]), max(datos[:, 1])])
plt.quiver(datos[:, 0], datos[:, 1], datos[:, 2], datos[:, 3], color="red", alpha=0.7)

plt.scatter([x_v], [y_v], label = "Filtrados")
# plt.scatter([x2], [y2], label="Sin filtrar")
# plt.legend()


#%%
C = cmap(datos[:, 2], datos[:, 3], 'plasma')

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(plt.imread(video + 'cuadros/0544.jpg'), extent = [min(x), max(x), min(y), max(y)])
ax.quiver(x,y,u,v, color=C)
# ax.scatter(x_centro, y_centro, c='r')
plt.xlabel('Distancia [cm]')
plt.ylabel('Distancia [cm]')
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

#Todo lo que viene a partir de ahora es para que promedie los datos
#Hasta le agregó el error, chequear!!!

# Número de bins para dividir los datos radiales
num_bins = 45

# Calcular el radio para cada punto
r_points = np.sqrt(x_filtrado**2 + y_filtrado**2)

# Calcular el histograma radial
hist, bin_edges = np.histogram(r_points, bins=num_bins)

# Calcular el promedio de las velocidades tangenciales en cada bin
vt_avg = np.zeros(num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

for i in range(num_bins):
    mask = (r_points >= bin_edges[i]) & (r_points < bin_edges[i + 1])
    vt_avg[i] = np.mean(vt[mask])

# Ajustar y graficar
popt, pcov = curve_fit(burgers, bin_centers, vt_avg, absolute_sigma=True) # Agregue sigma si tiene datos de error

graf = np.linspace(np.min(bin_centers), np.max(bin_centers), 1000)

plt.figure()
plt.plot(graf, burgers(graf, *popt))
plt.errorbar(bin_centers, vt_avg, yerr=np.std(vt[mask]), fmt='o', label='Datos Promediados')
plt.xlabel('Distancia al centro del vórtice [cm]')
plt.ylabel('Velocidad Tangencial [cm/s]')
plt.legend()
plt.show()

#%%

#Por si queremos graficar sólo los datos, sin el ajuste =

plt.figure()
plt.errorbar(bin_centers, vt_avg, yerr=np.std(vt[mask]), fmt='o')
plt.xlabel('Distancia al centro del vórtice [cm]')
plt.ylabel('Velocidad Tangencial [cm/s]')
plt.grid()
#plt.savefig('velocidadtang.png')
plt.show()



#%%


#PERTENECE AL CÓDIGO ANTERIOR, lo dejo acá abajo por las dudas
#Aca se ve el ajuste con todos los puntos dispersos sin promediar

# Modelos de vórtice
def rankine(r, Omega, c):
    return np.piecewise(r, [r < c, r >= c], [lambda x : x * Omega, lambda x : Omega * c**2 / x])

# Ajusto con Burgers y grafico
popt, pcov = curve_fit(burgers, r, vt, absolute_sigma=True) # <---- FALTA AGREGAR SIGMA

graf = np.linspace(np.min(r), np.max(r), 1000)

plt.figure()
plt.plot(graf, burgers(graf, *popt))
plt.plot(r, vt, ".")