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
    
    return [(x_centro, y_centro), (err_x_centro, err_y_centro)]

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

def rankine(r, Omega, c):
    return np.piecewise(r, [r < c, r >= c], [lambda x : x * Omega, lambda x : Omega * c**2 / x])

def velTangencial(datos, centro):
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
        
        vt_avg[i] = np.mean(seleccionados) if len(seleccionados) != 0 else 0
        vt_err[i] = np.std(seleccionados) / np.sqrt(len(seleccionados)) if len(seleccionados) != 0 else 0



    popt, pcov = curve_fit(burgers, bin_centers, vt_avg, sigma = vt_err, absolute_sigma = True)
    
    return bin_centers, vt_avg, vt_err, popt, pcov
    return r, vt, vt_err, popt, pcov

#%%

"""
Loop sobre todos los videos (no lo usamos)
"""
plt.close('all')

# Videos
velocidades = ['3-5', '4', '4-5']
fluidos = [30, 37, 50]

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
"""
Comparación para v3
"""
plt.close('all')
datos = cargarDatos('30v3/')

centro, error = calcularCentro(datos, porcentaje=0.02)
# centro = (5.89, 5.93) 50v3
centro = (5.5, 5.4)
plt.figure(figsize=(8,8))
x, y, u, v, u_err, v_err = datos.T
plt.quiver(x, y, u, v, color=cmap(u, v, 'plasma'))
plt.scatter(centro[0], centro[1])
plt.show()


r, vt, err_vt, popt, pcov = velTangencial(datos, centro)
<<<<<<< HEAD
minvel = 2

=======
minvel = 1.5
print(r.shape, vt.shape, err_vt.shape)
>>>>>>> a386d03da5c2319041b0c7f917b0652b080823e6
r = r[vt >= minvel]
# err_vt = err_vt[vt>=minvel]
vt = vt[vt>=minvel]

graf = np.linspace(min(r), max(r), 1000)

fig, ax = plt.subplots()
ax.plot(r, vt, ".", label = "Glicerina 30%")
ax.plot(graf, burgers(graf, *popt))
#%%
datos50 = cargarDatos('50v3/')

centro50, error50 = calcularCentro(datos50, porcentaje=0.02)
centro50 = (5.89, 5.93)
plt.figure(figsize=(8,8))
x50, y50, u50, v50, u_err50, v_err50 = datos50.T
plt.quiver(x50, y50, u50, v50, color=cmap(u50, v50, 'plasma'))
plt.scatter(centro50[0], centro50[1])
plt.show()


r50, vt50, err_vt50, popt50, pcov50 = velTangencial(datos50, centro50)

minvel50 = 1.5

r50 = r50[vt50 >= minvel50]
err_vt50 = err_vt50[vt50>=minvel50]
vt50 = vt50[vt50>=minvel50]

graf50 = np.linspace(min(r50), max(r50), 1000)
# Graficar
ax.plot(r50, vt50, ".", label = "Glicerina 50%")
ax.plot(graf50, burgers(graf50, *popt50))
# ax.plot(x, y, ".")
ax.legend()


#%%

videos = ['30v4e/', '30v3/', '30v3-5e/']

datos = {video : cargarDatos(video) for video in videos}

centros = {video : calcularCentro(datos[video], porcentaje=0.05) for video in videos}
centros['30v4e/'][0] = (5.53, 4.88)
centros['30v3/'][0] = (5.5, 5.37)
centros['30v3-5e/'][0] = (5.38, 4.8)

tangencial = {video : velTangencial(datos[video], centros[video][0]) for video in videos}

#%%

plt.close('all')

figAjuste, axAjuste = plt.subplots()

for video in videos:
    x, y, u, v, err_u, err_v = datos[video].T
    r, vt, err_vt, popt, pcov = tangencial[video]
    
    velmin = 1
    r = r[vt > velmin]
    err_vt = err_vt[vt > velmin]
    vt = vt[vt > velmin]
    figCampo, axCampo = plt.subplots(figsize=(10, 10))
    axCampo.set_title(video)
    axCampo.quiver(x, y, u, v, color=cmap(u, v, 'plasma'))
    axCampo.scatter([centros[video][0][0]], [centros[video][0][1]])
    
    graf = np.linspace(min(r), max(r), 1000)
    
    
    axAjuste.errorbar(r, vt, yerr = err_vt, label = video)
    axAjuste.plot(graf, burgers(graf, *popt))
axAjuste.legend()

#%%
#PARA AJUSTAR COMO ANTES, UN SOLO VIDEITO
plt.close('all')
datos = cargarDatos('30v3/')

centro, error = calcularCentro(datos, porcentaje=0.02)
# centro = (5.89, 5.93) 50v3
centro = (5.50, 5.37)
plt.figure(figsize=(8,8))
x, y, u, v, u_err, v_err = datos.T
plt.quiver(x, y, u, v, color=cmap(u, v, 'plasma'))
plt.scatter(centro[0], centro[1])
plt.show()


r, vt, err_vt, popt, pcov = velTangencial(datos, centro)
minvel = 1.5

r = r[vt >= minvel]
err_vt = err_vt[vt>=minvel]
vt = vt[vt>=minvel]

graf = np.linspace(min(r), max(r), 1000)

fig, ax = plt.subplots()
ax.plot(r, vt, ".", label = "Glicerina 30%")
ax.plot(graf, burgers(graf, *popt))

#%%

#Grafico definitivo para 1 único video con barras de error y todo lindo

plt.plot(r, vt, ".")
plt.plot(graf, burgers(graf, *popt))
plt.ylabel('Velocidad tangencial [cm/s]')
plt.xlabel('Distancia al centro del vórtice [cm]')
plt.errorbar(r, vt, yerr=err_vt, fmt='.')
plt.grid()
#plt.savefig('ajuste.png')

#%%

#Celda para el análisis estadístico

import scipy.stats as stats

#Defino parámetros que vamos a necesitar para el análisis

puntos = len(r)
params = len(popt)
y = vt
y_modelo = burgers(r,popt[0],popt[1]) #---> Acá cambiar según el modelo que estemos ajustando
yerr = err_vt

promedio = np.mean(y)
residuo = y - y_modelo
TSS = sum((y-promedio)**2)
RSS = sum(residuo**2)
ESS = sum((y_modelo-promedio)**2)

#Chi cuadrado: Calcula chi^2, p-valor y a partir de estos resultados tira la conclusión

chi_cuadrado = np.sum(((y-y_modelo)/yerr)**2)
p_chi = stats.chi2.sf(chi_cuadrado, puntos - 1 - params)

print('chi^2: ' + str(chi_cuadrado))
print('p-valor del chi^2: ' + str(p_chi))
if yerr[0]==0:
    print('No se declararon errores en la variable y.')
elif p_chi<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste a los datos.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste a los datos.')

#R cuadrado

R_cuadrado = 1-RSS/TSS
R_cuadrado_aj = 1-(RSS/TSS)*(puntos-params)/(puntos-1)

print('R^2: ' + str(R_cuadrado))
print('R^2 ajustado: ' + str(R_cuadrado_aj))

#F-test: Nos dice si la dependencia es por azar o no. Lo calcula y a partir de esos resultados nos tira la conclusión

F = ESS*(puntos-params)/RSS/(params-1)
p_f = stats.f.sf(F,params-1,puntos-params)

print('Test F: ' + str(F))
print('p-valor del F: ' + str(p_f))
if p_f<0.05:
    print('Se rechaza la hipótesis de que el modelo ajuste tan bien como uno sin variables independientes.')
else:
    print('No se puede rechazar la hipótesis de que el modelo ajuste tan bien como uno sin variables independientes.')

