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

fluido = 'destilada' # 'destilada', 'glicerina 25', 'glicerina 50'
velocidad = 'rapido' # 'lento', 'medio', 'rapido'

# Abrir campo de velocidades promediado (el que se guarda en procesamiento.py)
path = "nuevas mediciones/%s %s dos filtros/%s %s.npz" %(fluido,velocidad,fluido,velocidad)
f = np.load(path)

#%%

# Definimos las coordenadas y las velocidades
x = f['x'].reshape(f['x'].shape[0]*f['x'].shape[1])
y = f['y'].reshape(f['y'].shape[0]*f['y'].shape[1])
u = f['U'].reshape(f['U'].shape[0]*f['U'].shape[1])
v = f['V'].reshape(f['V'].shape[0]*f['V'].shape[1])
err_u = f['Uerr'].reshape(f['Uerr'].shape[0]*f['Uerr'].shape[1])
err_v = f['Verr'].reshape(f['Verr'].shape[0]*f['Verr'].shape[1])

# Recorte de la imagen. Esto creo que no hace falta si usamos las imágenes recortadas con preprocess
mins_x = {'destilada lento': 0, 'destilada medio': 0, 'destilada rapido': 1.5,
         'glicerina 25 lento': 1.5, 'glicerina 25 rapido': 2.5,
          'glicerina 50 lento': 2.5, 'glicerina 50 rapido': 2.5}
maxs_x = {'destilada lento': 20, 'destilada medio': 20, 'destilada rapido': 18.5,
         'glicerina 25 lento': 17.5, 'glicerina 25 rapido': 18,
          'glicerina 50 lento': 18, 'glicerina 50 rapido': 18}
mins_y = {'destilada lento': 2.5, 'destilada medio': 2.5, 'destilada rapido': 2.5,
         'glicerina 25 lento': 2.5, 'glicerina 25 rapido': 2.5,
          'glicerina 50 lento': 2.5, 'glicerina 50 rapido': 2.5}
maxs_y = {'destilada lento': 18, 'destilada medio': 18, 'destilada rapido': 18,
         'glicerina 25 lento': 16, 'glicerina 25 rapido': 18,
          'glicerina 50 lento': 17, 'glicerina 50 rapido': 17}
min_x = mins_x['%s %s' %(fluido,velocidad)]
max_x = maxs_x['%s %s' %(fluido,velocidad)]
min_y = mins_y['%s %s' %(fluido,velocidad)]
max_y = maxs_y['%s %s' %(fluido,velocidad)]

idx_recorte = np.array([i for i in range(len(x)) if (x[i]>min_x and x[i]<max_x) and (y[i]>min_y and y[i]<max_y)])
x = x[idx_recorte]
y = y[idx_recorte]
u = u[idx_recorte]
v = v[idx_recorte]
err_u = err_u[idx_recorte]
err_v = err_v[idx_recorte]
######################

# Gráfico del campo de velocidades
color = np.hypot(u, v)
color = (color - min(color)) / (max(color) - min(color))
C = plt.cm.Blues(color)

fig, ax = plt.subplots(1,1, figsize=(8,8))
Q = ax.quiver(x,y,u,v, color=C)

# Seleccionamos los puntos que utilizaremos para calcular el centro (filtramos con un mínimo de velocidad)
v_threshold = 7.75

idx = np.array([i for i in range(len(x)) if (u[i]**2 + v[i]**2) > v_threshold**2])
x_sel = x[idx]
y_sel = y[idx]
u_sel = u[idx]
v_sel = v[idx]

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

# F = [] # acá almacenaremos todas las rectas
# for i in range(len(x_sel)):
    # f = lambda X: m[i]*X + b[i] # ecuación de la recta de un solo punto
#     F.append(f)

# Buscamos los puntos de intersección entre todas las rectas
X_intercept = []
Y_intercept = []
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

# graficamos todas las rectas y los puntos de intersección:
fig, ax = plt.subplots()
X_lin = np.linspace(min_x,max_x,10) # vector de valores de x para graficar
for i in range(len(x_sel)):
    ax.plot(X_lin, m[i]*X_lin + b[i], c='k', lw=.5, alpha=.5)
ax.scatter(X_intercept, Y_intercept, c='r', s=5, alpha=.4, zorder=5)
ax.set_ylim(min_y,max_y)
ax.set_xlim(min_x,max_x)
ax.quiver(x,y,u,v, color=C, edgecolor='k', linewidth=.3, zorder=3)
plt.show()

#%%

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

# histograma 3D de los puntos de intersección:

bins = 64
# xbins = np.sort(np.concatenate((-np.logspace(0, np.log10(abs(min_x-x_centro)), int(bins/2)), 
#                        np.logspace(0, np.log10(abs(max_x-x_centro)), int(bins/2)))))
# ybins = np.sort(np.concatenate((-np.logspace(0, np.log10(abs(min_y-y_centro)), int(bins/2)), 
#                        np.logspace(0, np.log10(abs(max_y-y_centro)), int(bins/2)))))

r_max = np.sqrt((max_x-x_centro)**2 + (max_y-y_centro)**2)
xbins_pos = np.logspace(0, np.log10(abs(max_x-x_centro)), bins) # bines positivos en x
xbins_neg = np.sort(-np.logspace(0, np.log10(abs(min_x-x_centro)), bins)) # bines negativos en x
xbins = np.concatenate((xbins_neg, xbins_pos)) # bines en x
ybins_pos = np.logspace(0, np.log10(abs(max_y-y_centro)), bins) # bines positivos en y
ybins_neg = np.sort(-np.logspace(0, np.log10(abs(min_y-y_centro)), bins)) # bines negativos en y
ybins = np.concatenate((ybins_neg, ybins_pos)) # bines en y

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# hist, xedges, yedges = np.histogram2d(X_intercept-x_centro, Y_intercept-y_centro, bins=[xbins, ybins]) 
hist, xedges, yedges = np.histogram2d(X_intercept, Y_intercept, bins=bins, range=[[min_x,max_x],[min_y,max_y]])

# Construct arrays for the anchor positions of the *bins x bins* bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.4/bins, yedges[:-1] + 0.4/bins, indexing="ij")
# xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij") 
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the *bins x bins* bars.
dx = ((max_x - min_x)/(bins+1)) * np.ones_like(zpos)
dy = ((max_y - min_y)/(bins+1)) * np.ones_like(zpos)

# dx = np.array([abs(xbins[i]-xbins[i-1])*np.ones(len(xedges[:-1])) for i in range(1,len(xbins))]) 
# dy = np.array([abs(ybins[i]-ybins[i-1])*np.ones(len(yedges[:-1])) for i in range(1,len(ybins))]) 
dz = hist.ravel()
# dx = dx.reshape(len(dz))
# dy = dy.reshape(len(dz))
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='red')

plt.show()

#%%

df_hist = pd.DataFrame(hist, index=np.round(np.linspace(min_y, max_y, bins),1), columns=np.round(np.linspace(min_x, max_x, bins), 1))

fig, ax = plt.subplots()

sns.heatmap(np.log1p(df_hist)/np.log(10))
# sns.heatmap(hist)
ax.text(1.16, 0.35, r'$log_{10}(1+cuentas)$', fontsize=18, rotation=90, transform=ax.transAxes)
ax.set_xlabel('Posición en x (cm)', fontsize=18)
ax.set_ylabel('Posición en y (cm)', fontsize=18)
# ax.set_xlim(60,68)
# ax.set_ylim(60,72)
plt.show()

#%%


# ordenamos los puntos y sus velocidades según el módulo de su velocidad:
v_abs = np.hypot(u, v)
idx_sort = np.argsort(v_abs)
x_sorted = x[idx_sort]
y_sorted = y[idx_sort]
u_sorted = u[idx_sort]
v_sorted = v[idx_sort]
v_abs = v_abs[idx_sort]

# array con distintos valores del parámetro v_threshold
V_thresholds = np.linspace(v_abs[1], v_abs[-2], 100)

thetas = np.arctan(v_sorted/u_sorted) # ángulos que forma el vector velocidad con la horizontal

# calculo las rectas de cada punto que contienen al vector posición medido desde el centro:
ms = np.tan(thetas-np.pi/2) # pendientes
bs = y_sorted - ms*x_sorted # ordenadas al origen

# Buscamos los puntos de intersección entre todas las rectas:

# dónde guardaremos las coordenadas de los puntos de intersección:
# X_intercept = np.zeros((len(x_sorted), len(x_sorted)))
# Y_intercept = np.zeros((len(x_sorted), len(x_sorted)))

X_intercept = []
Y_intercept = []
for i in range(len(x_sorted)):
    m1 = ms[i]
    b1 = bs[i]
    for j in range(i+1,len(x_sorted)):
        m2 = ms[j]
        b2 = bs[j]
        x_intercept = (b2-b1)/(m1-m2)
        y_intercept = x_intercept*m1 + b1
        X_intercept.append(x_intercept)
        Y_intercept.append(y_intercept)
#         X_intercept[i,j] = x_intercept
#         Y_intercept[i,j] = y_intercept

X_centros = []
Y_centros = []
X_centro_stde = []
Y_centro_stde = []
for v_threshold in V_thresholds:
    idx_sel = np.array([i for i in range(len(v_abs)) if v_abs[i] > v_threshold])
    i_min = np.min(idx_sel)
    N = len(v_abs)
    x_sel = X_intercept[-int((N-(i_min-1))*(N-(i_min-1)+1)/2):]
    y_sel = Y_intercept[-int((N-(i_min-1))*(N-(i_min-1)+1)/2):]
    x_mean = np.mean(x_sel)
    y_mean = np.mean(y_sel)
    x_stde = np.std(x_sel)/np.sqrt(len(x_sel))
    y_stde = np.std(y_sel)/np.sqrt(len(y_sel))
    X_centros.append(x_mean)
    Y_centros.append(y_mean)
    X_centro_stde.append(x_stde)
    Y_centro_stde.append(y_stde)
    
#%%

fig1, ax1 = plt.subplots(1,2, sharey=True)

ax1[0].plot(V_thresholds, X_centros)
ax1[1].plot(V_thresholds, Y_centros)
ax1[0].set_title('Posición en x', fontsize=18)
ax1[1].set_title('Posición en y', fontsize=18)
ax1[0].set_xlabel(r'Umbral de |$\vec{v}$| (cm/s)', fontsize=16)
ax1[1].set_xlabel(r'Umbral de |$\vec{v}$| (cm/s)', fontsize=16)
ax1[0].set_ylabel('Posición (cm)', fontsize=16)
# ax1[0].set_yscale('log')
# ax1[1].set_yscale('log')
ax1[0].grid()
ax1[1].grid()

plt.show()

#%%

plt.figure(figsize=(12,6))
plt.plot(V_thresholds, X_centro_stde, c='red', label='Error estándar en x')
plt.plot(V_thresholds, Y_centro_stde, c='blue', label='Error estándar en y')
plt.xlabel(r'Umbral de |$\vec{v}$| (cm/s)', fontsize=16)
plt.ylabel('Error estándar (cm)', fontsize=16)
plt.legend(fontsize=16)
plt.yscale('log')
plt.grid()
plt.show()

#%%

x_corrido =x - X_centro # x_centro, X_centro_stde
y_corrido = y - Y_centro

r = np.abs(x_corrido+1j*y_corrido)
theta = np.angle(x_corrido+1j*y_corrido)
v_r= (x_corrido*u+y_corrido*v)/r
v_theta = (x_corrido*v-y_corrido*u)/(r)

err_v_theta = np.abs(x_corrido)*err_v/r + np.abs(y_corrido)*err_u/r + np.abs(v/r + v_theta*x_corrido/(r**2))*err_x_centro + np.abs(v_theta*y_corrido/(r**2) - u/r)*err_y_centro


#%%

#plt.plot(r,v_r,".",label = "$v_{r}(r)$")
plt.plot(r,v_theta,".",label = "$ v_{\Theta} (r)$", color = "green")
plt.errorbar(r, v_theta, yerr=err_v_theta, fmt='.', c='green', ecolor='black', elinewidth=1, ms=5, label='$ v_{\Theta} (r)$')
plt.xlabel('Radio (cm)')
plt.ylabel('Velocidad (cm/s)')
plt.legend()
plt.grid()
# plt.savefig("Gráficos campo.png")
plt.show()

#%%

def rankine(x,omega,c): # Hay que chequearla porque no se si funciona bien
    f = []
    for i in x:
        if i < c:
            f.append(omega*i)
        else:
            f.append((omega*c**2)/i)
    return np.array(f)

def rankine2(x,omega,c):
    return np.piecewise(x,[x<c, x>=c], [lambda x: omega*x, lambda x: (omega*c**2)/x])

def burgers(x,a,b):
  return a*(1-np.exp(-(x/b)**2))/x

#%%

def ajuste_sin_errores(f_ajuste,r,v_theta,mostrar_parametros = True):
    
    var_x = r
    var_y = v_theta
    
    popt, pcov = curve_fit(f_ajuste, var_x, var_y)#,sigma=err_y,absolute_sigma=True)#, p0 = [-200,0.3,250], maxfev=10000)#, absolute_sigma = True, sigma=err_var_y)
    a, b = popt
    err_a, err_b = np.sqrt(np.diag(pcov))
    
    # Declaramos nuestro nuevo dominio e imagen y graficamos el ajuste
    new_var_x = np.linspace(min(var_x), max(var_x), 10000)
    new_var_y = f_ajuste(new_var_x, a, b)
    
    # Graficamos los datos y el ajuste
    plt.plot(var_x,var_y, 'k.', label = '$v_{\Theta}(r)$')
    plt.plot(new_var_x, new_var_y, 'm-', label='Ajuste')
    # Título y labels
    plt.xlabel("Radio (cm)")
    plt.ylabel("Velocidad (cm/s)")
    plt.grid(True)
    plt.legend()
    plt.savefig('Ajuste promedio.png')
    plt.show()

    # Ahora defino omega y c (radio del vortice) segun el ajuste utilizado
    if mostrar_parametros:
        if f_ajuste == burgers:
            omega = a/(b**2)
            c = b
            err_omega = np.sqrt((err_a/(b**2))**2+(2*err_b*a/(b**3))**2)
            err_c = err_b

        if f_ajuste == rankine or f_ajuste == rankine2:
            omega = a
            c = b
            err_omega = err_a
            err_c = err_b

        print(f"La velocidad angular Omega es {round(omega,3)} con error {round(err_omega,3)}")
        print(f"El radio del vortice c es {round(c,3)} con error {round(err_c,4)}")
    
    return omega, err_omega, c, err_c, popt

#%%

omega, err_omega, c, err_c, popt = ajuste_sin_errores(burgers,r,v_theta, mostrar_parametros=False)

#%%

plt.plot(theta,v_r,".",label = "$v_{r}(r)$")
plt.plot(theta,v_theta,".",label = "$v_\Theta(r)$")
plt.xlabel('Angulo (rad)')
plt.ylabel('Velocidad (px/frame)')
plt.legend()
plt.savefig("Velocidades en funcion del angulo")
plt.show()

#%%

def promediar(r,v_theta, rango = 0.07):
    
    ordenamiento = sorted(zip(r,v_theta)) # Con esto estamos ordenando las listas de menor a mayor segun el radio, manteniendo la velocidad tangencial de cada valor de r
    r_ord = [i[0] for i in ordenamiento]
    v_t_ord = [i[1] for i in ordenamiento]
    
    r_promedio = [] # Definimos las listas donde guardaremos los promedios
    v_t_promedio = []

    i = 0 # Definimos los iteradores
    j = 0

    while i < len(r_ord): # Empezamos un loop
        l_r_prom = [] # Creamos listas para hacer el promedio entre los valores que esten dentro del rango
        l_v_t_prom = []
        while j < len(r_ord) and r_ord[j] <= r_ord[i] + rango: # Recorremos mientras el j no supere la longitud de r_ord y mientras el r j-esimo sea menor al r i-esimo mas el rango
            # En ese caso, metemos los valores j-esimos en las listas
            l_r_prom.append(r_ord[j])
            l_v_t_prom.append(v_t_ord[j])
            j += 1
        if len(l_r_prom) != 0: # Pedimos que las longitudes de las listas sean distinto de cero ya que sino la suma no funciona
            # Realizamos el promedio de los valores que cumplen el rango
            r_promedio.append(sum(l_r_prom)/len(l_r_prom)) 
            v_t_promedio.append(sum(l_v_t_prom)/len(l_v_t_prom))
        i = j # Ahora como ya recorrimos varios valores con j y no queremos volver a repetirlos, definimos que i arranque desde el ultimo j
    
    return r_promedio, v_t_promedio

#%%

r_promedio, v_t_promedio = promediar(r,v_theta[:])

omega, err_omega, c, err_c, popt = ajuste_sin_errores(burgers,r_promedio[:-28],v_t_promedio[:-28])

residuals = v_theta - burgers(r, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((v_theta-np.mean(v_theta))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

#%%

r_promedio, v_t_promedio = promediar(r,v_theta)

omega, err_omega, c, err_c, popt = ajuste_sin_errores(rankine2,r_promedio[:-28],v_t_promedio[:-28])

residuals = v_theta - rankine2(r, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((v_theta-np.mean(v_theta))**2)
r_squared = 1 - (ss_res / ss_tot)
print(r_squared)

#%%

