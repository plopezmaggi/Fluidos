# NI-DAQmx Python Documentation: https://nidaqmx-python.readthedocs.io/en/latest/index.html
# NI USB-621x User Manual: https://www.ni.com/pdf/manuals/371931f.pdf
import matplotlib.pyplot as plt
import numpy as np
import nidaqmx
import math
import time


#para saber el ID de la placa conectada (DevX)
system = nidaqmx.system.System.local()
for device in system.devices:
    print(device)
	

## Medicion por tiempo/samples de una sola vez

def medir(cant_puntos, fs):
  
    with nidaqmx.Task() as task:
        modo= nidaqmx.constants.TerminalConfiguration.DIFF
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai3", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai5", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai6", terminal_config = modo)
        task.ai_channels.add_ai_voltage_chan("Dev1/ai7", terminal_config = modo)
    
        task.timing.cfg_samp_clk_timing(rate= fs, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=int(cant_puntos))
        
        datos = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE, timeout= cant_puntos/fs+0.5)           
    datos = np.asarray(datos)    
    return datos

duracion = 0.1 #segundos
fs = 20000 #Frecuencia de muestreo
cant_puntos = int(duracion*fs)
y = medir(cant_puntos, fs)


Vchanel = np.zeros(7)
for indice in range(7):
    Vchanel[indice] = np.mean(y[indice,:])
plt.plot([1,2,3,4,5,6,7], Vchanel, marker='o', linestyle='')
plt.show()
