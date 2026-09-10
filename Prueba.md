<img src="image/cuello_reposo_welch.png" width="120" alt="cuello_reposo_welch">

## **5. Resultados** <a name="id7"></a>

Para la presentación de los resultados, emplearemos el script de Jupyter denominado "CodigosEMG.ipynb" que se ubica en el directorio del proyecto. Este archivo engloba los algoritmos diseñados para la representación gráfica de las señales biomédicas registradas, abarcando las etapas de filtrado digital y la evaluación en el dominio de la frecuencia.

### **5.1. Procesamiento y visualización de gráficas con python**
### a) Importación de librerías
En primer lugar, se importan los módulos y bibliotecas necesarios que se aplicarán de forma general para el tratamiento de todas las señales capturadas.
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, iirnotch
```
### b) Lectura de los datos
Se procede a cargar el documento .txt desde la ruta específica del equipo local (A continuación, se ilustra el procedimiento utilizando como referencia la señal del músculo Flexor radial del carpo en estado de reposo).
```python
ruta = r"D:\IB PUCP - UPCH 2022 mo Semestre\Introducción a las Señales Biomédicas\Lab 3 - Adquisicion EMG\pythoooooon\Flexor_Radial\Reposo
eposo1_flexor.txt"
datos = np.loadtxt(ruta, comments="#")
emg_signal = datos[:, -1]  # La señal EMG se ubica en la última columna del arreglo
```
### c) Definición de parámetros temporales
```python
f_samp = 1000  # Frecuencia de muestreo en Hz
vector_tiempo = np.linspace(0, len(emg_signal) / f_samp, len(emg_signal))
```
### d) Ploteo de la señal cruda
```python
plt.figure(figsize=(10, 4))
plt.plot(vector_tiempo, emg_signal, color="#1f77b4", linewidth=0.4)
plt.title("Señal EMG cruda - Flexor radial del carpo (Reposo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (uV)")
plt.grid(True)
plt.show()
```
### e) Etapa de filtrado
```python
# Diseño del filtro pasa-banda
def aplicar_pasabanda(señal, fm, f_min=20.0, f_max=450.0, orden_filtro=4):
    frec_nyquist = 0.5 * fm  
    lim_inf = f_min / frec_nyquist
    lim_sup = f_max / frec_nyquist
    b, a = butter(orden_filtro, [lim_inf, lim_sup], btype="band")
    return filtfilt(b, a, señal)

# Diseño del filtro rechaza-banda (Notch a 60 Hz)
def aplicar_notch(señal, fm, frec_interferencia=60.0, factor_Q=30.0):
    """
    frec_interferencia = Frecuencia específica a atenuar (Hz)
    factor_Q = Factor de calidad para determinar la selectividad del filtro
    """
    frec_nyquist = 0.5 * fm
    w0 = frec_interferencia / frec_nyquist
    b, a = iirnotch(w0, factor_Q)
    return filtfilt(b, a, señal)

# Ejecución del filtrado sobre la señal original
señal_filtrada = aplicar_pasabanda(emg_signal, f_samp)
señal_filtrada = aplicar_notch(señal_filtrada, f_samp, frec_interferencia=60.0, factor_Q=30.0)
```
### f) Ploteo de la señal procesada
```python
plt.figure(figsize=(10, 4))
plt.plot(vector_tiempo, señal_filtrada, color="#2ca02c", linewidth=0.4)
plt.title("Señal EMG procesada - Flexor radial del carpo (Reposo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (uV)")
plt.grid(True)
plt.show()
```
### g) Estimación de la Densidad Espectral (Método de Welch)
```python
frecuencias, pot_espectral = welch(señal_filtrada, f_samp, nperseg=1024)
plt.figure(figsize=(10, 4))
plt.semilogy(frecuencias, pot_espectral, color="#9467bd")
plt.title("Densidad Espectral de Potencia (Welch) - Flexor radial del carpo (Reposo)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (uV²/Hz)")
plt.grid(True)
plt.show()
```
### h) Transformada Rápida de Fourier (Espectro de Frecuencias)
```python
valores_fft = np.fft.rfft(señal_filtrada)
frecuencias_fft = np.fft.rfftfreq(len(señal_filtrada), 1/f_samp)
plt.figure(figsize=(10, 4))
plt.plot(frecuencias_fft, np.abs(valores_fft), color="#d62728", linewidth=0.6)
plt.title("Espectro de Magnitud (FFT) - Flexor radial del carpo (Reposo)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.xlim(0, 500)
plt.grid(True)
plt.show()
```

### **5.2. EMG - Flexor radial del carpo**

| Tipo de contracción / Ejercicio                  | Señal original | Señal filtrada                         |
|----------------------------------------------|----------|---------------------------------|
| Reposo                              | <img src="image/cuello_reposo_welch.png" width="120" alt="cuello_reposo_welch"> | <img src="image/cuello_reposo_welch.png" width="120" alt="cuello_reposo_welch"> |
| Movimiento leve sin oposición                        | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento acelerado       | colocar aqui la imagen        | colocar aqui la imagen |
| Movimiento fuerte con oposición                   | colocar aqui la imagen       | colocar aqui la imagen |

| Tipo de contracción / Ejercicio                  | Densidad Espectral de Potencia (Welch) | Espectro de Frecuencias (FFT)                         |
|----------------------------------------------|----------|---------------------------------|
| Reposo                              | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento leve sin oposición                        | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento acelerado       | colocar aqui la imagen        | colocar aqui la imagen |
| Movimiento fuerte con oposición                   | colocar aqui la imagen       | colocar aqui la imagen |

### **5.3. EMG - Fibras descendentes del trapecio**

| Tipo de contracción / Ejercicio                  | Señal original | Señal filtrada                         |
|----------------------------------------------|----------|---------------------------------|
| Reposo                              | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento leve sin oposición                        | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento acelerado       | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento fuerte con oposición                   | colocar aqui la imagen | colocar aqui la imagen |

| Tipo de contracción / Ejercicio                  | Densidad Espectral de Potencia (Welch) | Espectro de Frecuencias (FFT)                         |
|----------------------------------------------|----------|---------------------------------|
| Reposo                              | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento leve sin oposición                        | colocar aqui la imagen | colocar aqui la imagen |
| Movimiento acelerado       | colocar aqui la imagen        | colocar aqui la imagen |
| Movimiento fuerte con oposición                   | colocar aqui la imagen       | colocar aqui la imagen |

## **6. Análisis y discusión** <a name="id9"></a>

*[Esta sección se completará posteriormente, una vez que se hayan generado e insertado las gráficas correspondientes de las señales biomédicas]*

### **6.1. EMG - Flexor radial del carpo** <a name="id10"></a>

*[Espacio reservado para el análisis]*

### **6.2. EMG - Fibras descendentes del trapecio** <a name="id11"></a>

*[Espacio reservado para el análisis]*
