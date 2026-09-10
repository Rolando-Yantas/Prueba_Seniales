## **5. Resultados** <a name="id7"></a>

Ahora se analizarán los resultados obtenidos a partir del procesamiento de las señales de electromiografía superficial (EMG). Para cada condición experimental se visualizaron las señales en el dominio temporal y, posteriormente, se realizó el procesamiento necesario para obtener su representación filtrada, rectificada y su comportamiento en frecuencia.

El análisis se realizó considerando dos músculos: el **flexor radial del carpo** y las **fibras descendentes del trapecio**. Para cada uno se evaluaron tres condiciones de actividad muscular: reposo, movimiento leve sin oposición y movimiento fuerte con oposición.

### **5.1. Procesamiento y visualización de las señales EMG con Python**

Para el procesamiento de las señales se utilizaron herramientas de Python orientadas al análisis numérico, filtrado digital y representación gráfica. El procedimiento aplicado permitió observar las características de las señales antes y después del filtrado, así como estudiar su distribución en el dominio de la frecuencia.

### a) Importación de librerías

Inicialmente se incorporaron las librerías necesarias para realizar las operaciones matemáticas, generar las gráficas y aplicar los filtros digitales sobre las señales EMG.

```python
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt, welch, iirnotch 
```

### b) Carga de la señal

Se seleccionó el archivo correspondiente a cada registro EMG y se cargaron los datos almacenados en formato `.txt`. Como ejemplo, se muestra el procedimiento utilizado para importar una de las señales obtenidas durante el registro del **flexor radial del carpo en reposo**.

```python
ruta = r"D:\IB PUCP - UPCH 2022\7mo Semestre\Introducción a las Señales Biomédicas\Lab 3 - Adquisicion EMG\pythoooooon\Flexor radial\Reposo\reposo1_flexor.txt" 
datos = np.loadtxt(ruta, comments="#") 
emg = datos[:, -1]  # señal EMG ubicada en la última columna
```

### c) Definición de parámetros para la representación temporal

Una vez obtenida la señal, se establecieron los parámetros correspondientes a la frecuencia de muestreo y al eje temporal. Estos valores permiten relacionar cada muestra adquirida con el instante de tiempo en el que fue registrada.

```python
fs = 1000  # frecuencia de muestreo
tiempo = np.linspace(0, len(emg) / fs, len(emg))
```

### d) Representación de la señal original

Primero se visualizó la señal sin procesamiento adicional. Esta representación permite observar directamente la amplitud de la actividad eléctrica muscular y reconocer posibles componentes de ruido o perturbaciones presentes durante la adquisición.

```python
plt.figure(figsize=(10, 4)) 
plt.plot(tiempo, emg, color="#00008B", linewidth=0.4) 
plt.title("Señal EMG original - Flexor radial del carpo en reposo") 
plt.xlabel("Tiempo (s)") 
plt.ylabel("Amplitud (uV)") 
plt.grid() 
plt.show()
```

### e) Filtrado de la señal

Con el objetivo de reducir componentes no deseadas de la señal, se aplicó un filtro pasabanda y posteriormente un filtro Notch. El primero permite conservar principalmente el rango de frecuencias de interés de la señal EMG, mientras que el segundo está orientado a reducir la interferencia asociada a la frecuencia de la red eléctrica.

```python
# Función filtro pasa-banda
def filtro_pasabanda(señal, f_muestreo, frec_baja=20.0, frec_alta=450.0, orden=4): 
    nyquist = 0.5 * f_muestreo   
    bajo = frec_baja / nyquist 
    alto = frec_alta / nyquist 
    b, a = butter(orden, [bajo, alto], btype="band") 
    return filtfilt(b, a, señal) 

# Función filtro notch (60 Hz)
def filtro_notch(señal, f_muestreo, frec_notch=60.0, Q=30.0): 
    """
    frec_notch = frecuencia a eliminar (Hz)
    Q = factor de calidad (entre más alto, más selectivo es el notch)
    """
    nyquist = 0.5 * f_muestreo 
    w0 = frec_notch / nyquist 
    b, a = iirnotch(w0, Q) 
    return filtfilt(b, a, señal)

# Aplicación de filtros
emg_filtrada = filtro_pasabanda(emg, fs) 
emg_filtrada = filtro_notch(emg_filtrada, fs, frec_notch=60.0, Q=30.0)
```

### f) Representación de la señal filtrada

Luego del procesamiento se generó una nueva representación temporal de la señal. La comparación entre la señal original y la señal filtrada permite identificar los cambios producidos por el procesamiento y evaluar visualmente la reducción de componentes no deseadas.

```python
plt.figure(figsize=(10, 4)) 
plt.plot(tiempo, emg_filtrada, color="#006400", linewidth=0.4) 
plt.title("Señal EMG filtrada - Flexor radial del carpo en reposo") 
plt.xlabel("Tiempo (s)") 
plt.ylabel("Amplitud (uV)") 
plt.grid() 
plt.show()
```

### g) Obtención de la densidad espectral de potencia mediante Welch

Para estudiar cómo se distribuye la potencia de la señal en función de la frecuencia, se utilizó el método de Welch. Esta representación permite identificar las regiones de frecuencia que presentan una mayor contribución energética en cada condición de actividad muscular.

```python
frecs, psd = welch(emg_filtrada, fs, nperseg=1024) 
plt.figure(figsize=(10, 4)) 
plt.semilogy(frecs, psd, color="purple") 
plt.title("Densidad espectral de potencia (Welch) - Flexor radial del carpo en reposo") 
plt.xlabel("Frecuencia (Hz)") 
plt.ylabel("PSD (uV²/Hz)") 
plt.grid() 
plt.show()
```

### h) Obtención del espectro de frecuencias mediante FFT

Finalmente, se empleó la Transformada Rápida de Fourier (FFT) para representar la amplitud de los componentes frecuenciales presentes en la señal EMG. Esta representación complementa el análisis realizado mediante el método de Welch y facilita la identificación de los componentes predominantes de la señal.

```python
fft_vals = np.fft.rfft(emg_filtrada) 
fft_freqs = np.fft.rfftfreq(len(emg_filtrada), 1/fs) 
plt.figure(figsize=(10, 4)) 
plt.plot(fft_freqs, np.abs(fft_vals), color="darkred", linewidth=0.6) 
plt.title("Espectro de frecuencias (FFT) - Flexor radial del carpo en reposo") 
plt.xlabel("Frecuencia (Hz)") 
plt.ylabel("Amplitud") 
plt.xlim(0, 500) 
plt.grid() 
plt.show()
```

### **5.2. EMG - Flexor radial del carpo**

En esta sección se presentan las gráficas correspondientes al registro del flexor radial del carpo bajo las diferentes condiciones experimentales. Para cada tipo de actividad se incluyen la señal original, la señal filtrada, la densidad espectral de potencia obtenida mediante Welch y el espectro de frecuencias calculado mediante FFT.

| Entrenamiento                   | Señal original             | Señal filtrada             |
| ------------------------------- | -------------------------- | -------------------------- |
| Reposo                          | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento leve sin oposición   | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento acelerado            | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento fuerte con oposición | **Colocar aquí la imagen** | **Colocar aquí la imagen** |

| Entrenamiento                   | Densidad Espectral de Potencia (Welch) | Espectro de Frecuencias (FFT) |
| ------------------------------- | -------------------------------------- | ----------------------------- |
| Reposo                          | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento leve sin oposición   | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento acelerado            | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento fuerte con oposición | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |

### **5.3. EMG - Fibras descendentes del trapecio**

A continuación, se muestran los resultados obtenidos para el registro de las fibras descendentes del trapecio. Al igual que en el caso anterior, se presentan las señales correspondientes a las cuatro condiciones de actividad y sus respectivas representaciones después del procesamiento temporal y frecuencial.

| Entrenamiento                   | Señal original             | Señal filtrada             |
| ------------------------------- | -------------------------- | -------------------------- |
| Reposo                          | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento leve sin oposición   | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento acelerado            | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento fuerte con oposición | **Colocar aquí la imagen** | **Colocar aquí la imagen** |

| Entrenamiento                   | Densidad Espectral de Potencia (Welch) | Espectro de Frecuencias (FFT) |
| ------------------------------- | -------------------------------------- | ----------------------------- |
| Reposo                          | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento leve sin oposición   | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento acelerado            | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento fuerte con oposición | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |

## **6. Análisis y discusión** <a name="id9"></a>

### **6.1. EMG - Flexor radial del carpo** <a name="id9"></a>

**[Análisis y discusión por completar a partir de las gráficas obtenidas.]**

### **6.2. EMG - Fibras descendentes del trapecio** <a name="id9"></a>

**[Análisis y discusión por completar a partir de las gráficas obtenidas.]**
