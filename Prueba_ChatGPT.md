## **5. Resultados**

En esta sección se presentan los resultados obtenidos a partir de las señales de electromiografía superficial (EMG) registradas durante las diferentes condiciones de actividad muscular. El procesamiento permitió visualizar las señales tanto en el dominio temporal como en el dominio frecuencial, con el propósito de facilitar la comparación de la actividad eléctrica muscular entre las distintas condiciones evaluadas.

Las señales fueron procesadas mediante una serie de etapas que incluyeron la eliminación de la componente continua, filtrado pasabanda, rectificación de la señal y análisis frecuencial.

El procedimiento se aplicó a los registros obtenidos para el **flexor radial del carpo** y las **fibras descendentes del trapecio**, considerando las condiciones de reposo, movimiento leve sin oposición y movimiento fuerte con oposición.

### **5.1. Procesamiento y visualización de las señales EMG**

Para el procesamiento de las señales se emplearon herramientas de Python para el manejo de los datos, filtrado digital, análisis espectral y generación de las gráficas. A partir de los registros obtenidos durante la adquisición, se realizó el procesamiento de cada señal siguiendo una secuencia común para todas las condiciones experimentales.

### a) Importación de librerías

En primer lugar, se cargaron las librerías necesarias para realizar las operaciones matemáticas, el procesamiento de señales y la generación de las representaciones gráficas.

```python
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, iirnotch
```

### b) Selección y ubicación del registro

Para cada análisis se seleccionó el archivo correspondiente a la señal EMG que se deseaba procesar.

```python
from pathlib import Path

carpeta = Path(__file__).resolve().parent

ruta = carpeta / "basalflexocarpiradiales1.txt"

carpeta_resultados = carpeta / "resultados"
carpeta_resultados.mkdir(exist_ok=True)

if not ruta.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo:\n{ruta}\n"
        "Verifica que el nombre esté escrito correctamente."
    )
```


### c) Definición de los filtros

Para reducir las componentes no deseadas presentes en las señales se implementaron dos tipos de filtros. El primero corresponde a un filtro pasabanda entre **20 y 450 Hz**, utilizado para conservar el intervalo de frecuencias de interés de la señal EMG. Posteriormente, se aplicó un filtro Notch centrado en **60 Hz**, destinado a reducir la interferencia proveniente de la red eléctrica.

```python
def filtro_pasabanda(
    señal,
    frecuencia_muestreo,
    frecuencia_baja=20.0,
    frecuencia_alta=450.0,
    orden=4
):
    nyquist = frecuencia_muestreo / 2

    frecuencia_baja_normalizada = frecuencia_baja / nyquist
    frecuencia_alta_normalizada = frecuencia_alta / nyquist

    b, a = butter(
        orden,
        [frecuencia_baja_normalizada, frecuencia_alta_normalizada],
        btype="bandpass"
    )

    return filtfilt(b, a, señal)


def filtro_notch(
    señal,
    frecuencia_muestreo,
    frecuencia_notch=60.0,
    Q=30.0
):
    nyquist = frecuencia_muestreo / 2
    frecuencia_normalizada = frecuencia_notch / nyquist

    b, a = iirnotch(frecuencia_normalizada, Q)

    return filtfilt(b, a, señal)
```

### d) Carga y caracterización inicial de la señal

Los datos almacenados en el archivo fueron cargados mediante `numpy`. Debido a que la señal EMG se encuentra en la última columna del archivo generado durante la adquisición, esta columna fue seleccionada para realizar el procesamiento.

La frecuencia de muestreo utilizada fue de **1000 Hz**, a partir de la cual se construyó el vector temporal correspondiente al registro.

```python
datos = np.loadtxt(ruta, comments="#")

print("Dimensiones del archivo:", datos.shape)
print("Número de columnas:", datos.shape[1])

emg = datos[:, -1].astype(float)

fs = 1000

tiempo = np.arange(len(emg)) / fs

print("Número de muestras:", len(emg))
print("Duración:", round(len(emg) / fs, 2), "segundos")
print("Valor mínimo:", np.min(emg))
print("Valor máximo:", np.max(emg))
```

Además de obtener la señal, se verificaron características básicas del registro, como el número de muestras, duración, valor mínimo y valor máximo.

### e) Eliminación de la componente continua y filtrado

Antes de aplicar los filtros, se eliminó la componente continua de la señal mediante la resta de su valor medio. Posteriormente, se aplicó el filtro pasabanda de 20 a 450 Hz y, finalmente, el filtro Notch de 60 Hz.

```python
emg_sin_media = emg - np.mean(emg)

emg_filtrada = filtro_pasabanda(emg_sin_media, fs)

emg_filtrada = filtro_notch(
    emg_filtrada,
    fs,
    frecuencia_notch=60.0,
    Q=30.0
)
```

Con este procesamiento se obtuvo una señal con menor presencia de componentes continuas e interferencias fuera del rango de interés.

### f) Rectificación de la señal

Después del filtrado se realizó la rectificación de la señal EMG mediante el valor absoluto de cada muestra. Esta transformación permite expresar todas las variaciones de la señal en valores positivos, facilitando la visualización de la magnitud de la actividad muscular.

```python
emg_rectificada = np.abs(emg_filtrada)
```

### g) Representación de la señal original

La primera representación corresponde a la señal EMG directamente obtenida durante la adquisición, sin aplicar el procesamiento descrito anteriormente.

```python
plt.figure(figsize=(12, 4))
plt.plot(tiempo, emg, color="#00008B", linewidth=0.5)
plt.title("Señal EMG original – Flexor radial del carpo en reposo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Valor digital (ADC)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    carpeta_resultados / "antebrazo_reposo_original.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

### h) Representación de la señal filtrada

A continuación, se generó la representación temporal de la señal después de eliminar la componente continua y aplicar los filtros pasabanda y Notch.

```python
plt.figure(figsize=(12, 4))
plt.plot(tiempo, emg_filtrada, color="#006400", linewidth=0.5)
plt.title("Señal EMG filtrada – Flexor radial del carpo en reposo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud filtrada (ADC)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    carpeta_resultados / "antebrazo_reposo_filtrada.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

### i) Representación de la señal rectificada

La señal filtrada también fue rectificada mediante el cálculo de su valor absoluto. La gráfica resultante permite visualizar la magnitud de la actividad eléctrica sin considerar la polaridad de la señal.

```python
plt.figure(figsize=(12, 4))
plt.plot(tiempo, emg_rectificada, color="#D97706", linewidth=0.5)
plt.title("Señal EMG rectificada – Flexor radial del carpo en reposo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud absoluta (ADC)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    carpeta_resultados / "antebrazo_reposo_rectificada.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

### j) Densidad espectral de potencia mediante Welch

Para evaluar la distribución de la potencia de la señal en función de la frecuencia se utilizó el método de Welch. El tamaño del segmento se estableció como el menor valor entre 1024 muestras y la cantidad total de muestras disponibles.

```python
segmento = min(1024, len(emg_filtrada))

frecuencias, psd = welch(
    emg_filtrada,
    fs=fs,
    nperseg=segmento
)

plt.figure(figsize=(12, 4))
plt.semilogy(frecuencias, psd, color="purple")
plt.title(
    "Densidad espectral de potencia – "
    "Flexor radial del carpo en reposo"
)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("PSD (ADC²/Hz)")
plt.xlim(0, 500)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    carpeta_resultados / "antebrazo_reposo_welch.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

### k) Espectro de frecuencias mediante FFT

Finalmente, se obtuvo el espectro de frecuencias utilizando la Transformada Rápida de Fourier (FFT). En este caso, la amplitud fue normalizada de acuerdo con el número total de muestras para facilitar la interpretación de los componentes frecuenciales.

```python
numero_muestras = len(emg_filtrada)

fft_valores = np.fft.rfft(emg_filtrada)
fft_frecuencias = np.fft.rfftfreq(
    numero_muestras,
    d=1/fs
)

amplitud_fft = (
    2 / numero_muestras
) * np.abs(fft_valores)

plt.figure(figsize=(12, 4))
plt.plot(
    fft_frecuencias,
    amplitud_fft,
    color="darkred",
    linewidth=0.7
)

plt.title(
    "Espectro de frecuencias – "
    "Flexor radial del carpo en reposo"
)

plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.xlim(0, 500)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    carpeta_resultados / "antebrazo_reposo_fft.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```
---

### **5.2. EMG - Flexor radial del carpo**
En esta sección se presentan las gráficas correspondientes al registro del **flexor radial del carpo** para las tres condiciones evaluadas.

| Condición                       | Señal original             | Señal filtrada             | Señal rectificada          |
| ------------------------------- | -------------------------- | -------------------------- | -------------------------- |
| Reposo                          | **Colocar aquí la imagen** | **Colocar aquí la imagen** | **Colocar aquí la imagen** | 
| Movimiento leve sin oposición   | **Colocar aquí la imagen** | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento fuerte con oposición | **Colocar aquí la imagen** | **Colocar aquí la imagen** | **Colocar aquí la imagen** |

| Condición                       | Densidad Espectral de Potencia (Welch) | Espectro de Frecuencias (FFT) |
| ------------------------------- | -------------------------------------- | ----------------------------- |
| Reposo                          | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento leve sin oposición   | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento fuerte con oposición | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |


### **5.3. EMG - Fibras descendentes del trapecio**

En esta sección se presentan los resultados correspondientes a las **fibras descendentes del trapecio**. Se siguió el mismo procedimiento de procesamiento utilizado para el flexor radial del carpo.

| Condición                       | Señal original             | Señal filtrada             | Señal rectificada          |
| ------------------------------- | -------------------------- | -------------------------- | -------------------------- |
| Reposo                          | **Colocar aquí la imagen** | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento leve sin oposición   | **Colocar aquí la imagen** | **Colocar aquí la imagen** | **Colocar aquí la imagen** |
| Movimiento fuerte con oposición | **Colocar aquí la imagen** | **Colocar aquí la imagen** | **Colocar aquí la imagen** |

| Condición                       | Densidad Espectral de Potencia (Welch) | Espectro de Frecuencias (FFT) |
| ------------------------------- | -------------------------------------- | ----------------------------- |
| Reposo                          | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento leve sin oposición   | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |
| Movimiento fuerte con oposición | **Colocar aquí la imagen**             | **Colocar aquí la imagen**    |


## **6. Análisis y discusión**

### **6.1. EMG - Flexor radial del carpo**

**[Análisis y discusión por completar a partir de las gráficas y los parámetros obtenidos.]**

### **6.2. EMG - Fibras descendentes del trapecio**

**[Análisis y discusión por completar a partir de las gráficas y los parámetros obtenidos.]**
