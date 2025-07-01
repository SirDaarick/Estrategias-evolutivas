# Optimización de Viga Soldada con Estrategias Evolutivas

Este proyecto implementa un algoritmo de **estrategias evolutivas adaptativas** para minimizar el costo de una viga soldada respetando sus restricciones estructurales mediante penalizaciones. El sistema permite probar distintas configuraciones de parámetros y realizar análisis estadístico (ANOVA) para evaluar el impacto de cada combinación.

## Requisitos del sistema

Antes de ejecutar el proyecto, asegúrate de tener:

- **Python 3.8+** (recomendado 3.9 o superior)
- pip actualizado

## Dependencias

Instala las bibliotecas necesarias ejecutando:
```
pip install numpy pandas matplotlib statsmodels
```
**Dependencias principales:**
- `numpy` (operaciones numéricas y generación de aleatorios)
- `pandas` (manejo de datos)
- `matplotlib` (visualización de resultados)
- `statsmodels` (análisis ANOVA)

## Estructura de archivos esperada
```
proyecto/
├── beam_evolutivo.py           # Código con el algoritmo principal
├── ejecuciones.py              # Script que lanza los experimentos y ANOVA
├── generar_resumen.py          # Script que genera la tabla resumen del CSV
├── resultados_beam.csv         # Archivo generado con los resultados brutos
├── resumen_estadistico.csv     # Resumen estadístico generado
└── graficas/                   # Carpeta donde se guardan las gráficas
```

## Cómo usar el proyecto

### Ejecutar los experimentos

Lanza las pruebas con las distintas configuraciones:
```
python ejecuciones.py
```
Esto:
- Ejecuta todas las combinaciones de parámetros predefinidas.
- Guarda los resultados en `resultados_beam.csv`.
- Genera las gráficas de convergencia de la mejor ejecución.
- Imprime el análisis ANOVA.

### Generar el resumen estadístico

Una vez generados los resultados:
```
python generar_resumen.py
```
Esto:
- Lee `resultados_beam.csv`.
- Calcula promedio, mejor, peor y desviación estándar por configuración.
- Guarda `resumen_estadistico.csv`.

## Resultados

- Las gráficas de convergencia se guardan automáticamente en la carpeta `graficas/`.
- Los archivos `.csv` generados permiten análisis posteriores o visualización en herramientas externas como Excel.

## Notas

- Puedes modificar los parámetros a probar dentro de `ejecuciones.py` (valores de mu, alpha, sigma_inicial).
- Asegúrate de crear la carpeta `graficas/` si no existe, o modifica el código para crearla automáticamente.
- El análisis ANOVA permite identificar parámetros con efecto significativo sobre el desempeño del algoritmo.

