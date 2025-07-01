import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#----------------------
# FUNCIÓN OBJETIVO + RESTRICCIONES (Problema de la viga soldada)
#----------------------
def BEAM(x):
    # Costo de fabricación (función a minimizar)
    y = 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    # Parámetros físicos y límites del problema
    P = 6000         # Carga aplicada (lb)
    L = 14           # Longitud (pulgadas)
    E = 30e6         # Módulo de elasticidad (psi)
    G = 12e6         # Módulo de corte (psi)
    t_max = 13600    # Límite de tensión de corte (psi)
    s_max = 30000    # Límite de tensión (psi)
    d_max = 0.25     # Deflexión máxima permitida (pulgadas)

    # Cálculos intermedios mecánicos
    M = P * (L + x[1] / 2)  # Momento flector
    R = sqrt(0.25 * (x[1]**2 + (x[0] + x[2])**2))  # Radio de curvatura
    J = 2 * (sqrt(2) * x[0] * x[1] * (x[1]**2 / 12 + 0.25 * (x[0] + x[2])**2))  # Momento polar de inercia
    P_c = (4.013 * E / (6 * L**2)) * x[2] * x[3]**3 * (1 - 0.25 * x[2] * sqrt(E / G) / L)  # Carga crítica por pandeo

    # Cálculo de tensiones y deflexión
    t1 = P / (sqrt(2) * x[0] * x[1])  # Tensión de corte directa
    t2 = M * R / J                   # Tensión de corte por flexión
    t = sqrt(t1**2 + t1 * t2 * x[1] / R + t2**2)  # Tensión total
    s = 6 * P * L / (x[3] * x[2]**2)              # Tensión normal
    d = 4 * P * L**3 / (E * x[3] * x[2]**3)        # Deflexión

    # Restricciones (deben ser <= 0 para estar dentro de los límites)
    g = [
        t - t_max,  # Tensión de corte máxima
        s - s_max,  # Tensión normal máxima
        x[0] - x[3],  # Ancho de soldadura menor que el espesor del refuerzo
        y - 5.0,     # Volumen/costo no mayor a 5 in³
        0.125 - x[0],  # Ancho mínimo de soldadura
        d - d_max,    # Deflexión máxima
        P - P_c       # Carga aplicada menor que carga crítica
    ]

    g_round = np.round(np.array(g), 6)  # Redondea restricciones para estabilidad numérica
    w1 = 100  # Peso para penalización por magnitud
    w2 = 100  # Peso para penalización por cantidad

    # Penalizaciones
    phi = sum(max(gi, 0) for gi in g_round)      # Suma de las violaciones positivas
    viol = sum(gi > 0 for gi in g_round)         # Número de restricciones violadas

    # Función objetivo penalizada
    reward = y + w1 * phi + w2 * viol
    return reward

#----------------------
# INICIALIZACIÓN DE POBLACIÓN
#----------------------
def inicializar_poblacion(mu, sigma_inicial):
    poblacion = []
    for _ in range(mu):
        individuo = {
            'x': np.random.uniform([0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0]),  # Genera vector dentro de los límites
            'sigma': sigma_inicial
        }
        poblacion.append(individuo)
    return poblacion

#----------------------
# MUTACIÓN CORRELACIONADA
#----------------------
def mutar_correlacionada(padre, sigma_global, cov_matrix):
    x = padre['x'].copy()
    ruido = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=cov_matrix)  # Genera ruido correlacionado
    x_nuevo = x + sigma_global * ruido  # Aplica mutación global
    x_nuevo = np.clip(x_nuevo, [0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0])  # Asegura que esté en los rangos válidos
    return {'x': x_nuevo, 'sigma': sigma_global}

#----------------------
# SELECCIÓN (μ de λ)
#----------------------
def seleccion(hijos, mu):
    puntuados = [(ind, BEAM(ind['x'])) for ind in hijos]  # Evalúa la función objetivo para cada hijo
    puntuados.sort(key=lambda x: x[1])  # Ordena por aptitud (menor es mejor)
    seleccionados = [ind for ind, _ in puntuados[:mu]]  # Elige los mejores μ
    mejor = puntuados[0]
    peor = puntuados[-1]
    return seleccionados, mejor, peor

#----------------------
# ESTRATEGIAS EVOLUTIVAS (Adaptativas con mutación correlacionada)
#----------------------
def estrategias_evolutivas(mu=30, lambd=150, generaciones=500, sigma_inicial=1.5, alpha=1.15, exploracion_generaciones=15, retornar_historial=False, guardar_grafica=False, config_name=""):
    dim = 4  # Dimensión del vector de diseño
    poblacion = inicializar_poblacion(mu, sigma_inicial)
    sigma_global = sigma_inicial
    historial_mejor_aptitud = []
    historial_sigma = []
    sin_mejora = 0
    mejor_aptitud_global = float('inf')
    mejor_solucion_global = None
    cov_matrix = np.identity(dim) * 0.5  # Matriz de covarianza inicial

    for gen in range(generaciones):
        hijos = []
        exitosas = 0

        # Recalcula la matriz de covarianza cada 20 generaciones
        if gen % 20 == 0 and gen > 0:
            muestras = np.array([ind['x'] for ind in poblacion])
            if len(muestras) > 1:
                nueva_cov = np.cov(muestras.T)
                regularizacion = max(0.1 * sigma_global**2, 0.01)
                nueva_cov += np.identity(dim) * regularizacion  # Evita singularidad
                cov_matrix = nueva_cov

        for _ in range(lambd):
            padre = poblacion[np.random.randint(0, mu)]
            hijo = mutar_correlacionada(padre, sigma_global, cov_matrix)
            hijos.append(hijo)

            # Evalúa si el hijo mejoró al padre (para calcular tasa de éxito)
            if BEAM(hijo['x']) < BEAM(padre['x']):
                exitosas += 1

        # Adaptación de sigma con base en tasa de éxito
        tasa_exito = exitosas / lambd
        if tasa_exito > 0.2:
            sigma_global *= alpha
        elif tasa_exito < 0.2:
            sigma_global /= alpha
        sigma_global = np.clip(sigma_global, 0.05, 3.0)  # Limita el rango de sigma

        # Selección de los mejores individuos
        poblacion, mejor, peor = seleccion(hijos, mu)

        for ind in poblacion:
            ind['sigma'] = sigma_global  # Actualiza sigma en todos

        mejor_individuo, mejor_aptitud = mejor
        peor_individuo, peor_aptitud = peor

        # Actualiza mejor global si mejora
        if mejor_aptitud < mejor_aptitud_global:
            mejor_aptitud_global = mejor_aptitud
            mejor_solucion_global = mejor_individuo['x'].copy()
            sin_mejora = 0
        else:
            sin_mejora += 1

        # Si no mejora tras varias generaciones, explora
        if sin_mejora >= exploracion_generaciones:
            ruido = np.random.uniform(1.2, 2.5)
            sigma_global *= ruido
            sigma_global = min(sigma_global, 3.0)

            porcentaje = 0.3  # Reemplaza 30% de la población
            num_reemplazos = int(mu * porcentaje)
            nuevos = inicializar_poblacion(num_reemplazos, sigma_global)
            poblacion[-num_reemplazos:] = nuevos
            sin_mejora = 0
            print(f"[Gen {gen}] Exploración + Reemplazo parcial: σ = {sigma_global:.4f}")

        historial_mejor_aptitud.append(mejor_aptitud)
        historial_sigma.append(sigma_global)

        if gen % 50 == 0 or gen == generaciones - 1:
            print(f"Gen {gen:4d}: Mejor={mejor_aptitud:.6f}, σ={sigma_global:.4f}, Tasa_éxito={tasa_exito:.3f}")

        # Criterio de paro por estancamiento (si todos tienen mismo valor)
        if abs(peor_aptitud - mejor_aptitud) < 1e-6:
            print(f"\nCriterio de paro alcanzado en generación {gen}: Mejor ≈ Peor")
            break

    # Muestra los resultados finales
    print("\n🎯 RESULTADO FINAL:")
    print(f"Mejor aptitud: {mejor_aptitud_global:.8f}")
    print(f"Mejor solución: {mejor_solucion_global}")

    # Guardar gráficos de evolución si se desea
    if guardar_grafica:
        carpeta = "graficas_resultados"
        os.makedirs(carpeta, exist_ok=True)

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(historial_mejor_aptitud, 'b-')
        plt.title('Convergencia')
        plt.xlabel('Generación')
        plt.ylabel('Mejor aptitud')

        plt.subplot(1, 2, 2)
        plt.plot(historial_sigma, 'r-')
        plt.title('Evolución de σ')
        plt.xlabel('Generación')
        plt.ylabel('σ')

        plt.tight_layout()
        archivo = os.path.join(carpeta, f"convergencia_{config_name}.png")
        plt.savefig(archivo)
        plt.close()
        print(f"Gráfica guardada en {archivo}")

    if retornar_historial:
        return mejor_solucion_global, mejor_aptitud_global, historial_mejor_aptitud
    else:
        return mejor_solucion_global, mejor_aptitud_global

#----------------------
# EJECUCIÓN DEL ALGORITMO
#----------------------
if __name__ == "__main__":
    mejor_x, mejor_f = estrategias_evolutivas()  # Ejecuta el algoritmo con parámetros por defecto
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

#----------------------
# FUNCIÓN OBJETIVO + RESTRICCIONES (Problema de la viga soldada)
#----------------------
def BEAM(x):
    # Costo de fabricación (función a minimizar)
    y = 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    # Parámetros físicos y límites del problema
    P = 6000         # Carga aplicada (lb)
    L = 14           # Longitud (pulgadas)
    E = 30e6         # Módulo de elasticidad (psi)
    G = 12e6         # Módulo de corte (psi)
    t_max = 13600    # Límite de tensión de corte (psi)
    s_max = 30000    # Límite de tensión (psi)
    d_max = 0.25     # Deflexión máxima permitida (pulgadas)

    # Cálculos intermedios mecánicos
    M = P * (L + x[1] / 2)  # Momento flector
    R = sqrt(0.25 * (x[1]**2 + (x[0] + x[2])**2))  # Radio de curvatura
    J = 2 * (sqrt(2) * x[0] * x[1] * (x[1]**2 / 12 + 0.25 * (x[0] + x[2])**2))  # Momento polar de inercia
    P_c = (4.013 * E / (6 * L**2)) * x[2] * x[3]**3 * (1 - 0.25 * x[2] * sqrt(E / G) / L)  # Carga crítica por pandeo

    # Cálculo de tensiones y deflexión
    t1 = P / (sqrt(2) * x[0] * x[1])  # Tensión de corte directa
    t2 = M * R / J                   # Tensión de corte por flexión
    t = sqrt(t1**2 + t1 * t2 * x[1] / R + t2**2)  # Tensión total
    s = 6 * P * L / (x[3] * x[2]**2)              # Tensión normal
    d = 4 * P * L**3 / (E * x[3] * x[2]**3)        # Deflexión

    # Restricciones (deben ser <= 0 para estar dentro de los límites)
    g = [
        t - t_max,  # Tensión de corte máxima
        s - s_max,  # Tensión normal máxima
        x[0] - x[3],  # Ancho de soldadura menor que el espesor del refuerzo
        y - 5.0,     # Volumen/costo no mayor a 5 in³
        0.125 - x[0],  # Ancho mínimo de soldadura
        d - d_max,    # Deflexión máxima
        P - P_c       # Carga aplicada menor que carga crítica
    ]

    g_round = np.round(np.array(g), 6)  # Redondea restricciones para estabilidad numérica
    w1 = 100  # Peso para penalización por magnitud
    w2 = 100  # Peso para penalización por cantidad

    # Penalizaciones
    phi = sum(max(gi, 0) for gi in g_round)      # Suma de las violaciones positivas
    viol = sum(gi > 0 for gi in g_round)         # Número de restricciones violadas

    # Función objetivo penalizada
    reward = y + w1 * phi + w2 * viol
    return reward

#----------------------
# INICIALIZACIÓN DE POBLACIÓN
#----------------------
def inicializar_poblacion(mu, sigma_inicial):
    poblacion = []
    for _ in range(mu):
        individuo = {
            'x': np.random.uniform([0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0]),  # Genera vector dentro de los límites
            'sigma': sigma_inicial
        }
        poblacion.append(individuo)
    return poblacion

#----------------------
# MUTACIÓN CORRELACIONADA
#----------------------
def mutar_correlacionada(padre, sigma_global, cov_matrix):
    x = padre['x'].copy()
    ruido = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=cov_matrix)  # Genera ruido correlacionado
    x_nuevo = x + sigma_global * ruido  # Aplica mutación global
    x_nuevo = np.clip(x_nuevo, [0.1, 0.1, 0.1, 0.1], [2.0, 10.0, 10.0, 2.0])  # Asegura que esté en los rangos válidos
    return {'x': x_nuevo, 'sigma': sigma_global}

#----------------------
# SELECCIÓN (μ de λ)
#----------------------
def seleccion(hijos, mu):
    puntuados = [(ind, BEAM(ind['x'])) for ind in hijos]  # Evalúa la función objetivo para cada hijo
    puntuados.sort(key=lambda x: x[1])  # Ordena por aptitud (menor es mejor)
    seleccionados = [ind for ind, _ in puntuados[:mu]]  # Elige los mejores μ
    mejor = puntuados[0]
    peor = puntuados[-1]
    return seleccionados, mejor, peor

#----------------------
# ESTRATEGIAS EVOLUTIVAS (Adaptativas con mutación correlacionada)
#----------------------
def estrategias_evolutivas(mu=30, lambd=150, generaciones=500, sigma_inicial=1.5, alpha=1.15, exploracion_generaciones=15, retornar_historial=False, guardar_grafica=False, config_name=""):
    dim = 4  # Dimensión del vector de diseño
    poblacion = inicializar_poblacion(mu, sigma_inicial)
    sigma_global = sigma_inicial
    historial_mejor_aptitud = []
    historial_sigma = []
    sin_mejora = 0
    mejor_aptitud_global = float('inf')
    mejor_solucion_global = None
    cov_matrix = np.identity(dim) * 0.5  # Matriz de covarianza inicial

    for gen in range(generaciones):
        hijos = []
        exitosas = 0

        # Recalcula la matriz de covarianza cada 20 generaciones
        if gen % 20 == 0 and gen > 0:
            muestras = np.array([ind['x'] for ind in poblacion])
            if len(muestras) > 1:
                nueva_cov = np.cov(muestras.T)
                regularizacion = max(0.1 * sigma_global**2, 0.01)
                nueva_cov += np.identity(dim) * regularizacion  # Evita singularidad
                cov_matrix = nueva_cov

        for _ in range(lambd):
            padre = poblacion[np.random.randint(0, mu)]
            hijo = mutar_correlacionada(padre, sigma_global, cov_matrix)
            hijos.append(hijo)

            # Evalúa si el hijo mejoró al padre (para calcular tasa de éxito)
            if BEAM(hijo['x']) < BEAM(padre['x']):
                exitosas += 1

        # Adaptación de sigma con base en tasa de éxito
        tasa_exito = exitosas / lambd
        if tasa_exito > 0.2:
            sigma_global *= alpha
        elif tasa_exito < 0.2:
            sigma_global /= alpha
        sigma_global = np.clip(sigma_global, 0.05, 3.0)  # Limita el rango de sigma

        # Selección de los mejores individuos
        poblacion, mejor, peor = seleccion(hijos, mu)

        for ind in poblacion:
            ind['sigma'] = sigma_global  # Actualiza sigma en todos

        mejor_individuo, mejor_aptitud = mejor
        peor_individuo, peor_aptitud = peor

        # Actualiza mejor global si mejora
        if mejor_aptitud < mejor_aptitud_global:
            mejor_aptitud_global = mejor_aptitud
            mejor_solucion_global = mejor_individuo['x'].copy()
            sin_mejora = 0
        else:
            sin_mejora += 1

        # Si no mejora tras varias generaciones, explora
        if sin_mejora >= exploracion_generaciones:
            ruido = np.random.uniform(1.2, 2.5)
            sigma_global *= ruido
            sigma_global = min(sigma_global, 3.0)

            porcentaje = 0.3  # Reemplaza 30% de la población
            num_reemplazos = int(mu * porcentaje)
            nuevos = inicializar_poblacion(num_reemplazos, sigma_global)
            poblacion[-num_reemplazos:] = nuevos
            sin_mejora = 0
            print(f"[Gen {gen}] Exploración + Reemplazo parcial: σ = {sigma_global:.4f}")

        historial_mejor_aptitud.append(mejor_aptitud)
        historial_sigma.append(sigma_global)

        if gen % 50 == 0 or gen == generaciones - 1:
            print(f"Gen {gen:4d}: Mejor={mejor_aptitud:.6f}, σ={sigma_global:.4f}, Tasa_éxito={tasa_exito:.3f}")

        # Criterio de paro por estancamiento (si todos tienen mismo valor)
        if abs(peor_aptitud - mejor_aptitud) < 1e-6:
            print(f"\nCriterio de paro alcanzado en generación {gen}: Mejor ≈ Peor")
            break

    # Muestra los resultados finales
    print("\n🎯 RESULTADO FINAL:")
    print(f"Mejor aptitud: {mejor_aptitud_global:.8f}")
    print(f"Mejor solución: {mejor_solucion_global}")

    # Guardar gráficos de evolución si se desea
    if guardar_grafica:
        carpeta = "graficas_resultados"
        os.makedirs(carpeta, exist_ok=True)

        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(historial_mejor_aptitud, 'b-')
        plt.title('Convergencia')
        plt.xlabel('Generación')
        plt.ylabel('Mejor aptitud')

        plt.subplot(1, 2, 2)
        plt.plot(historial_sigma, 'r-')
        plt.title('Evolución de σ')
        plt.xlabel('Generación')
        plt.ylabel('σ')

        plt.tight_layout()
        archivo = os.path.join(carpeta, f"convergencia_{config_name}.png")
        plt.savefig(archivo)
        plt.close()
        print(f"Gráfica guardada en {archivo}")

    if retornar_historial:
        return mejor_solucion_global, mejor_aptitud_global, historial_mejor_aptitud
    else:
        return mejor_solucion_global, mejor_aptitud_global

#----------------------
# EJECUCIÓN DEL ALGORITMO
#----------------------
if __name__ == "__main__":
    mejor_x, mejor_f = estrategias_evolutivas()  # Ejecuta el algoritmo con parámetros por defecto
