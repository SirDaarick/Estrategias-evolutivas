import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from estrategias import estrategias_evolutivas  # Función que ejecuta la estrategia evolutiva personalizada
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

def ejecutar_experimentos():
    # Parámetros a evaluar en los experimentos
    mus = [30, 50, 100, 150]               # Cantidad de padres (μ)
    alphas = [1.05, 1.10, 1.15, 1.20]      # Factor de adaptación de sigma
    sigmas = [0.05, 0.5, 1.5, 3.0]         # Sigma inicial (desviación estándar inicial)
    n_replicas = 10                        # Réplicas por configuración para obtener medias más estables

    resultados = []                        # Lista para guardar resultados de todas las ejecuciones
    mejor_convergencia = None             # Historial de aptitud de la mejor ejecución
    mejor_aptitud = float('inf')          # Inicialmente se asume infinita aptitud (minimización)
    mejor_x = None                         # Solución de la mejor ejecución
    mejor_config = None                   # Configuración de parámetros de la mejor ejecución

    # Carpeta donde se guardarán las gráficas generadas
    carpeta_graficas = "graficas_resultados"
    os.makedirs(carpeta_graficas, exist_ok=True)

    # Se prueban todas las combinaciones de mu, alpha, sigma y sus réplicas
    for mu in mus:
        for alpha in alphas:
            for sigma_inicial in sigmas:
                for replica in range(n_replicas):
                    # Fijar semilla para reproducibilidad
                    seed = replica + 100
                    np.random.seed(seed)

                    # Nombre único para esta configuración (usado al guardar gráficas)
                    config_name = f"mu{mu}_alpha{alpha}_sigma{sigma_inicial}_rep{replica}"

                    # Ejecutar el algoritmo evolutivo con los parámetros actuales
                    mejor_x_tmp, aptitud, hist_aptitud = estrategias_evolutivas(
                        mu=mu,
                        lambd=mu*5,                         # λ se define como 5 veces μ
                        generaciones=500,                   # Número total de generaciones
                        sigma_inicial=sigma_inicial,
                        alpha=alpha,
                        exploracion_generaciones=15,        # Cuántas generaciones esperar para adaptar sigma
                        retornar_historial=True,            # Para graficar la convergencia
                        guardar_grafica=True,               # Guardar gráfico individual
                        config_name=config_name
                    )

                    # Guardar resultados de esta ejecución
                    resultados.append({
                        'mu': mu,
                        'alpha': alpha,
                        'sigma_inicial': sigma_inicial,
                        'replica': replica,
                        'aptitud': aptitud
                    })

                    # Si es la mejor aptitud hasta ahora, actualizar información
                    if aptitud < mejor_aptitud:
                        mejor_aptitud = aptitud
                        mejor_convergencia = hist_aptitud
                        mejor_x = mejor_x_tmp
                        mejor_config = {
                            'mu': mu,
                            'alpha': alpha,
                            'sigma_inicial': sigma_inicial,
                            'replica': replica
                        }

                    # Imprimir resumen de esta ejecución
                    print(f"Config (mu={mu}, alpha={alpha}, sigma={sigma_inicial}), réplica {replica}, aptitud={aptitud:.6f}")

    # Convertir resultados a DataFrame y guardar en CSV
    df = pd.DataFrame(resultados)
    df.to_csv("resultados_beam.csv", index=False)
    print("\nResultados guardados en resultados_beam.csv")

    # Realizar ANOVA para evaluar influencia de los factores
    formula = 'aptitud ~ C(mu) + C(alpha) + C(sigma_inicial) + C(mu):C(alpha) + C(mu):C(sigma_inicial) + C(alpha):C(sigma_inicial)'
    modelo = ols(formula, data=df).fit()
    anova_resultado = sm.stats.anova_lm(modelo, typ=2)

    # Mostrar resultados del ANOVA
    print("\nANOVA Resultados:")
    print(anova_resultado)

    # Gráficos de efectos principales de cada factor
    plt.figure(figsize=(12, 4))
    for i, factor in enumerate(['mu', 'alpha', 'sigma_inicial']):
        plt.subplot(1, 3, i + 1)
        df.groupby(factor)['aptitud'].mean().plot(kind='bar')
        plt.title(f"Efecto de {factor}")
        plt.ylabel("Aptitud promedio")
    plt.tight_layout()
    plt.savefig("graficas_resultados/efectos_principales.png")
    plt.close()
    print("Gráfica de efectos principales guardada en graficas_resultados/efectos_principales.png")

    # Gráfico de interacción entre mu y alpha
    plt.figure(figsize=(10, 5))
    for alpha_val in alphas:
        subset = df[df['alpha'] == alpha_val]
        medias = subset.groupby('mu')['aptitud'].mean()
        plt.plot(medias.index, medias.values, marker='o', label=f'alpha={alpha_val}')
    plt.xlabel("mu")
    plt.ylabel("Aptitud promedio")
    plt.title("Interacción mu x alpha")
    plt.legend()
    plt.grid(True)
    plt.savefig("graficas_resultados/interaccion_mu_alpha.png")
    plt.close()
    print("Gráfica de interacción mu x alpha guardada en graficas_resultados/interaccion_mu_alpha.png")

    # Gráfica de convergencia de la mejor ejecución encontrada
    plt.figure(figsize=(8, 5))
    plt.plot(mejor_convergencia, 'b-')
    plt.title('Convergencia - Mejor ejecución')
    plt.xlabel('Generación')
    plt.ylabel('Mejor aptitud')
    plt.grid(True)
    plt.savefig("graficas_resultados/convergencia_mejor.png")
    plt.show()

    # Imprimir información de la mejor ejecución encontrada
    print("\n🔹 Información de la mejor ejecución encontrada:")
    print(f"Aptitud: {mejor_aptitud:.8f}")
    print(f"Solución: {mejor_x}")
    print(f"Configuración: mu={mejor_config['mu']}, alpha={mejor_config['alpha']}, sigma_inicial={mejor_config['sigma_inicial']}, réplica={mejor_config['replica']}")
    print("Gráfica de convergencia guardada en graficas_resultados/convergencia_mejor.png")

# Ejecutar los experimentos si este script se llama directamente
if __name__ == "__main__":
    ejecutar_experimentos()
