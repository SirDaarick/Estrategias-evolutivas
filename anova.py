import pandas as pd

def generar_resumen(csv_path="Estrategias-evolutivas/resultados_beam.csv"):
    # Cargar los resultados existentes
    df = pd.read_csv(csv_path)

    # Generar el analisis ANOVA
    resumen = df.groupby(['mu', 'alpha', 'sigma_inicial']).agg(
        costo_promedio=('aptitud', 'mean'),
        costo_mejor=('aptitud', 'min'),
        costo_peor=('aptitud', 'max'),
        desviacion=('aptitud', 'std')
    ).reset_index()

    resumen.insert(0, 'config_id', range(1, len(resumen) + 1))
    print("\nRESUMEN ESTADÍSTICO POR CONFIGURACIÓN:")
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Guardar en un CSV
    resumen.to_csv("resumen_estadistico.csv", index=False)
    print("\nResumen estadístico guardado en resumen_estadistico.csv")

if __name__ == "__main__":
    generar_resumen()
