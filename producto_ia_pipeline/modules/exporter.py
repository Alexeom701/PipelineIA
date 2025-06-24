import pandas as pd
import os

def exportar_df(df, ruta_salida, formato='csv'):
    """Exporta un DataFrame a CSV o JSON con verificación de ruta"""
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    try:
        if formato == 'csv':
            df.to_csv(ruta_salida, index=False, encoding='utf-8')
        elif formato == 'json':
            df.to_json(ruta_salida, orient='records', force_ascii=False, indent=2)
        else:
            raise ValueError(f"Formato de exportación no soportado: {formato}")
    except Exception as e:
        raise RuntimeError(f"Error al exportar archivo: {e}")
