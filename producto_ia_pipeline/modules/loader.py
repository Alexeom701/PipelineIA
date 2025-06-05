# modules/loader.py
import pandas as pd

def load_data(filepath):
    """
    Carga datos hola desde un archivo CSV o JSON.
    """
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        df = pd.read_json(filepath)
    else:
        raise ValueError("Formato de archivo no soportado. Usa .csv o .json")
    print(f"Datos cargados exitosamente desde: {filepath}")
    return df