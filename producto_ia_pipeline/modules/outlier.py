import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from utils.logger import get_logger

logger = get_logger(__name__)

# Config global
MODEL_PATH = os.path.join("models", "outlier_detector", "outlier_model.pkl")

def detectar_outliers(df, columnas_a_evaluar=None):
    """
    Aplica un modelo IA para detectar outliers en los datos.
    
    Parámetros:
        df: DataFrame con los datos de productos
        columnas_a_evaluar: lista de columnas numéricas a evaluar

    Devuelve:
        df: DataFrame con una nueva columna 'es_outlier' marcada como True/False
    """
    if columnas_a_evaluar is None:
        columnas_a_evaluar = ["precio", "peso", "volumen"]  # puedes adaptar a tus columnas

    df_outlier = df.copy()
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Modelo de outlier cargado correctamente.")
    except Exception as e:
        logger.warning(f"No se pudo cargar el modelo entrenado. Se usará uno nuevo. {e}")
        model = IsolationForest(contamination=0.05, random_state=42)
        df_validos = df_outlier[columnas_a_evaluar].dropna()
        model.fit(df_validos)

    try:
        df_outlier["es_outlier"] = model.predict(df_outlier[columnas_a_evaluar])
        df_outlier["es_outlier"] = df_outlier["es_outlier"] == -1  # -1 son outliers
    except Exception as e:
        logger.error(f"Error al predecir outliers: {e}")
        df_outlier["es_outlier"] = False

    return df_outlier
