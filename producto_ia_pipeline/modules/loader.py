import os
import pandas as pd
import json
from utils.logger import configurar_logger

logger = configurar_logger()

ENCODINGS_PROBADOS = ['utf-8', 'latin1', 'ISO-8859-1']

def cargar_csv(ruta_archivo):
    for encoding in ENCODINGS_PROBADOS:
        try:
            df = pd.read_csv(ruta_archivo, encoding=encoding)
            logger.info(f"CSV cargado correctamente con encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            logger.warning(f"Fallo de codificación con: {encoding}")
        except pd.errors.EmptyDataError:
            raise RuntimeError("El archivo CSV está vacío.")
        except Exception as e:
            logger.warning(f"Otro error leyendo CSV con {encoding}: {e}")
    raise RuntimeError("No se pudo cargar el archivo CSV con los encodings disponibles.")

def cargar_json(ruta_archivo):
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info("JSON cargado correctamente con encoding utf-8")
        return df
    except json.JSONDecodeError:
        raise RuntimeError("El archivo JSON no tiene un formato válido.")
    except Exception as e:
        raise RuntimeError(f"Error cargando JSON: {e}")

def cargar_archivo(ruta_archivo):
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    extension = os.path.splitext(ruta_archivo)[1].lower()

    if extension == '.csv':
        return cargar_csv(ruta_archivo)
    elif extension == '.json':
        return cargar_json(ruta_archivo)
    else:
        raise ValueError(f"Formato de archivo no soportado: {extension}")
