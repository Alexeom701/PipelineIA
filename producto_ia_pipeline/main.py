import os
import yaml
import pandas as pd
import sys


sys.path.append(os.path.abspath("."))

from modules.loader import cargar_archivo
from modules.exporter import exportar_df
from modules.cleaner import limpiar_dataframe 

from utils.logger import configurar_logger
logger = configurar_logger()

def cargar_configuracion(ruta="config.yaml"):

    with open(ruta, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def procesar_archivo(config):

    ruta_entrada = config['entrada']['archivo']
    ruta_salida = config['salida']['archivo']
    formato_salida = config['salida'].get('formato', 'csv')

    logger.info(f"Cargando archivo desde: {ruta_entrada}")
    try:
        df = cargar_archivo(ruta_entrada)
        logger.info(f"Archivo cargado con forma: {df.shape}")

        
        logger.info("Aplicando limpieza híbrida (reglas + modelo)...")
        df = limpiar_dataframe(df)

        exportar_df(df, ruta_salida, formato=formato_salida)
        logger.info(f"Archivo exportado correctamente a: {ruta_salida}")

    except Exception as e:
        logger.exception(f"Error procesando el archivo: {e}")

def main():

    logger.info("Iniciando pipeline de limpieza...")
    config = cargar_configuracion()
    procesar_archivo(config)
    logger.info("Pipeline finalizado.")

if __name__ == "__main__":
    main()
