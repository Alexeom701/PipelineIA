import os
import yaml
import pandas as pd
from modules.loader import cargar_archivo
from modules.exporter import exportar_df
from utils.logger import configurar_logger

# Configurar el logger
logger = configurar_logger()

def cargar_configuracion(ruta="config.yaml"):
    """
    Carga la configuración desde un archivo YAML.
    """
    with open(ruta, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def procesar_archivo(config):
    """
    Carga datos desde la ruta de entrada y los exporta a la ruta de salida
    sin aplicar ninguna limpieza.
    """
    ruta_entrada = config['entrada']['archivo']
    ruta_salida = config['salida']['archivo']

    logger.info(f"Archivo de entrada: {ruta_entrada}")
    logger.info(f"Archivo de salida: {ruta_salida}")

    try:
        # Cargar los datos
        df = cargar_archivo(ruta_entrada)
        logger.info(f"Archivo cargado exitosamente: {ruta_entrada}")
        logger.info(f"Dimensiones del DataFrame: {df.shape}")

        # Exportar el DataFrame tal cual
        exportar_df(df, ruta_salida)
        logger.info(f"Archivo exportado exitosamente: {ruta_salida}")

    except Exception as e:
        logger.exception(f"Error al procesar el archivo: {e}")

def main():
    """
    Función principal para iniciar el pipeline de procesamiento de productos.
    """
    logger.info("Iniciando pipeline de procesamiento de productos (solo carga y exportación).")
    config = cargar_configuracion()
    procesar_archivo(config)
    logger.info("Pipeline finalizado.")

if __name__ == "__main__":
    main()