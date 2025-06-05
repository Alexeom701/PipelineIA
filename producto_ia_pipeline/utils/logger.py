import logging
import os
from datetime import datetime

def configurar_logger(nombre_logger="producto_ia", nivel_consola=logging.INFO, nivel_archivo=logging.DEBUG):
    """
    Crea y configura un logger centralizado para todo el pipeline.
    """
    # Crear carpeta de logs si no existe
    carpeta_logs = "logs"
    if not os.path.exists(carpeta_logs):
        os.makedirs(carpeta_logs)

    # Nombre del archivo de log con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_log = os.path.join(carpeta_logs, f"{nombre_logger}_{timestamp}.log")

    # Crear el logger
    logger = logging.getLogger(nombre_logger)
    logger.setLevel(logging.DEBUG)  # Captura todo, lo filtran los handlers

    # Evitar handlers duplicados si se llama varias veces
    if logger.hasHandlers():
        return logger

    # Formato común
    formato = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')

    # Handler de consola
    consola_handler = logging.StreamHandler()
    consola_handler.setLevel(nivel_consola)
    consola_handler.setFormatter(formato)
    logger.addHandler(consola_handler)

    # Handler de archivo
    archivo_handler = logging.FileHandler(ruta_log, encoding='utf-8')
    archivo_handler.setLevel(nivel_archivo)
    archivo_handler.setFormatter(formato)
    logger.addHandler(archivo_handler)

    logger.info(f"Logger inicializado. Logs en: {ruta_log}")
    return logger
