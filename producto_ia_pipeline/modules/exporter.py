import pandas as pd
import os

class DataExporter:
    def __init__(self):
        pass

    def exportar_datos(self, df: pd.DataFrame, ruta_salida: str, formato_archivo: str = 'csv', **kwargs):
        """
        Exporta un DataFrame de pandas a un formato de archivo especificado (CSV o JSON).

        Args:
            df (pd.DataFrame): El DataFrame a exportar.
            ruta_salida (str): La ruta donde se guardará el archivo.
            formato_archivo (str): El formato de salida deseado ('csv' o 'json').
                                   Por defecto es 'csv'.
            **kwargs: Argumentos adicionales de palabra clave para pasar a pandas to_csv o to_json.

        Raises:
            ValueError: Si el formato del archivo no es compatible.
            IOError: Si hay un problema al escribir el archivo.
        """
        # Asegurarse de que el directorio exista
        directorio_salida = os.path.dirname(ruta_salida)
        if directorio_salida and not os.path.exists(directorio_salida):
            os.makedirs(directorio_salida, exist_ok=True)
            print(f"Directorio de salida creado: {directorio_salida}")

        try:
            if formato_archivo.lower() == 'csv':
                df.to_csv(ruta_salida, index=False, **kwargs)
            elif formato_archivo.lower() == 'json':
                df.to_json(ruta_salida, orient='records', indent=4, **kwargs)
            else:
                raise ValueError(f"Formato de archivo no compatible: {formato_archivo}. Solo se admiten 'csv' y 'json'.")

            print(f"Datos exportados exitosamente a {ruta_salida} en formato {formato_archivo}.")
        except Exception as e:
            raise IOError(f"Error al exportar datos a {ruta_salida}: {e}")