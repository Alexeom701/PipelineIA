import pandas as pd
import os

class DataLoader:
    def __init__(self):
        pass

    def cargar_datos(self, ruta_archivo: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV o JSON en un DataFrame de pandas.

        Args:
            ruta_archivo (str): La ruta al archivo de entrada.

        Returns:
            pd.DataFrame: Los datos cargados.

        Raises:
            ValueError: Si el formato del archivo no es compatible o el archivo no existe.
        """
        if not os.path.exists(ruta_archivo):
            raise ValueError(f"Archivo no encontrado: {ruta_archivo}")

        _, extension_archivo = os.path.splitext(ruta_archivo)
        extension_archivo = extension_archivo.lower()

        if extension_archivo == '.csv':
            df = pd.read_csv(ruta_archivo)
        elif extension_archivo == '.json':
            df = pd.read_json(ruta_archivo)
        else:
            raise ValueError(f"Formato de archivo no compatible: {extension_archivo}. Solo se admiten .csv y .json.")

        print(f"Datos cargados exitosamente desde {ruta_archivo}")
        return df