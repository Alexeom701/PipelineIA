import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self):
        pass

    def limpiar_columnas_texto(self, df: pd.DataFrame, columnas: list = None) -> pd.DataFrame:
        """
        Realiza una limpieza básica de texto en columnas de tipo string específicas o todas:
        - Elimina espacios al inicio y al final.
        - Convierte el texto a mayúsculas.
        - Reemplaza múltiples espacios por un solo espacio.
        - Convierte valores no string a string antes de limpiar para evitar errores.

        Args:
            df (pd.DataFrame): El DataFrame de entrada.
            columnas (list, optional): Una lista de nombres de columnas a limpiar. Si es None,
                                      se limpiarán todas las columnas de tipo 'object' (string).

        Returns:
            pd.DataFrame: El DataFrame con las columnas de texto limpiadas.
        """
        df_limpio = df.copy()
        columnas_a_procesar = []

        if columnas is None:
            # Seleccionar todas las columnas de tipo 'object'
            columnas_a_procesar = df_limpio.select_dtypes(include=['object']).columns.tolist()
        else:
            # Filtrar solo las columnas existentes y de tipo 'object' o que puedan ser convertidas a string
            for col in columnas:
                if col in df_limpio.columns:
                    # Considerar columnas que son object o que no son numéricas para limpieza de texto
                    if pd.api.types.is_object_dtype(df_limpio[col]) or not pd.api.types.is_numeric_dtype(df_limpio[col]):
                        columnas_a_procesar.append(col)
                else:
                    print(f"Advertencia: La columna '{col}' especificada para limpieza de texto no existe en el DataFrame.")

        if not columnas_a_procesar:
            print("No se encontraron columnas de texto válidas para limpiar entre las especificadas o detectadas.")
            return df_limpio

        for col in columnas_a_procesar:
            # Asegurarse de que la columna sea de tipo string antes de aplicar métodos de string
            # Convertir a string explícitamente para manejar NaNs y otros tipos de datos sin errores
            df_limpio[col] = df_limpio[col].astype(str)
            df_limpio[col] = df_limpio[col].str.strip()
            df_limpio[col] = df_limpio[col].str.upper()
            df_limpio[col] = df_limpio[col].str.replace(r'\s+', ' ', regex=True) # Reemplazar múltiples espacios por uno
            # Opcional: Reemplazar 'NAN' o 'NONE' string que resultan de la conversión de np.nan
            df_limpio[col] = df_limpio[col].replace({'NAN': np.nan, 'NONE': np.nan})

        print(f"Limpieza de texto aplicada a las columnas: {', '.join(columnas_a_procesar)}")
        return df_limpio

    def manejar_valores_faltantes(self, df: pd.DataFrame, estrategia: str = 'eliminar_filas', columnas: list = None, valor_relleno: any = None) -> pd.DataFrame:
        """
        Maneja los valores faltantes (NaN) en el DataFrame.

        Args:
            df (pd.DataFrame): El DataFrame de entrada.
            estrategia (str): La estrategia para manejar los valores faltantes.
                              'eliminar_filas': Elimina filas con cualquier valor NaN en las columnas especificadas.
                              'rellenar_media': Rellena NaN numéricos con la media.
                              'rellenar_mediana': Rellena NaN numéricos con la mediana.
                              'rellenar_moda': Rellena NaN categóricos/numéricos con la moda.
                              'rellenar_valor_fijo': Rellena NaN con un valor específico (requiere 'valor_relleno').
            columnas (list, optional): Una lista de nombres de columnas a las que aplicar la estrategia.
                                      Si es None, la estrategia se aplica a todas las columnas.
            valor_relleno (any, optional): El valor a usar si la estrategia es 'rellenar_valor_fijo'.

        Returns:
            pd.DataFrame: El DataFrame después de manejar los valores faltantes.

        Raises:
            ValueError: Si se proporciona una estrategia no compatible o falta 'valor_relleno' para 'rellenar_valor_fijo'.
        """
        df_limpio = df.copy()
        columnas_a_procesar = []

        if columnas is None:
            columnas_a_procesar = df_limpio.columns.tolist()
        else:
            for col in columnas:
                if col in df_limpio.columns:
                    columnas_a_procesar.append(col)
                else:
                    print(f"Advertencia: La columna '{col}' especificada para manejo de valores faltantes no existe en el DataFrame.")

        if not columnas_a_procesar:
            print("No se encontraron columnas válidas para el manejo de valores faltantes.")
            return df_limpio

        filas_iniciales = len(df_limpio)

        if estrategia == 'eliminar_filas':
            df_limpio.dropna(subset=columnas_a_procesar, inplace=True)
            print(f"Se eliminaron {filas_iniciales - len(df_limpio)} filas con valores faltantes en las columnas especificadas.")
        elif estrategia.startswith('rellenar_'):
            for col in columnas_a_procesar:
                if df_limpio[col].isnull().any(): # Solo procesar si hay NaNs
                    if pd.api.types.is_numeric_dtype(df_limpio[col]):
                        if estrategia == 'rellenar_media':
                            val = df_limpio[col].mean()
                            df_limpio[col].fillna(val, inplace=True)
                            print(f"NaNs en columna numérica '{col}' rellenados con la media ({val}).")
                        elif estrategia == 'rellenar_mediana':
                            val = df_limpio[col].median()
                            df_limpio[col].fillna(val, inplace=True)
                            print(f"NaNs en columna numérica '{col}' rellenados con la mediana ({val}).")
                        elif estrategia == 'rellenar_moda':
                            if not df_limpio[col].mode().empty:
                                val = df_limpio[col].mode()[0]
                                df_limpio[col].fillna(val, inplace=True)
                                print(f"NaNs en columna numérica '{col}' rellenados con la moda ({val}).")
                            else:
                                print(f"No se pudo calcular la moda para la columna numérica '{col}'.")
                        elif estrategia == 'rellenar_valor_fijo':
                            if valor_relleno is not None:
                                df_limpio[col].fillna(valor_relleno, inplace=True)
                                print(f"NaNs en columna '{col}' rellenados con el valor fijo '{valor_relleno}'.")
                            else:
                                raise ValueError("Para la estrategia 'rellenar_valor_fijo', debe proporcionar un 'valor_relleno'.")
                        else:
                            raise ValueError(f"Estrategia de relleno no compatible para columnas numéricas: {estrategia}")
                    else: # Asumiendo tipo 'object'/categórico
                        if estrategia == 'rellenar_moda':
                            if not df_limpio[col].mode().empty:
                                val = df_limpio[col].mode()[0]
                                df_limpio[col].fillna(val, inplace=True)
                                print(f"NaNs en columna categórica '{col}' rellenados con la moda ('{val}').")
                            else:
                                print(f"No se pudo calcular la moda para la columna categórica '{col}'.")
                        elif estrategia == 'rellenar_valor_fijo':
                            if valor_relleno is not None:
                                df_limpio[col].fillna(valor_relleno, inplace=True)
                                print(f"NaNs en columna '{col}' rellenados con el valor fijo '{valor_relleno}'.")
                            else:
                                raise ValueError("Para la estrategia 'rellenar_valor_fijo', debe proporcionar un 'valor_relleno'.")
                        else:
                            print(f"Omitiendo columna no numérica '{col}' para la estrategia '{estrategia}' (solo 'rellenar_moda' o 'rellenar_valor_fijo' son aplicables aquí).")
        else:
            raise ValueError(f"Estrategia de manejo de valores faltantes no compatible: {estrategia}")

        return df_limpio