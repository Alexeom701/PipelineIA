import pandas as pd
import re
from dateutil.parser import parse
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class AICleaner:
    def __init__(self, config):
        self.config = config
        self.nlp_pipeline = None

        cleaner_model_path = config.get('cleaner_model_path')
        if cleaner_model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(cleaner_model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(cleaner_model_path)
                # Usamos una pipeline de texto a texto para las transformaciones de limpieza
                self.nlp_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
                print(f"Modelo de limpieza IA cargado desde: {cleaner_model_path}")
            except Exception as e:
                print(f"Advertencia: No se pudo cargar el modelo de limpieza avanzado desde {cleaner_model_path}. Error: {e}")
                print("El módulo de limpieza IA operará solo con reglas heurísticas.")

    def _apply_heuristic_rules(self, text):
        """
        Aplica un conjunto de reglas de limpieza básicas y heurísticas.
        Esto actúa como una primera capa de limpieza rápida y eficiente.
        """
        if not isinstance(text, str):
            return text

        # 1. Normalización de espacios y mayúsculas/minúsculas
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip() # Unifica y elimina espacios extra

        # 2. Unificación de unidades (ejemplos para productos)
        text = text.replace(' ml', 'ml').replace('ml.', 'ml')
        text = text.replace(' lts', 'l').replace('lt.', 'l')
        text = text.replace(' kgs', 'kg').replace('kg.', 'kg')
        text = text.replace(' gramos', 'gr').replace(' grs', 'gr')

        # 3. Formateo básico de fechas (puede ser más sofisticado con dateutil)
        try:
            # Intenta parsear como fecha si contiene números y separadores comunes de fecha
            if any(char.isdigit() for char in text) and any(sep in text for sep in ['/', '-', '.']):
                date_obj = parse(text, fuzzy=True, dayfirst=False) # dayfirst=False para M/D/Y o D/M/Y
                text = date_obj.strftime('%Y-%m-%d')
        except (ValueError, OverflowError):
            pass # No es una fecha, continuar

        # 4. Correcciones comunes y generalizadas (ejemplos)
        text = text.replace('boteya', 'botella')
        text = text.replace('asucar', 'azúcar')
        text = text.replace('refresko', 'refresco')
        text = text.replace('marlboro', 'Marlboro') # Ejemplo de capitalización específica si se requiere

        return text

    def _apply_ai_cleaning(self, text):
        """
        Aplica el modelo de IA (LLM/encoder-decoder) para limpieza avanzada.
        """
        if not isinstance(text, str) or not self.nlp_pipeline:
            return text # Si el modelo no está cargado, devuelve el texto original

        # La prompt es crucial. Podría ser "clean: " o "normalize: "
        # Experimenta con diferentes prompts para ver cuál funciona mejor con tu modelo.
        prompt = f"Limpia y normaliza el siguiente texto de un producto: {text}"
        try:
            # Ajusta max_new_tokens, num_beams y otros parámetros de generación según tu modelo
            # Asegúrate de que el modelo devuelva solo el texto limpio, no el prompt.
            result = self.nlp_pipeline(prompt, max_new_tokens=50, num_beams=1, do_sample=False)
            cleaned_text = result[0]['generated_text']
            # Algunos modelos pueden incluir el prompt en la salida, necesitarás quitarlo.
            if cleaned_text.startswith(prompt):
                cleaned_text = cleaned_text[len(prompt):].strip()
            return cleaned_text
        except Exception as e:
            print(f"Error al aplicar LLM para limpieza en '{text}': {e}. Devolviendo resultado de heurísticas.")
            return text # En caso de error, puedes devolver la versión limpia por heurísticas

    def clean_data(self, df: pd.DataFrame, columns_to_clean: list):
        """
        Aplica las reglas de limpieza a las columnas especificadas del DataFrame.
        Primero aplica heurísticas, luego el modelo de IA si está disponible.
        """
        cleaned_df = df.copy()

        for col in columns_to_clean:
            if col not in cleaned_df.columns:
                print(f"Advertencia: La columna '{col}' no se encontró en el DataFrame. Se omitirá.")
                continue

            print(f"Limpiando columna '{col}'...")
            # Aplicar primero las reglas heurísticas a toda la columna
            cleaned_df[col] = cleaned_df[col].apply(self._apply_heuristic_rules)

            # Si el modelo de IA está cargado, aplicar limpieza avanzada
            if self.nlp_pipeline:
                # Nota: Aplicar a cada celda con apply es lento para LLMs grandes.
                # Para datasets muy grandes, considera procesar en batches o usar librerías
                # optimizadas para inferencia con LLMs (ej., vLLM o Hugging Face Inference Endpoints).
                print(f"Aplicando limpieza avanzada con IA a la columna '{col}' (puede tomar tiempo)...")
                cleaned_df[col] = cleaned_df[col].apply(self._apply_ai_cleaning)

        return cleaned_df