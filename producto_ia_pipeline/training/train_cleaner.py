import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import os
import torch

def train_cleaner_model(data_path, output_dir, model_name="google/flan-t5-small", epochs=5, batch_size=8, max_seq_length=128):
    """
    Fine-tunes un modelo de lenguaje (LLM o encoder-decoder) para la tarea de limpieza de texto.

    Args:
        data_path (str): Ruta al archivo CSV con datos de entrenamiento.
                         Debe tener al menos dos columnas: 'dirty_text' y 'clean_text'.
                         Es crucial que estos datos cubran una amplia variedad de
                         "suciedad" y "limpieza" esperada en diferentes sectores de productos.
        output_dir (str): Directorio donde se guardará el modelo entrenado.
        model_name (str): Nombre del modelo base de Hugging Face a usar (ej. "google/flan-t5-small").
        epochs (int): Número de épocas para el entrenamiento.
        batch_size (int): Tamaño del batch para el entrenamiento.
        max_seq_length (int): Longitud máxima de secuencia para el tokenizer.
    """
    print(f"Cargando datos de entrenamiento desde: {data_path}")
    try:
        df_train = pd.read_csv(data_path)
        if 'dirty_text' not in df_train.columns or 'clean_text' not in df_train.columns:
            raise ValueError("El archivo de entrenamiento debe contener las columnas 'dirty_text' y 'clean_text'.")
    except FileNotFoundError:
        print(f"Error: Archivo de datos no encontrado en {data_path}. Asegúrate de tener un CSV.")
        return
    except ValueError as e:
        print(f"Error en el formato del archivo de datos: {e}")
        return

    # Convertir DataFrame a Dataset de Hugging Face
    dataset = Dataset.from_pandas(df_train)

    print(f"Cargando tokenizer y modelo base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocesar los datos para el formato del modelo Seq2Seq
    def preprocess_function(examples):
        # Es vital que la prompt sea consistente con cómo se usará en `cleaner_ai.py`
        inputs = [f"clean: {text}" for text in examples["dirty_text"]]
        model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True, padding="max_length")

        labels = tokenizer(text_target=examples["clean_text"], max_length=max_seq_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=df_train.columns.tolist())

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch", # Guardar al final de cada época
        save_total_limit=1,    # Mantener solo el último checkpoint
        evaluation_strategy="no", # Si tienes un conjunto de validación, cámbialo a "epoch"
        logging_dir='./logs',
        logging_steps=50,      # Frecuencia de logueo
        learning_rate=2e-5,    # Tasa de aprendizaje común para fine-tuning
        fp16=torch.cuda.is_available(), # Habilitar FP16 si hay GPU para entrenamiento más rápido
    )

    # Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer, # Pasar el tokenizer al Trainer
    )

    print("Iniciando entrenamiento del modelo de limpieza...")
    trainer.train()
    print("Entrenamiento completado.")

    # Guardar el modelo y el tokenizer final
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo de limpieza IA y tokenizer guardados en: {output_dir}")

if __name__ == "__main__":
    # Define la ruta de tus datos de entrenamiento y el directorio de salida del modelo.
    # Es crucial generar un archivo CSV con columnas 'dirty_text' y 'clean_text'.
    # Estos ejemplos deben abarcar la diversidad de los datos de productos y su suciedad.
    training_data_file = 'data/training/cleaner_training_data.csv'
    cleaner_model_output_dir = 'models/cleaner/'

    os.makedirs(os.path.dirname(training_data_file), exist_ok=True)
    os.makedirs(cleaner_model_output_dir, exist_ok=True)

    # --- Generación de datos de entrenamiento de ejemplo (¡Muy importante para la flexibilidad!) ---
    # Para que tu IA sea capaz de limpiar "casi cualquier archivo",
    # ¡necesitas un conjunto de datos de entrenamiento extremadamente diverso y representativo!
    # Incluye ejemplos de diferentes sectores (alimentación, electrónica, ropa, etc.)
    # y diferentes tipos de "suciedad" (typos, formatos inconsistentes, información irrelevante).
    if not os.path.exists(training_data_file):
        print(f"Creando un archivo de datos de entrenamiento de ejemplo en {training_data_file}")
        sample_training_data = pd.DataFrame({
            'dirty_text': [
                'coca-cola lata 355ml',
                'pEpsi botella 2 litros',
                '   Agua pura     ',
                'Leche semidescremada, 1 L.',
                'Papas fritas, 150grs.',
                'Cigarros marlboro light',
                'Fecha_pedido: 01/01/2023',
                'FECHA DE ENTRADA: ENERO 15, 2024',
                'iPhone XII Pro Max 128gb (usado)', # Ejemplo de sector electrónico
                'CamiSetA azUl, talla m', # Ejemplo de sector de ropa
                'Pasta dental Fluoride + menta 75gr', # Otro sector
                'Detergente liquido 1.5 lts "lava_mas"', # Otro sector
                'Laptop hp i5-10th gen 8gb ram ssd 256gb', # Más electrónica
                'Zapatos deportivos NIke air max negross t. 42',
                'Queso oaxaka, 500 gr.'
            ],
            'clean_text': [
                'Coca-Cola Lata 355ml',
                'Pepsi Botella 2 litros',
                'Agua Pura',
                'Leche Semidescremada 1 litro',
                'Papas Fritas 150 gramos',
                'Cigarrillos Marlboro Light',
                '2023-01-01',
                '2024-01-15',
                'iPhone 12 Pro Max 128GB',
                'Camiseta Azul, Talla M',
                'Pasta Dental Fluoride Menta 75g',
                'Detergente Líquido 1.5 litros "Lava Más"',
                'Laptop HP Core i5 10ma Generación 8GB RAM 256GB SSD',
                'Zapatos Deportivos Nike Air Max Negros Talla 42',
                'Queso Oaxaca 500 gramos'
            ]
        })
        sample_training_data.to_csv(training_data_file, index=False)
        print("Archivo de ejemplo creado. **¡Añade muchísimos más datos reales y diversos para un mejor entrenamiento!**")
        print("La calidad y diversidad de tus datos de entrenamiento determinarán la versatilidad de tu IA.")
    # --- Fin de la generación de datos de ejemplo ---

    # Ejecutar el entrenamiento
    train_cleaner_model(
        data_path=training_data_file,
        output_dir=cleaner_model_output_dir,
        model_name="google/flan-t5-small", # Este es un buen punto de partida, pequeño pero efectivo
        epochs=10, # Aumenta las épocas para un entrenamiento más robusto
        batch_size=4 # Ajusta según la memoria de tu GPU
    )