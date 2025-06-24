import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
import yaml
from utils.logger import configurar_logger

logger = configurar_logger()

def cargar_config(ruta="config.yaml"):
    with open(ruta, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = cargar_config()
cleaner_cfg = config["entrenamiento_cleaner"]

MODEL_NAME = cleaner_cfg["modelo_base"]
TRAIN_PATH = cleaner_cfg["datos_entrenamiento"]
OUTPUT_DIR = cleaner_cfg["salida_modelo"]
BATCH_SIZE = cleaner_cfg["batch_size"]
EPOCHS = cleaner_cfg["epochs"]
MAX_INPUT_LEN = cleaner_cfg["max_input_length"]
MAX_TARGET_LEN = cleaner_cfg["max_target_length"]

class CleaningDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_length=96, max_target_length=96):
        self.input_texts = dataframe['texto_original'].tolist()
        self.target_texts = dataframe['texto_limpio'].tolist()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.input_texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            self.target_texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

df = pd.read_csv(TRAIN_PATH)
logger.info(f"Datos de entrenamiento cargados: {df.shape}")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

logger.info(f"Cargando modelo base: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

train_dataset = CleaningDataset(train_df, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)
val_dataset = CleaningDataset(val_df, tokenizer, MAX_INPUT_LEN, MAX_TARGET_LEN)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=20,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
logger.info("Iniciando entrenamiento del modelo de limpieza...")
trainer.train()

logger.info(f"Guardando modelo entrenado en: {OUTPUT_DIR}")
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
logger.info("Modelo y tokenizer guardados exitosamente.")
