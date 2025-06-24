import pandas as pd
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dateutil import parser


MODEL_PATH = "models/cleaner"
EMPTY_FIELD_PLACEHOLDER = "Sin_info"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

COLUMNA_ALIAS = {
    "nombre_producto": ["product_name", "name", "name_product", "producto_nombre", "nombre", "nombre_producto"],
    "SKU": ["sku", "product_sku", "codigo_sku"],
    "precio": ["price", "cost", "product_price", "valor", "precio_producto"],
    "stock": ["availability", "inventory", "in_stock", "existencias", "cantidad_disponible"],
    "color": ["colour", "product_color", "color_producto"],
    "talla": ["size", "product_size", "tamaño", "medida", "talla_producto"],
    "descripcion_corta": ["short_description", "descripcion", "desc", "product_description", "descripcion_producto"],
    "unidad_de_medida": ["unit", "measurement_unit", "unidad", "unidad_medida"],
    "fecha": ["date", "publish_date", "release_date", "fecha_publicacion"],
    "url_imagen": ["image_url", "url", "product_image", "link_imagen", "imagen_url"],
    "codigo_barras": ["barcode", "ean", "product_code", "codigo", "codigo_producto"]
}

def normalizar_nombre_columna(nombre_original):
    nombre_limpio = nombre_original.lower()
    for nombre_canonico, aliases in COLUMNA_ALIAS.items():
        if nombre_limpio == nombre_canonico or nombre_limpio in aliases:
            return nombre_canonico
    return nombre_limpio

# === Reglas personalizadas ===
def limpiar_fecha(texto):
    if not isinstance(texto, str):
        texto = str(texto)
    texto = texto.strip().lower()
    texto = re.sub(r"(de|del)", "", texto)
    texto = texto.replace(",", " ").replace(".", "/").replace("-", "/")
    texto = re.sub(r"\s+", " ", texto)
    try:
        fecha = parser.parse(texto, dayfirst=True, fuzzy=True)
        return fecha.strftime("%d/%m/%Y")
    except Exception:
        return None

def limpiar_precio(texto):
    texto = str(texto).strip().replace(",", ".")
    texto = re.sub(r"[^\d\.]", "", texto)
    try:
        return str(round(float(texto), 2))
    except:
        return None

def limpiar_stock(texto):
    texto = str(texto).lower().strip()
    if "sin stock" in texto:
        return "0"
    texto = texto.replace("+", "")
    texto = re.sub(r"[^\d]", "", texto)
    return texto if texto else None

def limpiar_url(texto):
    texto = str(texto).strip()
    if not texto.startswith("http") and texto != "":
        texto = "https://" + texto
    return texto if texto != "" else None

def limpiar_celda(nombre_columna, valor):
    if pd.isna(valor) or str(valor).strip() == "":
        return EMPTY_FIELD_PLACEHOLDER

    texto = str(valor).strip()
    nombre_col = normalizar_nombre_columna(nombre_columna)
    resultado = None

    if "fecha" in nombre_col:
        resultado = limpiar_fecha(texto)
    elif "precio" in nombre_col:
        resultado = limpiar_precio(texto)
    elif "stock" in nombre_col:
        resultado = limpiar_stock(texto)
    elif "url" in nombre_col:
        resultado = limpiar_url(texto)

    if resultado is not None:
        prompt_type = "clean"
        texto = resultado
    else:
        prompt_type = "dirty"

    prompt = f"{prompt_type} {nombre_col}: {texto}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=96).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=126)

    model_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return model_output if model_output.strip() else EMPTY_FIELD_PLACEHOLDER

def limpiar_dataframe(df):
    tqdm.pandas()
    for col in df.columns:
        df[col] = df[col].progress_apply(lambda x: limpiar_celda(col, x))
    return df
