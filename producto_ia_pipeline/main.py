from modules.loader import load_file
from modules.cleaner import clean_data
from modules.classifier import classify_sector
from modules.standardizer import standardize_fields
from modules.extractor import extract_attributes
from modules.transformer import transform_data
from modules.outlier import detect_outliers
from modules.exporter import export_data

def main(input_path, output_path):
    # 1. Cargar archivo
    df = load_file(input_path)

    # 2. Limpieza básica
    df = clean_data(df)

    # 3. Clasificación por sector con IA
    df = classify_sector(df)

    # 4. Estandarización de unidades, marcas, etc.
    df = standardize_fields(df)

    # 5. Extracción de atributos clave
    df = extract_attributes(df)

    # 6. Transformaciones (IVA, conversiones, etc.)
    df = transform_data(df)

    # 7. Detección de outliers o errores
    df = detect_outliers(df)

    # 8. Exportación del resultado limpio
    export_data(df, output_path)

if __name__ == "__main__":
    input_path = "data/raw/ejemplo.csv"
    output_path = "data/processed/ejemplo_limpio.csv"
    main(input_path, output_path)
