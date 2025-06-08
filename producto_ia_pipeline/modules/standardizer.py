import pandas as pd
import os

# Diccionarios de estandarización
unit_mapping = {
    'kg': 'kg', 'kilos': 'kg', 'kilogramo': 'kg',
    'lt': 'L', 'lts': 'L', 'litro': 'L',
    'mililitros': 'mL', 'ml': 'mL'
}

brand_mapping = {
    'techimport co.': 'TechImport',
    'tecnologia, s.a.': 'Tecnología S.A.',
    'peritech': 'PeriTech',
    'mundo hogar"" inc.': 'Mundo Hogar'
}

category_mapping = {
    'electrónica': 'Electrónica',
    'elect': 'Electrónica',
    'hogar': 'Hogar',
    'tecnologia': 'Tecnología'
}


def standardize_column(df, column, mapping):
    df[column] = df[column].astype(str).str.lower().map(mapping).fillna(df[column])
    return df


def apply_standardization(df):
    if 'Proveedor' in df.columns:
        df = standardize_column(df, 'Proveedor', brand_mapping)
    if 'Categoría' in df.columns:
        df = standardize_column(df, 'Categoría', category_mapping)
    return df


if __name__ == "__main__":
    # Construir ruta al archivo CSV
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "processed", "testlim.csv")

    # Leer CSV
    df = pd.read_csv(csv_path)

    print("🔎 Columnas disponibles:")
    print(df.columns)

    print("\n🔎 ANTES:")
    print(df[['Categoría', 'Proveedor']])

    # Aplicar estandarización
    df_cleaned = apply_standardization(df)

    print("\n✅ DESPUÉS DE STANDARDIZER:")
    print(df_cleaned[['Categoría', 'Proveedor']])
