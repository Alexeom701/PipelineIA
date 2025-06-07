import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# --- 1. Cargar y unir datasets ---
# Supón que tienes varios CSVs por categoría
archivos = ["moda_hombre.csv", "camaras.csv", "hogar.csv"]

dfs = []
for archivo in archivos:
    categoria = archivo.replace(".csv", "")
    df = pd.read_csv(archivo)
    df = df[["name"]]  # solo nos interesa la columna 'name'
    df["category"] = categoria  # agregar categoría desde el nombre del archivo
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# --- 2. Limpiar el texto ---
def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.strip()

print("Ejemplo antes:", df['name'].iloc[0])
df['name'] = df['name'].astype(str).apply(limpiar)
print("Ejemplo después:", df['name'].iloc[0])

# --- 3. Vectorización ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['name'])

# --- 4. Codificación de etiquetas ---
encoder = LabelEncoder()
y = encoder.fit_transform(df['category'])

# --- 5. Entrenamiento ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# --- 6. Evaluación ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# --- 7. Guardado ---
joblib.dump(clf, "modelo_categoria.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(encoder, "encoder.pkl")
print("Modelo guardado correctamente.")
