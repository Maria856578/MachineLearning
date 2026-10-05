"""Funciones y constantes compartidas por el informe.

Centraliza la carga del dataset, la partición reproducible, los nombres de las
variables y la construcción de los pipelines. La partición, la semilla y la
ingeniería de variables son idénticas a las de la primera fase del proyecto, de
modo que los resultados de ambos informes son directamente comparables.
"""
import os
from pathlib import Path

# Evita que joblib intente contar núcleos físicos con utilidades no disponibles en Windows.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "covertype.csv"
DASHBOARD_DIR = PROJECT_ROOT / "data_dashboard"
TARGET = "Cover_Type"

# Configuración óptima seleccionada por validación cruzada en la primera fase del proyecto
# (rejilla de 12 combinaciones, F1 macro = 0,566 ± 0,013). Se reutiliza sin repetir la búsqueda.
BEST_PARAMS = {"engineered": True, "C": 1.0, "class_weight": None}

NUM_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILD_COLS = [f"Wilderness_Area_{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type_{i}" for i in range(1, 41)]
BIN_COLS = WILD_COLS + SOIL_COLS
FEATURES = NUM_COLS + BIN_COLS

CLASS_NAMES = {
    1: "Abeto/pícea (Spruce/Fir)",
    2: "Pino lodgepole",
    3: "Pino ponderosa",
    4: "Álamo/sauce (Cottonwood/Willow)",
    5: "Álamo temblón (Aspen)",
    6: "Abeto de Douglas",
    7: "Krummholz",
}
CLASS_SHORT = {
    1: "1 Abeto/pícea",
    2: "2 P. lodgepole",
    3: "3 P. ponderosa",
    4: "4 Álamo/sauce",
    5: "5 Á. temblón",
    6: "6 A. Douglas",
    7: "7 Krummholz",
}
WILD_NAMES = {
    1: "Rawah",
    2: "Neota",
    3: "Comanche Peak",
    4: "Cache la Poudre",
}

DISPLAY_NAMES = {
    "Elevation": "Elevación (m)",
    "Aspect": "Orientación (° azimut)",
    "Slope": "Pendiente (°)",
    "Horizontal_Distance_To_Hydrology": "Dist. horizontal a agua (m)",
    "Vertical_Distance_To_Hydrology": "Dist. vertical a agua (m)",
    "Horizontal_Distance_To_Roadways": "Dist. a carreteras (m)",
    "Hillshade_9am": "Sombreado 9 a. m. (0-255)",
    "Hillshade_Noon": "Sombreado mediodía (0-255)",
    "Hillshade_3pm": "Sombreado 3 p. m. (0-255)",
    "Horizontal_Distance_To_Fire_Points": "Dist. a puntos de incendio (m)",
}

CLASS_PALETTE = {
    1: "#1b6f5c",
    2: "#4c9a2a",
    3: "#c8a13a",
    4: "#b0413e",
    5: "#e07b39",
    6: "#5b6fb3",
    7: "#8e5ea2",
}

LOG_COLS = [
    "Horizontal_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
]


def load_covertype():
    """Carga el CSV original y devuelve predictoras y objetivo."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return df, X, y


def split_covertype(X, y):
    """Partición estratificada 80/20 fijada con la semilla del proyecto."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def stratified_sample(X, y, n, random_state=RANDOM_STATE):
    """Submuestra estratificada de tamaño n (se usa solo por costo computacional)."""
    if n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=random_state)
    return Xs, ys


def wilderness_label(X):
    """Nombre del área silvestre de cada fila a partir de las columnas indicadoras."""
    idx = X[WILD_COLS].to_numpy().argmax(axis=1) + 1
    return pd.Series(idx, index=X.index).map(WILD_NAMES)


def soil_label(X):
    """Número del tipo de suelo de cada fila a partir de las columnas indicadoras."""
    return pd.Series(X[SOIL_COLS].to_numpy().argmax(axis=1) + 1, index=X.index)


def engineer_features(X):
    """Ingeniería de variables sin parámetros aprendidos (no genera fuga).

    - Orientación circular: seno y coseno del azimut.
    - log1p en las distancias horizontales, que tienen cola derecha larga.
    - Distancia euclidiana al agua superficial más cercana.
    - Elevación relativa al agua: elevación menos la distancia vertical.
    """
    X = X.copy()
    rad = np.deg2rad(X["Aspect"])
    X["Aspect_sin"] = np.sin(rad)
    X["Aspect_cos"] = np.cos(rad)
    X["Dist_Hydrology_Euclid"] = np.hypot(
        X["Horizontal_Distance_To_Hydrology"], X["Vertical_Distance_To_Hydrology"]
    )
    X["Elevation_Minus_VDH"] = X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    for col in LOG_COLS:
        X[col] = np.log1p(X[col])
    return X.drop(columns="Aspect")


ENG_NUM_COLS = [c for c in NUM_COLS if c != "Aspect"] + [
    "Aspect_sin",
    "Aspect_cos",
    "Dist_Hydrology_Euclid",
    "Elevation_Minus_VDH",
]


def build_preprocessor(engineered=False):
    """Preprocesamiento: escalado de numéricas e indicadoras sin cambios."""
    if engineered:
        scale = ColumnTransformer(
            [("num", StandardScaler(), ENG_NUM_COLS), ("bin", "passthrough", BIN_COLS)]
        )
        return Pipeline(
            [
                ("features", FunctionTransformer(engineer_features)),
                ("scale", scale),
            ]
        )
    return ColumnTransformer(
        [("num", StandardScaler(), NUM_COLS), ("bin", "passthrough", BIN_COLS)]
    )


def build_logreg(engineered=True, C=1.0, class_weight=None, max_iter=2000):
    """Pipeline completo del modelo base."""
    return Pipeline(
        [
            ("prep", build_preprocessor(engineered)),
            (
                "clf",
                LogisticRegression(
                    C=C, class_weight=class_weight, max_iter=max_iter, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
