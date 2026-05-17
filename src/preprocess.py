"""
preprocess.py — Chargement et prétraitement du Restaurant Revenue Dataset
==========================================================================
Ce module expose trois fonctions publiques :
  - load_data()         : charge le CSV et retourne un DataFrame Pandas
  - preprocess(df)      : nettoie, encode, impute, normalise et divise les données
  - get_feature_names() : retourne la liste des noms de features après encodage

Utilisation typique :
    from src.preprocess import load_data, preprocess, get_feature_names
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Configuration du logger ──────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Chemin vers le dataset ────────────────────────────────────────────────────
DATA_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "restaurant_revenue_regression_realistic.csv",
)

# ── Colonnes du dataset ──────────────────────────────────────────────────────
NUMERIC_FEATURES: list[str] = [
    "daily_customers",       # Nombre de clients par jour
    "average_order_value",   # Valeur moyenne d'une commande (€)
    "marketing_budget",      # Budget marketing mensuel (€)
    "delivery_orders",       # Nombre de commandes en livraison
    "staff_count",           # Nombre d'employés
    "customer_rating",       # Note moyenne des clients (1-5)
    "social_media_score",    # Score de présence sur les réseaux sociaux (0-100)
]

CATEGORICAL_FEATURES: list[str] = [
    "restaurant_type",       # Type de restaurant (Cafe, Fast-food, etc.)
    "city",                  # Ville (Paris, Marseille, etc.)
]

TARGET_NAME: str = "monthly_revenue"  # Revenu mensuel du restaurant (€)

# ── Encodeurs et scaler partagés (remplis lors du preprocessing) ──────────────
_label_encoders: dict[str, LabelEncoder] = {}
_scaler: StandardScaler | None = None
_feature_names_after_encoding: list[str] = []


def get_feature_names() -> list[str]:
    """Retourne la liste des noms de features après encodage."""
    if _feature_names_after_encoding:
        return _feature_names_after_encoding
    # Avant le preprocessing, on retourne les noms bruts
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_label_encoders() -> dict[str, LabelEncoder]:
    """Retourne les LabelEncoders utilisés pour les colonnes catégorielles."""
    return _label_encoders


def get_scaler() -> StandardScaler | None:
    """Retourne le StandardScaler utilisé pour la normalisation."""
    return _scaler


def load_data() -> pd.DataFrame:
    """
    Charge le Restaurant Revenue Dataset depuis le fichier CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame contenant toutes les colonnes du dataset.
    """
    logger.info("Chargement du dataset Restaurant Revenue depuis %s…", DATA_PATH)
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(
                f"Fichier introuvable : {DATA_PATH}\n"
                "Vérifiez que le fichier CSV est bien dans le dossier data/"
            )
        df = pd.read_csv(DATA_PATH)
        logger.info(
            "Dataset chargé avec succès — %d lignes, %d colonnes.",
            len(df),
            len(df.columns),
        )
        return df
    except Exception as exc:
        logger.error("Erreur lors du chargement du dataset : %s", exc)
        raise


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Nettoie, encode, impute, normalise et divise le DataFrame en train/test.

    Étapes appliquées :
      1. Suppression des doublons
      2. Encodage des colonnes catégorielles (LabelEncoder)
      3. Imputation des valeurs manquantes par la médiane (SimpleImputer)
      4. Normalisation des features avec StandardScaler
      5. Séparation en X_train, X_test, y_train, y_test

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu de load_data().
    test_size : float
        Proportion des données allouées au test (défaut 0.2).
    random_state : int
        Graine aléatoire pour la reproductibilité (défaut 42).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    global _label_encoders, _scaler, _feature_names_after_encoding

    logger.info("Début du prétraitement des données…")

    # 1. Suppression des doublons
    nb_avant = len(df)
    df = df.drop_duplicates()
    nb_apres = len(df)
    if nb_avant != nb_apres:
        logger.info("%d doublon(s) supprimé(s).", nb_avant - nb_apres)
    else:
        logger.info("Aucun doublon détecté.")

    # 2. Vérification de la colonne cible
    if TARGET_NAME not in df.columns:
        raise ValueError(
            f"La colonne cible '{TARGET_NAME}' est absente du DataFrame. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # 3. Encodage des colonnes catégorielles avec LabelEncoder
    df_encoded = df.copy()
    _label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            _label_encoders[col] = le
            logger.info("Colonne '%s' encodée — %d catégories.", col, len(le.classes_))

    # 4. Séparation features / cible
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available_features = [c for c in all_features if c in df_encoded.columns]
    _feature_names_after_encoding = available_features

    X: pd.DataFrame = df_encoded[available_features].copy()
    y: pd.Series = df_encoded[TARGET_NAME].copy()

    # 5. Gestion des valeurs manquantes (imputation par la médiane)
    nb_manquants = X.isnull().sum().sum()
    if nb_manquants > 0:
        logger.info("%d valeur(s) manquante(s) détectée(s) — imputation par la médiane.", nb_manquants)
    else:
        logger.info("Aucune valeur manquante dans les features.")

    imputer = SimpleImputer(strategy="median")
    X_imputed: np.ndarray = imputer.fit_transform(X)

    # 6. Normalisation (StandardScaler : moyenne=0, écart-type=1)
    _scaler = StandardScaler()
    X_scaled: np.ndarray = _scaler.fit_transform(X_imputed)
    logger.info("Normalisation StandardScaler appliquée.")

    # 7. Division train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y.to_numpy(),
        test_size=test_size,
        random_state=random_state,
    )
    logger.info(
        "Division train/test effectuée — train : %d, test : %d.",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


# ── Point d'entrée rapide pour tester le module seul ─────────────────────────
if __name__ == "__main__":
    df = load_data()
    print("Aperçu du DataFrame :")
    print(df.head())
    print(f"\nDimensions : {df.shape}")
    print(f"\nStatistiques descriptives :\n{df.describe().round(2)}")
    print(f"\nTypes de colonnes :\n{df.dtypes}")
    print(f"\nValeurs uniques (catégorielles) :")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            print(f"  {col}: {df[col].unique().tolist()}")

    X_train, X_test, y_train, y_test = preprocess(df)
    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
