"""
train.py — Entraînement et tracking MLflow des modèles de régression
=====================================================================
Ce script :
  1. Charge les variables d'environnement depuis .env
  2. Se connecte à DagsHub via dagshub.init()
  3. Entraîne 5 modèles de régression avec suivi MLflow complet
  4. Logue paramètres, métriques et artefacts pour chaque modèle
  5. Affiche un tableau récapitulatif des performances

Lancement :
    python src/train.py
"""

import logging
import os
import sys

# Essayer de configurer sys.stdout et sys.stderr en UTF-8 pour éviter les erreurs d'encodage sur Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Ajouter la racine du projet au PYTHONPATH pour les imports relatifs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.preprocess import load_data, preprocess

# ── Chargement des variables d'environnement ─────────────────────────────────
load_dotenv()

# ── Configuration du logger ──────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Afficher aussi dans la console lors de l'entraînement
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
)
logger = logging.getLogger(__name__)
logger.addHandler(console_handler)

# ── Lecture des variables d'environnement ─────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
DAGSHUB_USERNAME: str = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_TOKEN: str = os.getenv("DAGSHUB_TOKEN", "")
MODEL_NAME: str = os.getenv("MODEL_NAME", "RestaurantRevenueModel")

# Validation basique des variables obligatoires
_missing = [
    name
    for name, val in {
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "DAGSHUB_USERNAME": DAGSHUB_USERNAME,
        "DAGSHUB_TOKEN": DAGSHUB_TOKEN,
    }.items()
    if not val
]
if _missing:
    raise EnvironmentError(
        f"Variables d'environnement manquantes : {', '.join(_missing)}\n"
        "Vérifiez votre fichier .env (copiez .env.example → .env et remplissez les valeurs)."
    )


# ── Connexion DagsHub ─────────────────────────────────────────────────────────
def _init_dagshub() -> None:
    """Initialise la connexion DagsHub et configure MLflow."""
    try:
        # L'URI est de la forme : https://dagshub.com/USERNAME/REPO_NAME.mlflow
        uri_path = MLFLOW_TRACKING_URI.replace("https://dagshub.com/", "").replace(".mlflow", "")
        _, repo_name = uri_path.split("/", 1)

        # Configurer les tokens d'API pour le mode Headless/Non-interactif
        os.environ["DAGSHUB_CLIENT_TOKEN"] = DAGSHUB_TOKEN
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

        # Ajouter le token de manière programmatique via le module d'authentification
        try:
            dagshub.auth.add_app_token(DAGSHUB_TOKEN)
        except Exception as auth_exc:
            logger.warning("Impossible d'ajouter le token via add_app_token : %s", auth_exc)

        dagshub.init(
            repo_owner=DAGSHUB_USERNAME,
            repo_name=repo_name,
            mlflow=True,
        )
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        logger.info("Connexion DagsHub etablie - repo : %s/%s", DAGSHUB_USERNAME, repo_name)
    except Exception as exc:
        logger.error("Echec de la connexion DagsHub : %s", exc)
        raise


# ── Définition des modèles à entraîner ───────────────────────────────────────
def _get_models() -> dict[str, object]:
    """Retourne un dictionnaire {nom: instance du modèle}."""
    return {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
            eval_metric="rmse",
        ),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }


# ── Calcul des métriques de régression ───────────────────────────────────────
def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcule RMSE, MAE et R² entre les valeurs réelles et prédites."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


# ── Entraînement d'un modèle avec tracking MLflow ────────────────────────────
def _train_and_log(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    experiment_name: str,
) -> dict[str, float]:
    """Entraîne un modèle, calcule ses métriques et les enregistre dans MLflow."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        logger.info("  -> Entrainement de %s...", model_name)

        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métriques
        metrics = _compute_metrics(y_test, y_pred)

        # Log des hyperparamètres
        try:
            params = model.get_params()
            mlflow.log_params(params)
        except Exception as exc:
            logger.warning("Impossible de logguer les paramètres de %s : %s", model_name, exc)

        # Log des métriques
        mlflow.log_metrics(metrics)

        # Enregistrement du modèle dans le Model Registry MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{MODEL_NAME}_{model_name}",
        )

        logger.info(
            "  [OK] %s - RMSE=%.4f | MAE=%.4f | R2=%.4f",
            model_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )
        return metrics


# ── Pipeline principal d'entraînement ─────────────────────────────────────────
def main() -> None:
    """Point d'entrée principal : charge les données, entraîne tous les modèles."""
    logger.info("=" * 60)
    logger.info("DEMARRAGE DU PIPELINE D'ENTRAINEMENT")
    logger.info("=" * 60)

    # Initialisation DagsHub / MLflow
    _init_dagshub()

    # Chargement et prétraitement des données
    logger.info("Chargement et pretraitement du dataset...")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    logger.info(
        "Donnees pretes - X_train=%s | X_test=%s", X_train.shape, X_test.shape
    )

    # Entraînement de tous les modèles
    experiment_name = "Restaurant_Revenue_Regression"
    models = _get_models()
    results: list[dict] = []

    logger.info("\nEntrainement des %d modeles :", len(models))
    for model_name, model in models.items():
        try:
            metrics = _train_and_log(
                model_name=model_name,
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experiment_name=experiment_name,
            )
            results.append({"Modele": model_name, **metrics})
        except Exception as exc:
            logger.error("Echec de l'entrainement de %s : %s", model_name, exc)

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    if results:
        df_results = pd.DataFrame(results).sort_values("r2", ascending=False)
        df_results.index = range(1, len(df_results) + 1)
        print("\n" + "=" * 60)
        print("  RESULTATS COMPARATIFS DES MODELES")
        print("=" * 60)
        print(df_results.to_string(index=True))
        print("=" * 60)
        champion = df_results.iloc[0]
        print(
            f"\n[CHAMPION] Meilleur modele selon R2 : {champion['Modele']}"
            f" (R2={champion['r2']:.4f})"
        )
        print(
            "\n[INFO] Pour designer le champion, rendez-vous dans l'interface DagsHub :\n"
            "   Models -> selectionnez la version souhaitee -> Stage -> Production"
        )

    logger.info("Pipeline d'entrainement termine avec succes.")


if __name__ == "__main__":
    main()

