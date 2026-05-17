"""
main.py — Application Streamlit : Restaurant Revenue Predictor
===============================================================
Lancement : streamlit run app/main.py
"""

import logging
import os
import sys

# ── Ajouter la racine du projet au PYTHONPATH ─────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
import google.generativeai as genai

from src.preprocess import (
    load_data,
    preprocess,
    get_feature_names,
    get_label_encoders,
    get_scaler,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_NAME
)

# ── Chargement des variables d'environnement ─────────────────────────────────
load_dotenv()

# ── Création du dossier logs ──────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

# ── Configuration du logger ───────────────────────────────────────────────────
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration globale Streamlit ──────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Revenue Predictor",
    layout="wide",
    page_icon="🍽️",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 1rem; border: 1px solid #0f3460;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════════════════
# Chargement du dataset, initialisation du scaler/encodeurs et chargement modèle
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def init_data_and_preprocessing():
    """Charge les données et initialise les encodeurs et scaler en mémoire."""
    df = load_data()
    # On appelle preprocess pour remplir _label_encoders et _scaler dans src.preprocess
    preprocess(df)
    return df

df_raw = init_data_and_preprocessing()
feature_names = get_feature_names()
label_encoders = get_label_encoders()
scaler = get_scaler()


def get_secret(key: str, default: str = "") -> str:
    """Récupère un secret depuis l'environnement ou st.secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

@st.cache_resource(show_spinner="Chargement du modèle champion…")
def load_champion():
    """Charge le modèle en Production depuis le MLflow Model Registry."""
    try:
        mlflow.set_tracking_uri(get_secret("MLFLOW_TRACKING_URI"))
        
        dagshub_user = get_secret("DAGSHUB_USERNAME")
        dagshub_token = get_secret("DAGSHUB_TOKEN")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        os.environ["MLFLOW_TRACKING_TOKEN"] = dagshub_token
        os.environ["DAGSHUB_CLIENT_TOKEN"] = dagshub_token
        
        model_name = get_secret("MODEL_NAME", "RestaurantRevenueModel")
        model_stage = get_secret("MODEL_STAGE", "Production")
        model_uri = f"models:/{model_name}/{model_stage}"
        
        logger.info(f"Tentative de chargement du modèle depuis URI : {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Modèle champion chargé avec succès.")
        return model, None
    except Exception as exc:
        logger.error("Impossible de charger le modèle champion : %s", exc)
        return None, str(exc)

# ════════════════════════════════════════════════════════════════════════════════
# ONGLETS
# ════════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🍽️ Présentation",
    "🎯 Prédiction unitaire",
    "📂 Prédiction batch CSV",
    "🤖 Analyse Gemini IA",
    "📊 Dashboard & Visualisation",
])

# ────────────────────────────────────────────────────────────────────────────────
# ONGLET 1 — Présentation
# ────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<h1 class="main-title">🍽️ Restaurant Revenue Predictor</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        **Bienvenue** sur cette application de prédiction du revenu mensuel de restaurants.

        > Le modèle prédit le **revenu mensuel** (`monthly_revenue`) en euros (€),
        > à partir de 9 caractéristiques (clients quotidiens, budget marketing, type, ville, etc.).
        """
    )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Modèles entraînés")
        modeles_df = pd.DataFrame({
            "Modèle": ["RandomForest", "XGBoost", "GradientBoosting", "Ridge", "LinearRegression"],
            "Hyperparamètres clés": [
                "n_estimators=200", "n_estimators=200, lr=0.1",
                "n_estimators=200", "alpha=1.0", "—",
            ],
            "Type": ["Ensemble", "Boosting", "Boosting", "Régularisée", "Linéaire"],
        })
        st.dataframe(modeles_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🏗️ Architecture de la solution")
        st.markdown(
            """
            ```
            Restaurant Revenue Dataset (CSV local)
                    ↓
            src/preprocess.py  → Nettoyage, encodage, imputation, scaling
                    ↓
            src/train.py       → 5 modèles + MLflow tracking
                    ↓
            DagsHub MLflow Registry (champion = Production)
                    ↓
            app/main.py (Streamlit) → Prédictions & IA
            ```
            """
        )

    st.divider()
    st.subheader("🔗 Liens utiles")
    dagshub_user = os.getenv("DAGSHUB_USERNAME", "USERNAME")
    st.markdown(
        f"- 🧪 [DagsHub MLflow](https://dagshub.com/{dagshub_user})\n"
        f"- 📦 [GitHub Repository](https://github.com/{dagshub_user})\n"
    )


# ────────────────────────────────────────────────────────────────────────────────
# ONGLET 2 — Prédiction unitaire
# ────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("🎯 Prédiction unitaire")
    st.markdown("Renseignez les caractéristiques du restaurant pour estimer son revenu mensuel.")

    model, err_msg = load_champion()
    if model is None:
        st.error(
            f"⚠️ Impossible de charger le modèle champion depuis MLflow.\n\n"
            f"**Erreur MLflow** : `{err_msg}`\n\n"
            "Vérifiez que :\n"
            "1. Les variables `.env` sont correctement renseignées\n"
            "2. Un modèle est bien en stage **Production** dans DagsHub\n"
            "3. Vous avez lancé `python src/train.py` au préalable"
        )
    else:
        st.success("✅ Modèle champion chargé avec succès.")

        with st.form("prediction_form"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                daily_customers = st.number_input("Clients par jour", min_value=0, value=150, step=10)
                average_order_value = st.number_input("Valeur moyenne cmd (€)", min_value=0.0, value=25.0, step=1.0)
                
            with col2:
                marketing_budget = st.number_input("Budget marketing (€)", min_value=0.0, value=1000.0, step=100.0)
                delivery_orders = st.number_input("Commandes livraison", min_value=0, value=50, step=5)

            with col3:
                staff_count = st.number_input("Nombre d'employés", min_value=1, value=10, step=1)
                customer_rating = st.slider("Note clients (1-5)", 1.0, 5.0, 4.2, 0.1)

            with col4:
                social_media_score = st.slider("Score réseaux sociaux", 0, 100, 60, 1)
                # Extraire les classes uniques pour les selectbox
                types_resto = df_raw["restaurant_type"].unique().tolist() if "restaurant_type" in df_raw.columns else ["Cafe", "Fast-food"]
                villes = df_raw["city"].unique().tolist() if "city" in df_raw.columns else ["Paris", "Marseille"]
                
                restaurant_type = st.selectbox("Type de restaurant", types_resto)
                city = st.selectbox("Ville", villes)

            submitted = st.form_submit_button("🔮 Prédire le revenu", use_container_width=True)

        if submitted:
            try:
                # 1. Créer un DataFrame avec une ligne
                input_data = {
                    "daily_customers": [daily_customers],
                    "average_order_value": [average_order_value],
                    "marketing_budget": [marketing_budget],
                    "delivery_orders": [delivery_orders],
                    "staff_count": [staff_count],
                    "customer_rating": [customer_rating],
                    "social_media_score": [social_media_score],
                    "restaurant_type": [restaurant_type],
                    "city": [city]
                }
                input_df = pd.DataFrame(input_data)
                
                # 2. Encodage des catégorielles
                for col in CATEGORICAL_FEATURES:
                    if col in label_encoders:
                        le = label_encoders[col]
                        # Gérer les valeurs non vues avec une valeur par défaut ou -1 (simplification)
                        if input_df[col].iloc[0] in le.classes_:
                            input_df[col] = le.transform(input_df[col].astype(str))
                        else:
                            # Valeur inconnue
                            input_df[col] = 0
                
                # 3. Aligner sur les colonnes d'entraînement
                input_df = input_df[feature_names]
                
                # 4. Scaling
                input_scaled = scaler.transform(input_df)

                # 5. Prédiction
                prediction = model.predict(input_scaled)[0]

                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("💰 Revenu mensuel estimé", f"{prediction:,.2f} €")
                col_res2.metric("👥 Clients estimés/mois", f"{daily_customers * 30:,.0f}")
                col_res3.metric("📈 Budget Marketing", f"{marketing_budget:,.2f} €")

                logger.info(
                    "Prédiction unitaire — prédiction=%.2f €", prediction
                )
                st.info(f"ℹ️ La prédiction est basée sur le modèle en stage **Production** du registry MLflow.")
            except Exception as exc:
                st.error(f"Erreur lors de la prédiction : {exc}")
                logger.error("Erreur prédiction unitaire : %s", exc)


# ────────────────────────────────────────────────────────────────────────────────
# ONGLET 3 — Prédiction batch CSV
# ────────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("📂 Prédiction batch CSV")
    st.markdown(
        "Importez un fichier CSV contenant les mêmes colonnes pour obtenir des prédictions en lot."
    )

    colonnes_attendues = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    st.info(
        f"**Colonnes attendues :** `{', '.join(colonnes_attendues)}`\n\n"
        "Le fichier doit avoir ces colonnes exactement (dans n'importe quel ordre)."
    )

    uploaded_file = st.file_uploader("📎 Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            logger.info("Fichier CSV uploadé — %d lignes", len(df_upload))

            st.subheader("Aperçu des données importées")
            st.dataframe(df_upload.head(), use_container_width=True)

            model_batch, err_msg_batch = load_champion()
            if model_batch is None:
                st.error(f"⚠️ Modèle non disponible. Vérifiez votre configuration MLflow.\nErreur : {err_msg_batch}")
            else:
                if st.button("🚀 Lancer les prédictions", use_container_width=True):
                    missing_cols = [c for c in colonnes_attendues if c not in df_upload.columns]
                    if missing_cols:
                        st.error(f"Colonnes manquantes dans le CSV : {missing_cols}")
                    else:
                        try:
                            # Prétraitement du batch
                            batch_encoded = df_upload.copy()
                            for col in CATEGORICAL_FEATURES:
                                if col in label_encoders:
                                    le = label_encoders[col]
                                    # Mapper les valeurs connues, mettre 0 pour les inconnues
                                    batch_encoded[col] = batch_encoded[col].astype(str).map(
                                        lambda s: le.transform([s])[0] if s in le.classes_ else 0
                                    )
                            
                            batch_encoded = batch_encoded[feature_names]
                            
                            # Imputation valeurs manquantes (médiane simple pour le batch)
                            batch_encoded = batch_encoded.fillna(batch_encoded.median())
                            
                            batch_scaled = scaler.transform(batch_encoded)
                            
                            preds = model_batch.predict(batch_scaled)
                            
                            df_result = df_upload.copy()
                            df_result["prediction_revenue"] = preds.round(2)

                            st.subheader("📋 Résultats des prédictions")
                            st.dataframe(df_result, use_container_width=True)

                            csv_out = df_result.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="⬇️ Télécharger les résultats (CSV)",
                                data=csv_out,
                                file_name="predictions_restaurant_revenue.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                            logger.info(
                                "Prédiction batch terminée — %d prédictions générées.", len(preds)
                            )
                        except Exception as exc:
                            st.error(f"Erreur lors des prédictions batch : {exc}")
                            logger.error("Erreur prédiction batch : %s", exc)
        except Exception as exc:
            st.error(f"Erreur lors de la lecture du CSV : {exc}")
            logger.error("Erreur lecture CSV : %s", exc)


# ────────────────────────────────────────────────────────────────────────────────
# ONGLET 4 — Analyse IA Générative Gemini
# ────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.title("🤖 Analyse IA Générative — Gemini")
    st.markdown(
        "Posez vos questions sur les revenus des restaurants, "
        "l'interprétation des prédictions ou l'amélioration du modèle."
    )

    gemini_api_key = get_secret("GEMINI_API_KEY", "")
    if not gemini_api_key or gemini_api_key == "ta_cle_gemini":
        st.warning(
            "⚠️ Clé API Gemini non configurée ou invalide. "
            "Ajoutez une vraie `GEMINI_API_KEY` dans votre fichier `.env`."
        )
    else:
        # Configuration Gemini
        genai.configure(api_key=gemini_api_key)

        SYSTEM_CONTEXT = (
            "Tu es un expert en gestion de restaurant et en Machine Learning. "
            "Aide l'utilisateur à interpréter les prédictions de revenu mensuel. "
            "Réponds en français, de façon claire et structurée."
        )

        st.subheader("💡 Questions suggérées")
        col_q1, col_q2, col_q3 = st.columns(3)
        question_preset = ""

        if col_q1.button("📈 Facteurs influents", use_container_width=True):
            question_preset = "Quels facteurs influencent le plus le revenu d'un restaurant ?"
        if col_q2.button("📊 Augmenter le revenu", use_container_width=True):
            question_preset = "Quelles stratégies marketing augmentent le panier moyen ?"
        if col_q3.button("🔧 Améliorer le modèle", use_container_width=True):
            question_preset = "Comment améliorer les prédictions d'un modèle XGBoost pour un restaurant ?"

        st.divider()
        user_question = st.text_area(
            "❓ Votre question",
            value=question_preset,
            height=120,
            placeholder="Ex : Quel est l'impact du score des réseaux sociaux sur le revenu ?",
        )

        if st.button("✨ Analyser avec Gemini", use_container_width=True):
            if not user_question.strip():
                st.warning("Veuillez saisir une question avant d'analyser.")
            else:
                try:
                    with st.spinner("Gemini analyse votre question…"):
                        gemini_model = genai.GenerativeModel(
                            model_name="gemini-1.5-flash",
                            system_instruction=SYSTEM_CONTEXT,
                        )
                        response = gemini_model.generate_content(user_question)
                        st.subheader("💬 Réponse de Gemini")
                        st.write(response.text)
                        logger.info("Question Gemini : %s", user_question[:200])
                except Exception as exc:
                    st.error(f"Erreur lors de l'appel à Gemini : {exc}")
                    logger.error("Erreur Gemini : %s", exc)


# ────────────────────────────────────────────────────────────────────────────────
# ONGLET 5 — Dashboard & Visualisation
# ────────────────────────────────────────────────────────────────────────────────
with tab5:
    st.title("📊 Dashboard & Visualisation")

    try:
        df_viz = df_raw.copy()

        # ── Métriques résumées ──────────────────────────────────────────────
        st.subheader("📈 Statistiques globales du Dataset")
        revenu = df_viz[TARGET_NAME]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💰 Revenu moyen", f"{revenu.mean():,.0f} €")
        m2.metric("📊 Revenu médian", f"{revenu.median():,.0f} €")
        m3.metric("⬇️ Revenu minimum", f"{revenu.min():,.0f} €")
        m4.metric("⬆️ Revenu maximum", f"{revenu.max():,.0f} €")

        st.divider()

        # ── Graphique 1 : histogramme distribution des revenus ──────────────
        st.subheader("📊 Distribution des revenus mensuels")
        fig_hist = px.histogram(
            df_viz,
            x=TARGET_NAME,
            nbins=50,
            color_discrete_sequence=["#fda085"],
            labels={TARGET_NAME: "Revenu mensuel (€)"},
            title="Distribution des revenus mensuels des restaurants",
            template="plotly_dark",
        )
        fig_hist.update_layout(bargap=0.05, title_font_size=16)
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Graphique 2 : scatter Clients vs Revenu ─────────────────────────
        st.subheader("👥 Clients quotidiens vs Revenu — coloré par Type")
        fig_scatter = px.scatter(
            df_viz,
            x="daily_customers",
            y=TARGET_NAME,
            color="restaurant_type" if "restaurant_type" in df_viz.columns else None,
            opacity=0.7,
            labels={
                "daily_customers": "Clients par jour",
                TARGET_NAME: "Revenu mensuel (€)",
                "restaurant_type": "Type de restaurant",
            },
            title="Relation entre le nombre de clients et le revenu",
            template="plotly_dark",
        )
        fig_scatter.update_layout(title_font_size=16)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── Graphique 3 : heatmap de corrélation ────────────────────────────
        st.subheader("🔥 Heatmap de corrélation")
        # On ne garde que les numériques pour la heatmap
        df_numeric = df_viz.select_dtypes(include=[np.number])
        corr_matrix = df_numeric.corr().round(2)
        fig_heat = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix.values,
                texttemplate="%{text}",
                textfont={"size": 11},
                hoverongaps=False,
            )
        )
        fig_heat.update_layout(
            title="Matrice de corrélation",
            title_font_size=16,
            template="plotly_dark",
            height=600,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    except Exception as exc:
        st.error(f"Erreur lors du chargement des visualisations : {exc}")
        logger.error("Erreur dashboard visualisation : %s", exc)
