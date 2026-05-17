# 🍽️ Restaurant Revenue Predictor

Application MLOps complète de prédiction du revenu mensuel de restaurants, avec tracking d'expériences MLflow sur DagsHub et interface Streamlit.

---

## 🗂️ Structure du projet

```
machine-learning/
├── app/
│   └── main.py           # Application Streamlit (5 onglets)
├── src/
│   ├── preprocess.py     # Chargement & prétraitement du dataset
│   └── train.py          # Entraînement des modèles + tracking MLflow
├── data/
│   └── restaurant_revenue_regression_realistic.csv # Le dataset
├── logs/                 # Fichiers de logs (auto-créé)
├── .env                  # Variables d'environnement (ne pas committer)
├── .env.example          # Template des variables d'environnement
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Créer et activer l'environnement virtuel

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer les variables d'environnement

Votre fichier `.env` doit contenir (avec vos vraies valeurs) :
```env
MLFLOW_TRACKING_URI=https://dagshub.com/USERNAME/REPO_NAME.mlflow
DAGSHUB_USERNAME=ton_username
DAGSHUB_TOKEN=ton_token_dagshub
GEMINI_API_KEY=ta_cle_gemini
MODEL_NAME=RestaurantRevenueModel
MODEL_STAGE=Production
```

---

## 🌐 Créer un compte DagsHub et lier le repo

1. Créez un compte sur [dagshub.com](https://dagshub.com)
2. Créez un nouveau dépôt (bouton **New Repository**)
3. Dans votre repo DagsHub → **Remote** → copiez l'URI MLflow :
   ```
   https://dagshub.com/USERNAME/REPO_NAME.mlflow
   ```
4. Générez un token d'accès : **Settings → Access Tokens → New Token**

---

## 🚀 Ordre d'exécution

```bash
# Étape 1 — Vérifier le prétraitement (optionnel)
python src/preprocess.py

# Étape 2 — Entraîner les modèles et tracker avec MLflow
python src/train.py

# Étape 3 — Lancer l'application Streamlit
streamlit run app/main.py
```

---

## 🏆 Désigner le modèle Champion dans DagsHub

1. Connectez-vous à votre dépôt DagsHub
2. Cliquez sur **Experiments** → **Models**
3. Sélectionnez le modèle avec les meilleures métriques (R² le plus élevé)
4. Cliquez sur la version souhaitée → **Stage** → **Transition to Production**
5. Confirmez la transition
6. L'application Streamlit chargera ce modèle automatiquement.

---

## ☁️ Déploiement sur Streamlit Community Cloud

1. Poussez votre code sur GitHub.
2. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io).
3. Cliquez **New app** → sélectionnez votre repo GitHub.
4. Configurez :
   - **Branch** : `main`
   - **Main file path** : `app/main.py`
5. **Gestion des secrets** — cliquez sur **Advanced settings → Secrets** et collez le contenu de votre `.env` au format TOML.
6. Cliquez **Deploy**.

---

## 🤖 Modèles entraînés

| Modèle | Hyperparamètres clés |
|---|---|
| RandomForestRegressor | n_estimators=200, random_state=42 |
| LinearRegression | — |
| XGBRegressor | n_estimators=200, learning_rate=0.1 |
| GradientBoostingRegressor | n_estimators=200 |
| Ridge | alpha=1.0 |

---

## 📊 Dataset

- **Fichier** : `restaurant_revenue_regression_realistic.csv`
- **Taille** : ~800 observations
- **Features numériques** : clients par jour, valeur moyenne de commande, budget marketing, commandes livraison, employés, note client, score réseaux sociaux.
- **Features catégorielles** : type de restaurant, ville.
- **Cible** : `monthly_revenue` (revenu mensuel en €)
