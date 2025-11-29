import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

print("Chargement des données...")
df = pd.read_csv("C:/Users/HP/Desktop/CHD_ streamlit_App/CHD.csv", sep=";")

# Nettoyage rapide
df["famhist"] = df["famhist"].str.strip().str.lower()

# Séparer X et y
X = df.drop("chd", axis=1)
y = df["chd"]

# Colonnes numériques et catégorielles
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = ["famhist"]

# Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=123, stratify=y
)

# -----------------------------
#  Logistic Regression avec PCA
# -----------------------------
print("Entraînement Logistic Regression avec PCA...")

# Déterminer le nombre de composantes pour 90% de variance
pca_tmp = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA()),
    ("model", LogisticRegression(max_iter=500))
])
pca_tmp.fit(X_train, y_train)
cumsum = np.cumsum(pca_tmp.named_steps["pca"].explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.9) + 1
print("Composantes nécessaires pour 90% variance :", n_components)

pipeline_lr_pca = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA(n_components=n_components)),
    ("model", LogisticRegression(max_iter=500))
])
pipeline_lr_pca.fit(X_train, y_train)
y_pred = pipeline_lr_pca.predict(X_test)
print("Résultats Logistic Regression avec PCA :")
print(classification_report(y_test, y_pred))

# -----------------------------
#  Logistic Regression sans PCA
# -----------------------------
print("Entraînement Logistic Regression sans PCA...")
pipeline_lr_no_pca = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=500))
])
pipeline_lr_no_pca.fit(X_train, y_train)
y_pred_no_pca = pipeline_lr_no_pca.predict(X_test)
print("Résultats Logistic Regression sans PCA :")
print(classification_report(y_test, y_pred_no_pca))

# -----------------------------
#  KNN avec SMOTE et GridSearch
# -----------------------------
print("Entraînement KNN avec SMOTE (rapide pour test)...")

pipeline_knn = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=123)),
    ("pca", PCA(n_components=min(n_components, 5))),  # limite les composants pour accélérer
    ("knn", KNeighborsClassifier())
])

params = {"knn__n_neighbors": [3, 5]}  # moins de valeurs pour tester rapidement

grid = GridSearchCV(
    pipeline_knn,
    param_grid=params,
    cv=3,       # cross-validation plus rapide
    scoring="f1",
    n_jobs=-1   # utilise tous les cœurs CPU
)

grid.fit(X_train, y_train)
y_pred_knn = grid.predict(X_test)
print("Résultats KNN avec SMOTE :")
print(classification_report(y_test, y_pred_knn))

# -----------------------------
#  Sauvegarde du meilleur modèle
# -----------------------------
print("Sélection et sauvegarde du meilleur modèle...")
best_model = grid.best_estimator_
best_model.fit(X, y)  # entraînement final sur tout le dataset
joblib.dump(best_model, "Model.pkl")
print("Modèle sauvegardé dans Model.pkl ")
