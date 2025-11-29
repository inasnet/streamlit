import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

# Chargement des données
df = pd.read_csv("C:/Users/HP/Desktop/CHD_ streamlit_App/CHD.csv", sep=";")
df["famhist"] = df["famhist"].str.strip().str.lower()

# Séparation X/y
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=123, stratify=y
)

# Déterminer le nombre de composantes pour 90% variance
pca_tmp = Pipeline([
    ("preprocessor", preprocessor),
    ("pca", PCA()),
    ("model", LogisticRegression(max_iter=500))
])
pca_tmp.fit(X_train, y_train)
cumsum = np.cumsum(pca_tmp.named_steps["pca"].explained_variance_ratio_)
n_components = np.argmax(cumsum >= 0.9) + 1
print("Composantes nécessaires pour 90% variance :", n_components)

# Logistic Regression avec SMOTE et PCA
pipeline_lr_smote = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=123)),
    ("pca", PCA(n_components=n_components)),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=123))
])

pipeline_lr_smote.fit(X_train, y_train)

# Prédictions
y_pred = pipeline_lr_smote.predict(X_test)
y_pred_proba = pipeline_lr_smote.predict_proba(X_test)[:, 1]

print("Classification report :")
print(classification_report(y_test, y_pred))

# Afficher quelques probabilités
print("Exemple de probabilités :")
print(y_pred_proba[:10])

# Sauvegarde du modèle
pipeline_lr_smote.fit(X, y)  # entraînement final sur tout le dataset
joblib.dump(pipeline_lr_smote, "Model.pkl")
print("Modèle sauvegardé dans Model.pkl")

