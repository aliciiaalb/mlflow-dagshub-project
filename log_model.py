import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Charger des données d'exemple
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Modèle simple
model = LogisticRegression()
model.fit(X_train, y_train)

# Logger le modèle dans un répertoire "model"
mlflow.sklearn.save_model(model, path="model")
print("✅ Modèle enregistré dans le dossier 'model/'")
