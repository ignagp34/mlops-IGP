"""
Experimento de prueba: LogisticRegression sobre PIMA Diabetes con MLflow.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# Cargar datos
data_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
df = pd.read_csv(data_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split 80/20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: escalado + regresion logistica
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

# MLflow - usar ruta corta para evitar problemas con caracteres especiales
tracking_uri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
os.makedirs(tracking_uri, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{tracking_uri.replace(os.sep, '/')}")
mlflow.set_experiment("diabetes-prueba")

with mlflow.start_run(run_name="logistic-regression-baseline"):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parametros
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Log metricas
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1", f1)

    # Log modelo y registrar
    mlflow.sklearn.log_model(
        pipe,
        artifact_path="model",
        registered_model_name="diabetes-prueba-lr",
        input_example=X_test.iloc[:1]
    )

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\nRun ID: {mlflow.active_run().info.run_id}")
    print("Modelo registrado como 'diabetes-prueba-lr'")
