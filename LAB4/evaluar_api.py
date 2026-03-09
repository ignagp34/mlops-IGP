"""
Evaluacion del modelo desplegado via API de MLflow sobre test set.
"""
import argparse
import os
import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelo via MLflow API")
    parser.add_argument("--port", type=int, default=5001, help="Puerto del servidor MLflow")
    args = parser.parse_args()

    # Cargar test set guardado
    test_path = os.path.join(os.path.dirname(__file__), "test_set.csv")
    test_df = pd.read_csv(test_path)
    y_test = test_df["Outcome"]
    X_test = test_df.drop("Outcome", axis=1)

    # Llamar a la API
    url = f"http://127.0.0.1:{args.port}/invocations"
    payload = {
        "dataframe_split": X_test.to_dict(orient="split")
    }

    print(f"Enviando {len(X_test)} muestras a {url}...")
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return

    predictions = response.json().get("predictions", response.json())
    y_pred = pd.Series(predictions)

    # Metricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print("METRICAS SOBRE TEST SET (via API)")
    print(f"{'='*50}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print(f"\nMatriz de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No diabetes", "Diabetes"]))


if __name__ == "__main__":
    main()
