"""
Experimentos completos: Grid Search con criterio medico y registro en MLflow.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

# ---------- Cargar datos ----------
data_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
df = pd.read_csv(data_path)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ---------- Splits ----------
# 1er split: 90% train_val / 10% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
# 2do split: ~77.8% train / ~22.2% val (del 90%) => 70/20 del total
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.222, random_state=42, stratify=y_train_val
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Guardar test set para evaluacion posterior via API
test_df = X_test.copy()
test_df["Outcome"] = y_test.values
test_df.to_csv(os.path.join(os.path.dirname(__file__), "test_set.csv"), index=False)
print("Test set guardado en test_set.csv")

# ---------- Grids de hiperparametros ----------
model_configs = {
    "LogisticRegression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "param_grid": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear"],
            "clf__class_weight": [None, "balanced"],
        }
    },
    "RandomForest": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "param_grid": {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [3, 5, 10, None],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": [None, "balanced"],
        }
    },
    "GradientBoosting": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        "param_grid": {
            "clf__n_estimators": [50, 100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5],
        }
    },
}

# ---------- Grid Search + MLflow ----------
tracking_uri = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mlruns")
os.makedirs(tracking_uri, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{tracking_uri.replace(os.sep, '/')}")
mlflow.set_experiment("diabetes-completo")

results = []

for model_name, config in model_configs.items():
    print(f"\n{'='*60}")
    print(f"Grid Search: {model_name}")
    print(f"{'='*60}")

    grid = GridSearchCV(
        config["pipeline"],
        config["param_grid"],
        scoring="recall",
        cv=5,
        n_jobs=-1,
        refit=False
    )
    grid.fit(X_train, y_train)

    # Top-3 configs por recall en CV
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results = cv_results.sort_values("mean_test_score", ascending=False)
    top3 = cv_results.head(3)

    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        params = row["params"]
        print(f"\n  Top-{rank}: {params}")

        # Reconstruir pipeline con los mejores params
        pipe = config["pipeline"].__class__(config["pipeline"].steps)
        pipe.set_params(**params)

        # Reentrenar en train+val
        X_tv = pd.concat([X_train, X_val])
        y_tv = pd.concat([y_train, y_val])
        pipe.fit(X_tv, y_tv)

        # Evaluar en val (para comparacion)
        y_val_pred = pipe.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)

        # Extraer params relevantes (sin prefijo clf__)
        clean_params = {k.replace("clf__", ""): v for k, v in params.items()}

        result = {
            "model": model_name,
            "rank": rank,
            "params": clean_params,
            "cv_recall": row["mean_test_score"],
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "pipeline": pipe,
        }
        results.append(result)

        # Log en MLflow
        run_name = f"{model_name}-top{rank}"
        with mlflow.start_run(run_name=run_name):
            for k, v in clean_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("rank_cv", rank)

            mlflow.log_metric("cv_recall", row["mean_test_score"])
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_precision", val_prec)
            mlflow.log_metric("val_recall", val_rec)
            mlflow.log_metric("val_f1", val_f1)

            mlflow.sklearn.log_model(
                pipe,
                artifact_path="model",
                input_example=X_val.iloc[:1]
            )

        print(f"    CV Recall: {row['mean_test_score']:.4f}")
        print(f"    Val -> Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, "
              f"Rec: {val_rec:.4f}, F1: {val_f1:.4f}")

# ---------- Seleccion con criterio medico ----------
print(f"\n{'='*60}")
print("SELECCION DE MODELOS (criterio medico)")
print(f"{'='*60}")
print("Primario: Recall >= 0.65 (detectar diabeticos)")
print("Secundario: F1 (equilibrar precision)")
print("Desempate: Accuracy")

# Filtrar por recall >= 0.65
candidates = [r for r in results if r["val_recall"] >= 0.65]
if not candidates:
    print("\nNingun modelo alcanza Recall >= 0.65. Relajando umbral...")
    candidates = sorted(results, key=lambda x: x["val_recall"], reverse=True)[:3]

# Ordenar: recall desc, f1 desc, accuracy desc
candidates.sort(key=lambda x: (-x["val_recall"], -x["val_f1"], -x["val_accuracy"]))

# Registrar top 2-3 modelos
n_register = min(3, len(candidates))
print(f"\nModelos seleccionados para registro ({n_register}):\n")
print(f"{'Modelo':<25} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Razon'}")
print("-" * 75)

for i, cand in enumerate(candidates[:n_register]):
    model_label = f"{cand['model']}-top{cand['rank']}"
    reason = "Mejor recall" if i == 0 else ("Mejor F1" if i == 1 else "Alternativa")

    print(f"{model_label:<25} {cand['val_recall']:>8.4f} {cand['val_f1']:>8.4f} "
          f"{cand['val_accuracy']:>8.4f} {reason}")

    # Registrar en Model Registry
    reg_name = f"diabetes-{cand['model'].lower()}"
    if i == 0:
        reg_name = "diabetes-best"

    with mlflow.start_run(run_name=f"registro-{model_label}"):
        for k, v in cand["params"].items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_type", cand["model"])
        mlflow.log_param("selection_reason", reason)
        mlflow.log_metric("val_recall", cand["val_recall"])
        mlflow.log_metric("val_f1", cand["val_f1"])
        mlflow.log_metric("val_accuracy", cand["val_accuracy"])
        mlflow.log_metric("val_precision", cand["val_precision"])

        mlflow.sklearn.log_model(
            cand["pipeline"],
            artifact_path="model",
            registered_model_name=reg_name,
            input_example=X_val.iloc[:1]
        )
    print(f"  -> Registrado como '{reg_name}'")

print(f"\n{'='*60}")
print("Experimentos completados. Ejecuta 'mlflow ui' para ver resultados.")
