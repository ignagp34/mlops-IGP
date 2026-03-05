"""
app-iris-ct: Continuous Training extension for ML-FastAPI-Docker
================================================================
Extends app-iris with:
  - POST /train      -> reentrenamiento incremental con nuevas muestras
  - GET  /model/info -> version activa, metricas, historial
  - POST /predict    -> inferencia (igual que app-iris, con version activa)
  - GET  /health     -> estado del servicio
"""

import json
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuracion de rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model_active.joblib"
HISTORY_PATH = MODELS_DIR / "training_history.json"

# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

class IrisSample(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Longitud del sepalo (cm)")
    sepal_width: float  = Field(..., example=3.5, description="Anchura del sepalo (cm)")
    petal_length: float = Field(..., example=1.4, description="Longitud del petalo (cm)")
    petal_width: float  = Field(..., example=0.2, description="Anchura del petalo (cm)")


class LabeledSample(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float  = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float  = Field(..., example=0.2)
    label: int = Field(..., ge=0, le=2, example=0,
                       description="0=setosa, 1=versicolor, 2=virginica")


class ActivationPolicyType(str, Enum):
    any_improvement = "any_improvement"
    min_delta = "min_delta"
    per_class_f1 = "per_class_f1"


class ActivationPolicy(BaseModel):
    type: ActivationPolicyType = Field(
        default=ActivationPolicyType.any_improvement,
        description="Tipo de politica de activacion"
    )
    min_delta: float = Field(
        default=0.01,
        ge=0.0, le=1.0,
        description="Delta minimo de mejora en accuracy (solo para politica min_delta)"
    )
    target_class: int = Field(
        default=0,
        ge=0, le=2,
        description="Clase objetivo cuyo F1 debe mejorar (solo para politica per_class_f1)"
    )


class TrainRequest(BaseModel):
    samples: List[LabeledSample] = Field(
        ..., min_items=5,
        description="Nuevas muestras etiquetadas para reentrenamiento (minimo 5)"
    )
    retrain_from_scratch: bool = Field(
        False,
        description="Si True, ignora datos anteriores y entrena solo con las muestras enviadas"
    )
    activation_policy: ActivationPolicy = Field(
        default_factory=ActivationPolicy,
        description="Politica de activacion del modelo (por defecto: any_improvement)"
    )


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    model_version: str


class TrainResponse(BaseModel):
    status: str
    model_version: str
    accuracy_new: float
    accuracy_previous: Optional[float]
    model_updated: bool
    message: str
    activation_policy: str = "any_improvement"
    gate_reason: str = ""


class ModelInfo(BaseModel):
    active_version: str
    trained_at: str
    accuracy: float
    n_training_samples: int
    algorithm: str
    history: List[dict]


# ---------------------------------------------------------------------------
# Utilidades de persistencia
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def load_history() -> List[dict]:
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def get_active_model_meta() -> Optional[dict]:
    history = load_history()
    return history[-1] if history else None


# ---------------------------------------------------------------------------
# Quality gate: evalua si el nuevo modelo debe activarse
# ---------------------------------------------------------------------------

def evaluate_quality_gate(
    policy: ActivationPolicy,
    accuracy_new: float,
    accuracy_previous: Optional[float],
    y_val: np.ndarray,
    y_pred_new: np.ndarray,
    y_val_prev: Optional[np.ndarray] = None,
    y_pred_prev: Optional[np.ndarray] = None,
) -> tuple:
    """Evalua la politica de activacion. Devuelve (should_activate, reason)."""
    if accuracy_previous is None:
        return True, "Primer modelo, se activa automaticamente"

    if policy.type == ActivationPolicyType.any_improvement:
        passed = accuracy_new >= accuracy_previous
        reason = (
            f"Accuracy {accuracy_new:.4f} >= anterior ({accuracy_previous:.4f})"
            if passed else
            f"Accuracy {accuracy_new:.4f} < anterior ({accuracy_previous:.4f})"
        )

    elif policy.type == ActivationPolicyType.min_delta:
        threshold = accuracy_previous + policy.min_delta
        passed = accuracy_new >= threshold
        reason = (
            f"Accuracy {accuracy_new:.4f} >= umbral ({threshold:.4f} = "
            f"{accuracy_previous:.4f} + delta {policy.min_delta})"
            if passed else
            f"Accuracy {accuracy_new:.4f} < umbral ({threshold:.4f} = "
            f"{accuracy_previous:.4f} + delta {policy.min_delta})"
        )

    elif policy.type == ActivationPolicyType.per_class_f1:
        target = policy.target_class
        f1_new = f1_score(y_val, y_pred_new, labels=[0, 1, 2], average=None, zero_division=0.0)
        f1_new_class = float(f1_new[target])

        if y_val_prev is not None and y_pred_prev is not None:
            f1_prev = f1_score(y_val_prev, y_pred_prev, labels=[0, 1, 2], average=None, zero_division=0.0)
            f1_prev_class = float(f1_prev[target])
        else:
            f1_prev_class = 0.0

        passed = f1_new_class >= f1_prev_class
        reason = (
            f"F1 clase {target} ({CLASS_NAMES[target]}): "
            f"nuevo={f1_new_class:.4f} vs anterior={f1_prev_class:.4f}"
        )

    else:
        passed = accuracy_new >= accuracy_previous
        reason = "Politica desconocida, usando any_improvement por defecto"

    return passed, reason


# ---------------------------------------------------------------------------
# Bootstrap: si no existe modelo, lo entrenamos con el dataset original
# ---------------------------------------------------------------------------

def bootstrap_model():
    """Entrena un modelo base con el dataset Iris completo al arrancar."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))

    version = "v1.0-base"
    joblib.dump(clf, MODEL_PATH)

    history = [{
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": round(accuracy, 4),
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": "bootstrap (iris dataset completo)"
    }]
    save_history(history)
    print(f"[bootstrap] Modelo base creado -> version={version}, accuracy={accuracy:.4f}")


# ---------------------------------------------------------------------------
# App FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Continuous Training API",
    description=(
        "Extension MLOps de app-iris. Sirve predicciones y permite reentrenar "
        "el modelo con nuevas muestras etiquetadas, registrando el historial de versiones."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    if not MODEL_PATH.exists():
        bootstrap_model()
    else:
        meta = get_active_model_meta()
        if meta:
            print(f"[startup] Modelo activo cargado -> version={meta['version']}, "
                  f"accuracy={meta['accuracy']}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health():
    meta = get_active_model_meta()
    return {
        "status": "ok",
        "active_model_version": meta["version"] if meta else "none",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inferencia"])
def predict(sample: IrisSample):
    """
    Realiza una prediccion con el modelo activo.
    Devuelve la clase predicha, su nombre y la version del modelo usado.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Modelo no disponible. Llama primero a /train.")

    clf = joblib.load(MODEL_PATH)
    X = np.array([[
        sample.sepal_length,
        sample.sepal_width,
        sample.petal_length,
        sample.petal_width
    ]])
    pred = int(clf.predict(X)[0])
    meta = get_active_model_meta()

    return PredictResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        model_version=meta["version"] if meta else "unknown"
    )


@app.post("/train", response_model=TrainResponse, tags=["Entrenamiento"])
def train(request: TrainRequest):
    """
    Reentrena el modelo con las nuevas muestras enviadas.

    - Si retrain_from_scratch=False (por defecto), las nuevas muestras se anaden
      al dataset de entrenamiento anterior (si existe) y se reentrena sobre el total.
    - Si retrain_from_scratch=True, solo se usan las muestras enviadas.
    - El nuevo modelo reemplaza al activo solo si supera la politica de activacion.
    - Cada entrenamiento queda registrado en el historial aunque no se active.

    Politicas de activacion disponibles:
    - any_improvement: accuracy_new >= accuracy_previous (por defecto)
    - min_delta: accuracy_new >= accuracy_previous + min_delta
    - per_class_f1: F1 de la clase objetivo debe mejorar
    """

    # 1. Preparar nuevas muestras
    new_X = np.array([[s.sepal_length, s.sepal_width, s.petal_length, s.petal_width]
                       for s in request.samples])
    new_y = np.array([s.label for s in request.samples])

    # 2. Recuperar accuracy del modelo activo
    history = load_history()
    previous_accuracy = history[-1]["accuracy"] if history else None

    # 3. Construir dataset de entrenamiento
    data_file = MODELS_DIR / "accumulated_data.joblib"

    if not request.retrain_from_scratch and data_file.exists():
        saved = joblib.load(data_file)
        X_train = np.vstack([saved["X"], new_X])
        y_train = np.concatenate([saved["y"], new_y])
        source = f"incremental (+{len(new_X)} muestras nuevas, {len(saved['X'])} anteriores)"
    else:
        X_train, y_train = new_X, new_y
        source = f"desde cero ({len(new_X)} muestras)"

    # 4. Necesitamos al menos 2 clases para entrenar
    if len(np.unique(y_train)) < 2:
        raise HTTPException(
            status_code=422,
            detail="El dataset de entrenamiento debe contener al menos 2 clases distintas."
        )

    # 5. Entrenar nuevo modelo
    clf_new = LogisticRegression(max_iter=300, random_state=42)

    # Evaluacion: si hay suficientes datos, usamos split; si no, evaluamos en train
    if len(X_train) >= 20:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        clf_new.fit(X_tr, y_tr)
        y_pred_new = clf_new.predict(X_val)
        accuracy_new = float(accuracy_score(y_val, y_pred_new))
        eval_note = f"validacion con {len(X_val)} muestras"
    else:
        clf_new.fit(X_train, y_train)
        X_val, y_val = X_train, y_train
        y_pred_new = clf_new.predict(X_val)
        accuracy_new = float(accuracy_score(y_val, y_pred_new))
        eval_note = "evaluacion en train (dataset pequeno, < 20 muestras)"

    accuracy_new = round(accuracy_new, 4)

    # 6. Evaluar quality gate con la politica seleccionada
    y_pred_prev = None
    y_val_prev = None
    if request.activation_policy.type == ActivationPolicyType.per_class_f1 and MODEL_PATH.exists():
        clf_prev = joblib.load(MODEL_PATH)
        y_pred_prev = clf_prev.predict(X_val)
        y_val_prev = y_val

    model_updated, gate_reason = evaluate_quality_gate(
        policy=request.activation_policy,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        y_val=y_val,
        y_pred_new=y_pred_new,
        y_val_prev=y_val_prev,
        y_pred_prev=y_pred_prev,
    )

    version = f"v{len(history) + 1}.0-{uuid.uuid4().hex[:6]}"
    status = "activado" if model_updated else "rechazado"

    if model_updated:
        joblib.dump(clf_new, MODEL_PATH)
        joblib.dump({"X": X_train, "y": y_train}, data_file)
        message = f"Nuevo modelo activado. {gate_reason}"
    else:
        message = (
            f"Modelo NO activado. {gate_reason}. "
            "El modelo activo se mantiene sin cambios."
        )

    # 7. Registrar en historial
    history.append({
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "accuracy": accuracy_new,
        "n_training_samples": len(X_train),
        "algorithm": "LogisticRegression",
        "source": source,
        "eval_note": eval_note,
        "status": status,
        "activated": model_updated,
        "activation_policy": request.activation_policy.type.value,
        "gate_reason": gate_reason,
    })
    save_history(history)

    return TrainResponse(
        status=status,
        model_version=version,
        accuracy_new=accuracy_new,
        accuracy_previous=previous_accuracy,
        model_updated=model_updated,
        message=message,
        activation_policy=request.activation_policy.type.value,
        gate_reason=gate_reason,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
def model_info():
    """
    Devuelve informacion del modelo activo y el historial completo de entrenamientos.
    """
    history = load_history()
    if not history:
        raise HTTPException(status_code=404, detail="No hay ningun modelo entrenado aun.")

    active = history[-1]
    # El modelo activo es el ultimo con activated=True (o el primero si es bootstrap)
    active_entries = [h for h in history if h.get("activated", True)]
    active = active_entries[-1] if active_entries else history[-1]

    return ModelInfo(
        active_version=active["version"],
        trained_at=active["trained_at"],
        accuracy=active["accuracy"],
        n_training_samples=active["n_training_samples"],
        algorithm=active["algorithm"],
        history=history
    )


@app.delete("/model/history", tags=["Modelo"])
def reset_history():
    """
    [CUIDADO] Elimina el historial y el modelo activo. Fuerza bootstrap en el proximo arranque.
    Util para pruebas y demostracion.
    """
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    data_file = MODELS_DIR / "accumulated_data.joblib"
    if data_file.exists():
        data_file.unlink()
    bootstrap_model()
    return {"status": "ok", "message": "Historial eliminado y modelo base restaurado."}
