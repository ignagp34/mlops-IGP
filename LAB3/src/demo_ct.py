#!/usr/bin/env python3
"""
demo_ct.py  -  Script de demostracion del flujo Continuous Training
====================================================================
Simula un ciclo MLOps completo:
  1. Consulta el modelo base (bootstrap)
  2. Hace predicciones
  3. Envia nuevas muestras correctamente etiquetadas -> modelo mejora
  4. Envia muestras con ruido/errores -> modelo empeora, NO se activa
  5. Prueba politica min_delta (requiere mejora minima)
  6. Prueba politica per_class_f1 (evalua F1 por clase)
  7. Muestra el historial final de versiones

Uso:
    python demo_ct.py [--host http://localhost:8000]
"""

import argparse
import json
import sys
import time

import requests

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Muestras de ejemplo
# ---------------------------------------------------------------------------

# 20 muestras bien etiquetadas (mezcla de las 3 clases)
GOOD_SAMPLES = [
    # setosa (clase 0)
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "label": 0},
    {"sepal_length": 4.7, "sepal_width": 3.2, "petal_length": 1.3, "petal_width": 0.2, "label": 0},
    {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2, "label": 0},
    {"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4, "label": 0},
    {"sepal_length": 4.6, "sepal_width": 3.4, "petal_length": 1.4, "petal_width": 0.3, "label": 0},
    {"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2, "label": 0},
    {"sepal_length": 4.4, "sepal_width": 2.9, "petal_length": 1.4, "petal_width": 0.2, "label": 0},
    # versicolor (clase 1)
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "label": 1},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5, "label": 1},
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5, "label": 1},
    {"sepal_length": 5.5, "sepal_width": 2.3, "petal_length": 4.0, "petal_width": 1.3, "label": 1},
    {"sepal_length": 6.5, "sepal_width": 2.8, "petal_length": 4.6, "petal_width": 1.5, "label": 1},
    {"sepal_length": 5.7, "sepal_width": 2.8, "petal_length": 4.5, "petal_width": 1.3, "label": 1},
    # virginica (clase 2)
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5, "label": 2},
    {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9, "label": 2},
    {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1, "label": 2},
    {"sepal_length": 6.3, "sepal_width": 2.9, "petal_length": 5.6, "petal_width": 1.8, "label": 2},
    {"sepal_length": 6.5, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2, "label": 2},
    {"sepal_length": 7.6, "sepal_width": 3.0, "petal_length": 6.6, "petal_width": 2.1, "label": 2},
    {"sepal_length": 4.9, "sepal_width": 2.5, "petal_length": 4.5, "petal_width": 1.7, "label": 2},
]

# 10 muestras con etiquetas incorrectas (simula ruido en los datos nuevos)
NOISY_SAMPLES = [
    {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "label": 2},
    {"sepal_length": 4.9, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.2, "label": 1},
    {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "label": 0},
    {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5, "label": 2},
    {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5, "label": 1},
    {"sepal_length": 5.8, "sepal_width": 2.7, "petal_length": 5.1, "petal_width": 1.9, "label": 0},
    {"sepal_length": 5.0, "sepal_width": 3.6, "petal_length": 1.4, "petal_width": 0.2, "label": 1},
    {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 4.9, "petal_width": 1.5, "label": 2},
    {"sepal_length": 7.1, "sepal_width": 3.0, "petal_length": 5.9, "petal_width": 2.1, "label": 0},
    {"sepal_length": 5.4, "sepal_width": 3.9, "petal_length": 1.7, "petal_width": 0.4, "label": 2},
]

PREDICT_SAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title=""):
    print("\n" + "\u2500" * 60)
    if title:
        print(f"  {title}")
        print("\u2500" * 60)


def ok(msg):   print(f"  \u2705  {msg}")
def warn(msg): print(f"  \u26a0\ufe0f   {msg}")
def info(msg): print(f"  \u2139\ufe0f   {msg}")
def err(msg):  print(f"  \u274c  {msg}")


def get(host, path):
    r = requests.get(f"{host}{path}", timeout=10)
    r.raise_for_status()
    return r.json()


def post(host, path, body):
    r = requests.post(f"{host}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def print_train_result(result):
    """Muestra el resultado de un entrenamiento de forma uniforme."""
    if result["model_updated"]:
        ok(f"Nuevo modelo ACTIVADO -> version: {result['model_version']}")
    else:
        ok(f"Modelo RECHAZADO -> version: {result['model_version']}")
    prev_acc = f"{result['accuracy_previous']:.4f}" if result.get('accuracy_previous') is not None else 'N/A'
    info(f"Accuracy nuevo: {result['accuracy_new']:.4f}  |  Anterior: {prev_acc}")
    info(f"Politica: {result.get('activation_policy', 'any_improvement')}")
    info(f"Razon: {result.get('gate_reason', '-')}")
    info(result["message"])


# ---------------------------------------------------------------------------
# Pasos del demo
# ---------------------------------------------------------------------------

def step_health(host):
    sep("PASO 0 - Health check")
    data = get(host, "/health")
    ok(f"Servicio activo. Modelo activo: {data['active_model_version']}")


def step_model_info(host, label="Estado actual del modelo"):
    sep(label)
    data = get(host, "/model/info")
    ok(f"Version activa : {data['active_version']}")
    ok(f"Entrenado el   : {data['trained_at']}")
    ok(f"Accuracy       : {data['accuracy']:.4f}")
    ok(f"N muestras     : {data['n_training_samples']}")
    ok(f"Algoritmo      : {data['algorithm']}")
    print()
    info(f"Historial de versiones ({len(data['history'])} entradas):")
    for entry in data["history"]:
        activated_icon = "\U0001f7e2" if entry.get("activated", True) else "\U0001f534"
        policy = entry.get("activation_policy", "-")
        print(f"    {activated_icon} {entry['version']}  |  acc={entry['accuracy']:.4f}  "
              f"|  {entry.get('status','-')}  |  policy={policy}")


def step_predict(host):
    sep("PASO 1 - Prediccion con el modelo base")
    result = post(host, "/predict", PREDICT_SAMPLE)
    ok(f"Prediccion: clase {result['prediction']} ({result['class_name']}), "
       f"version modelo: {result['model_version']}")


def step_train_good(host):
    sep("PASO 2 - Reentrenamiento con muestras CORRECTAS (any_improvement)")
    info(f"Enviando {len(GOOD_SAMPLES)} muestras bien etiquetadas...")
    result = post(host, "/train", {
        "samples": GOOD_SAMPLES,
        "retrain_from_scratch": False,
    })
    print_train_result(result)


def step_train_noisy(host):
    sep("PASO 3 - Reentrenamiento con muestras RUIDOSAS (any_improvement)")
    info(f"Enviando {len(NOISY_SAMPLES)} muestras con etiquetas erroneas...")
    result = post(host, "/train", {
        "samples": NOISY_SAMPLES,
        "retrain_from_scratch": True,
    })
    print_train_result(result)


def step_train_min_delta(host):
    sep("PASO 4 - Reentrenamiento con politica MIN_DELTA (delta=0.05)")
    info(f"Enviando {len(GOOD_SAMPLES)} muestras correctas con politica min_delta...")
    info("Se requiere que accuracy mejore al menos 0.05 sobre el modelo activo.")
    result = post(host, "/train", {
        "samples": GOOD_SAMPLES,
        "retrain_from_scratch": False,
        "activation_policy": {
            "type": "min_delta",
            "min_delta": 0.05,
        },
    })
    print_train_result(result)


def step_train_per_class_f1(host):
    sep("PASO 5 - Reentrenamiento con politica PER_CLASS_F1 (clase=1, versicolor)")
    info(f"Enviando {len(GOOD_SAMPLES)} muestras correctas con politica per_class_f1...")
    info("Se activa solo si el F1 de versicolor (clase 1) mejora respecto al modelo activo.")
    result = post(host, "/train", {
        "samples": GOOD_SAMPLES,
        "retrain_from_scratch": False,
        "activation_policy": {
            "type": "per_class_f1",
            "target_class": 1,
        },
    })
    print_train_result(result)


def step_predict_after(host):
    sep("PASO 6 - Prediccion con el modelo final")
    result = post(host, "/predict", PREDICT_SAMPLE)
    ok(f"Prediccion: clase {result['prediction']} ({result['class_name']}), "
       f"version modelo: {result['model_version']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo Continuous Training - Iris API")
    parser.add_argument("--host", default="http://localhost:8000",
                        help="URL base del servidor (default: http://localhost:8000)")
    parser.add_argument("--reset", action="store_true",
                        help="Resetea el historial antes de ejecutar el demo")
    args = parser.parse_args()

    print("\n" + "\u2550" * 60)
    print("  \U0001f338  DEMO: Iris Continuous Training API")
    print("\u2550" * 60)
    print(f"  Host: {args.host}")

    try:
        if args.reset:
            sep("RESET - Restaurando modelo base")
            requests.delete(f"{args.host}/model/history", timeout=10)
            ok("Historial eliminado. Modelo base restaurado.")
            time.sleep(0.5)

        step_health(host=args.host)
        step_model_info(host=args.host, label="PASO 0b - Info modelo base")
        step_predict(host=args.host)
        step_train_good(host=args.host)
        step_train_noisy(host=args.host)
        step_train_min_delta(host=args.host)
        step_train_per_class_f1(host=args.host)
        step_predict_after(host=args.host)
        step_model_info(host=args.host, label="RESUMEN FINAL - Historial de versiones")

        sep()
        ok("Demo completado. Abre http://localhost:8000/docs para explorar la API.")

    except requests.exceptions.ConnectionError:
        err(f"No se puede conectar al servidor en {args.host}")
        err("Asegurate de que el servidor esta arrancado:")
        print("    uvicorn main:app --reload")
        print("  o con Docker:")
        print("    docker run -d -p 8000:80 iris-ct")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        err(f"Error HTTP: {e}")
        err(f"Respuesta: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
