# LAB 3 -- `app-iris-ct`: Continuous Training con FastAPI

## Estructura del proyecto

```
app-iris-ct/
+-- main.py              <- servidor FastAPI extendido
+-- demo_ct.py           <- script de demostracion del flujo CT
+-- Dockerfile
+-- requirements.txt
+-- models/              <- creada automaticamente en runtime
|   +-- model_active.joblib       <- modelo activo serializado
|   +-- accumulated_data.joblib   <- dataset acumulado entre entrenamientos
|   +-- training_history.json     <- registro de versiones
+-- README.md
```

---

## Endpoints

| Metodo | Ruta               | Descripcion                                              |
| ------- | ------------------ | --------------------------------------------------------- |
| GET     | `/health`        | Estado del servicio y version del modelo activo          |
| POST    | `/predict`       | Prediccion con el modelo activo                          |
| POST    | `/train`         | Reentrenamiento con nuevas muestras etiquetadas           |
| GET     | `/model/info`    | Metadata del modelo activo e historial de versiones       |
| DELETE  | `/model/history` | Resetea el historial (para pruebas)                       |

### Esquema de `/train` (request)

```json
{
  "samples": [
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2,
      "label": 0
    }
  ],
  "retrain_from_scratch": false,
  "activation_policy": {
    "type": "any_improvement",
    "min_delta": 0.01,
    "target_class": 0
  }
}
```

- `label` acepta `0` (setosa), `1` (versicolor) o `2` (virginica).
- Se requieren **minimo 5 muestras** por request.
- `retrain_from_scratch: false` -> las nuevas muestras se **acumulan** al dataset anterior.
- `retrain_from_scratch: true` -> se entrena **solo** con las muestras enviadas.
- `activation_policy` es opcional (por defecto: `any_improvement`).

### Politicas de activacion

| Politica | Campo(s) relevante(s) | Comportamiento |
| --- | --- | --- |
| `any_improvement` | - | Activa si `accuracy_new >= accuracy_previous` |
| `min_delta` | `min_delta` (float, 0-1) | Activa si `accuracy_new >= accuracy_previous + min_delta` |
| `per_class_f1` | `target_class` (0, 1 o 2) | Activa si el F1 de la clase objetivo mejora respecto al modelo anterior |

**Ejemplos de uso:**

```json
// Politica por defecto (any_improvement) - no es necesario incluir activation_policy
{
  "samples": [...],
  "retrain_from_scratch": false
}

// Requiere al menos 5% de mejora en accuracy
{
  "samples": [...],
  "activation_policy": {
    "type": "min_delta",
    "min_delta": 0.05
  }
}

// Activa solo si mejora el F1 de versicolor (clase 1)
{
  "samples": [...],
  "activation_policy": {
    "type": "per_class_f1",
    "target_class": 1
  }
}
```

### Esquema de `/train` (response)

```json
{
  "status": "activado",
  "model_version": "v2.0-a3f9b1",
  "accuracy_new": 0.9667,
  "accuracy_previous": 0.9333,
  "model_updated": true,
  "message": "Nuevo modelo activado. Accuracy 0.9667 >= anterior (0.9333)",
  "activation_policy": "any_improvement",
  "gate_reason": "Accuracy 0.9667 >= anterior (0.9333)"
}
```

### Esquema de `/model/info` (response)

```json
{
  "active_version": "v2.0-a3f9b1",
  "trained_at": "2024-11-15T10:23:44Z",
  "accuracy": 0.9667,
  "n_training_samples": 120,
  "algorithm": "LogisticRegression",
  "history": [
    {
      "version": "v1.0-base",
      "trained_at": "2024-11-15T09:00:00Z",
      "accuracy": 0.9333,
      "n_training_samples": 120,
      "source": "bootstrap (iris dataset completo)",
      "activated": true
    },
    {
      "version": "v2.0-a3f9b1",
      "trained_at": "2024-11-15T10:23:44Z",
      "accuracy": 0.9667,
      "n_training_samples": 140,
      "source": "incremental (+20 muestras nuevas, 120 anteriores)",
      "activated": true,
      "activation_policy": "any_improvement",
      "gate_reason": "Accuracy 0.9667 >= anterior (0.9333)"
    }
  ]
}
```

---

## Logica del gate de calidad

La politica por defecto (`any_improvement`) mantiene el comportamiento clasico:

```
accuracy_nuevo >= accuracy_anterior  ->  ACTIVAR y guardar modelo
accuracy_nuevo <  accuracy_anterior  ->  RECHAZAR, mantener modelo anterior
```

Con `min_delta`, se exige una mejora minima configurable para evitar activaciones laterales
(modelos que no mejoran realmente pero pasan el gate por diferencias marginales).

Con `per_class_f1`, se evalua el rendimiento en una clase especifica, util cuando
una clase es mas critica que otras (ej. detectar una especie rara).

El registro del intento de entrenamiento **siempre** queda en el historial, incluso si el
modelo es rechazado. Esto permite auditar que datos degradaron el modelo y que politica
se uso en cada decision.

---

## Instrucciones de desarrollo

### Prerrequisitos

```bash
pip install -r requirements.txt
```

### Arrancar el servidor en local

```bash
uvicorn main:app --reload
```

Accede a la documentacion interactiva en: [http://localhost:8000/docs](http://localhost:8000/docs)

### Arrancar con Docker

```bash
# Construir imagen
docker build -t iris-ct .

# Lanzar contenedor con volumen para persistir los modelos
docker run -d -p 8000:80 -v iris-ct-models:/app/models --name iris-ct iris-ct
```

Con el volumen `-v iris-ct-models:/app/models` los modelos entrenados **sobreviven** al reinicio del contenedor.

### Ejecutar la demo completa

```bash
# Servidor debe estar arrancado primero
python demo_ct.py --reset
```

El flag `--reset` restaura el modelo base antes de ejecutar el flujo. La demo ejecuta:

1. Health check y estado del modelo base
2. Prediccion con modelo base
3. Entrenamiento con muestras correctas (any_improvement) -> activado
4. Entrenamiento con muestras ruidosas (any_improvement) -> rechazado
5. Entrenamiento con politica min_delta (delta=0.05) -> probablemente rechazado
6. Entrenamiento con politica per_class_f1 (versicolor) -> depende del F1
7. Prediccion final y resumen del historial

---

## Ejercicios de la actividad

**Ejercicio 1 -- Reproducir el workflow**
Arranca el servidor, ejecuta `demo_ct.py --reset` y comprueba que los pasos funcionan
correctamente. Captura la salida completa del terminal y el JSON de `/model/info` al final.

**Ejercicio 2 -- Explorar el historial**

Tras ejecutar la demo, abre `models/training_history.json` y responde:

- Cuantas versiones se han registrado?
- Que versiones fueron activadas y cuales rechazadas?
- Por que el modelo entrenado con muestras ruidosas fue rechazado?
- Que politica de activacion se uso en cada entrenamiento?

**Ejercicio 3 -- Entrenamiento incremental vs. desde cero**

Realiza dos llamadas a `/train` con el mismo conjunto de 10 muestras:

- Primera vez con `retrain_from_scratch: false`
- Segunda vez con `retrain_from_scratch: true`

Observas diferencias en el accuracy? Por que?

---

## Referencias

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Google MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Joblib documentation](https://joblib.readthedocs.io/)
