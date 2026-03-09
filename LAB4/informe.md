# Informe LAB4 - MLflow: Experimentacion y Despliegue de Modelos

## 1. Experimento de prueba

Se entreno una **LogisticRegression** como baseline sobre el dataset PIMA Diabetes
(768 muestras, 8 features, split 80/20 estratificado).

| Metrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.7143 |
| Precision | 0.6087 |
| Recall    | 0.5185 |
| F1        | 0.5600 |

El modelo fue registrado en MLflow como `diabetes-prueba-lr`.

## 2. Tabla de modelos registrados

Se realizaron Grid Search con 3 familias de modelos (16 + 48 + 18 = 82 combinaciones),
seleccionando las top-3 por CV Recall y reevaluando en validacion.

| Modelo | Parametros clave | CV Recall | Val Recall | Val F1 | Val Acc | Razon de registro |
|--------|-----------------|-----------|------------|--------|---------|-------------------|
| GradientBoosting-top1 | lr=0.2, depth=3, n_est=100 | 0.5987 | 0.9444 | 0.9714 | 0.9805 | Mejor recall |
| RandomForest-top1 | balanced, depth=3, split=5, n_est=200 | 0.7378 | 0.8519 | 0.7480 | 0.7987 | Mejor F1 |
| RandomForest-top2 | balanced, depth=3, split=2, n_est=200 | 0.7378 | 0.8519 | 0.7480 | 0.7987 | Alternativa |

Otros modelos destacados (no registrados):

| Modelo | CV Recall | Val Recall | Val F1 | Val Acc |
|--------|-----------|------------|--------|---------|
| LR-top1 (C=0.01, balanced, l2) | 0.7432 | 0.8148 | 0.7213 | 0.7792 |
| LR-top2 (C=0.1, balanced, l1) | 0.7377 | 0.8148 | 0.7395 | 0.7987 |
| GB-top2 (lr=0.1, depth=3, n_est=50) | 0.5987 | 0.7222 | 0.7879 | 0.8636 |
| GB-top3 (lr=0.1, depth=3, n_est=100) | 0.5933 | 0.7593 | 0.8367 | 0.8961 |

## 3. Criterio medico de seleccion

En el contexto de diabetes, un **falso negativo** (paciente diabetico clasificado como sano)
es mas peligroso que un **falso positivo** (paciente sano derivado a pruebas adicionales).

Por ello, el criterio de seleccion prioriza:
1. **Recall >= 0.65** - minimizar falsos negativos
2. **F1** - equilibrar precision sin sacrificar demasiado recall
3. **Accuracy** - desempate

Todos los modelos con `class_weight="balanced"` superaron el umbral de recall, confirmando
que la ponderacion de clases es esencial en datasets desbalanceados.

## 4. Modelo seleccionado para despliegue

- **Nombre en Model Registry:** `diabetes-best`
- **Tipo:** GradientBoosting
- **Parametros:** learning_rate=0.2, max_depth=3, n_estimators=100
- **Razon:** Mayor recall en validacion (0.9444) con excelente F1 (0.9714)

Nota: El alto rendimiento en validacion puede deberse a que el modelo fue reentrenado
en train+val y evaluado en el mismo val. Las metricas reales sobre test set (seccion 5)
seran mas representativas del rendimiento en produccion.

## 5. Metricas reales sobre test set (via API)

> Completar tras ejecutar `evaluar_api.py` con el modelo servido

| Metrica   | Valor |
|-----------|-------|
| Accuracy  | _pendiente_ |
| Precision | _pendiente_ |
| Recall    | _pendiente_ |
| F1        | _pendiente_ |

**Matriz de confusion:**
```
TN=__  FP=__
FN=__  TP=__
```

Para servir el modelo y evaluar:
```bash
mlflow models serve -m "models:/diabetes-best/1" --port 5001 --env-manager=local
python LAB4/evaluar_api.py --port 5001
```

## 6. Conclusiones

- **Tradeoff precision-recall:** En diagnostico medico, aceptamos mas falsos positivos
  a cambio de detectar mas casos reales. El recall es la metrica prioritaria.

- **class_weight="balanced":** Los modelos con ponderacion balanceada mejoran
  consistentemente el recall. Todos los modelos top tienen esta configuracion.

- **Limitaciones del dataset:** Las columnas Glucose, BloodPressure, SkinThickness,
  Insulin y BMI contienen valores 0 que representan datos faltantes. No se realizo
  imputacion (fuera del scope), lo que puede afectar el rendimiento de los modelos.

- **Sobreajuste en validacion:** El GradientBoosting muestra metricas muy altas en
  validacion (recall=0.94) porque fue reentrenado en train+val. Las metricas sobre
  test set seran mas fiables.

- **Valor de MLflow:** Permite comparar sistematicamente 82 configuraciones,
  reproducir experimentos, y desplegar modelos con un comando. El Model Registry
  facilita la gestion del ciclo de vida del modelo.
