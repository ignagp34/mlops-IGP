# LAB4 - MLflow: Experimentación y Despliegue de Modelos

## Experimento de Prueba

- Entrenamiento de un modelo
- Registrar dicho modelo
- Exponerlo a través de una API, y comprobar que hace las predicciones correctamente

## Experimentos Completos

- Dividir el dataset en training (70%), validación (20%) y test (10%)
- Entrenar el modelo con los datasets de training y validación
- Ajustar los parámetros, utilizando diferentes combinaciones (por ejemplo con técnicas de Grid Search)
- Hacer una selección de los mejores y registrarlos en MLflow. Para elegir los mejores considera las diferentes métricas: accuracy, precision, recall y F1 score. Utiliza criterios de carácter médico para decidir qué métrica(s) son más relevantes
- Exponer el modelo finalmente seleccionado a través de la API de MLflow
- Evaluar las métricas reales del modelo sobre el dataset de test (10%). Utiliza para ello un script que obtenga las predicciones de todo el dataset a través de la API
- Elabora tus conclusiones

## Entregables

- Tabla de parámetros y métricas de los modelos que hayas decidido incluir en el registro, especificando la razón de por qué es útil registrar estos y no los de otros runs que hayas realizado
- Script de obtención de las métricas de la API (aplicado al dataset de test)
- Conclusiones
