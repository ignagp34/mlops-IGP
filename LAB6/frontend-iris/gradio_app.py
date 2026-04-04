import gradio as gr
import requests
import os

# URL del backend — en Docker Compose se inyecta como variable de entorno
# (BACKEND_URL=http://backend:8000); fuera de Docker usa localhost por defecto
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000") + "/predict"

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width":  sepal_width,
        "petal_length": petal_length,
        "petal_width":  petal_width,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        species    = result.get("species", "unknown")
        confidence = result.get("confidence")
        if confidence is not None:
            return f"{species}  (confidence: {confidence:.1%})"
        return species
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_iris,
    inputs=["number", "number", "number", "number"],
    outputs="text",
    live=True,
    title="Iris Species Identifier",
    description="Enter the four measurements to predict the Iris species."
)

if __name__ == "__main__":
    # server_name="0.0.0.0" es necesario para que Gradio sea accesible
    # desde fuera del contenedor Docker (sin esto solo escucha en loopback)
    interface.launch(server_name="0.0.0.0", server_port=7860)
