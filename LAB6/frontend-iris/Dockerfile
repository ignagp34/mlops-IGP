# =============================================================
# LAB 6 — frontend-iris/Dockerfile
# Imagen del frontend Gradio
# =============================================================

FROM python:3.11-slim

WORKDIR /app

# Copiar requirements primero → mejor aprovechamiento de la caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

EXPOSE 7860

# URL del backend por defecto; se sobreescribe en Docker Compose con
# BACKEND_URL=http://backend:8000 para usar la red interna del stack.
ENV BACKEND_URL=http://localhost:8000

CMD ["python", "gradio_app.py"]
