# =============================================================
# LAB 6 — backend-iris/Dockerfile
# Imagen del backend FastAPI con health check robusto
# =============================================================

FROM python:3.11-slim

WORKDIR /app

# Copiar requirements primero → mejor aprovechamiento de la caché de capas.
# Si sólo cambia el código (no las dependencias), Docker reutiliza esta capa
# y no reinstala paquetes innecesariamente.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

EXPOSE 8000

# HEALTHCHECK robusto:
# - interval=30s  → Docker comprueba el estado cada 30 segundos
# - timeout=10s   → la comprobación tiene 10 s para responder
# - start_period=60s → periodo de gracia inicial; el modelo puede tardar
#                      en descargarse desde HF Hub antes de estar listo
# - retries=3     → 3 fallos consecutivos → estado "unhealthy"
#
# Se usa urllib (stdlib) en lugar de curl porque python:3.11-slim no
# incluye curl. urlopen lanza HTTPError para respuestas 4xx/5xx (p. ej.
# 503 mientras el modelo carga) → exit 1 → unhealthy.
# Cuando /health devuelve 200 → exit 0 → healthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
