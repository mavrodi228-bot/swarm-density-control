FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 PORT=7860 AGMEN_DATA_DIR=/data AGMEN_PROFILE=public \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    AGMEN_MAX_RSS_MB=1536 SWARM_ENABLE_REAL=0 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements-core.txt requirements-runtime.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY *.py ./
COPY edge/ ./edge/
COPY model_data/ ./model_data/
COPY --from=frontend /frontend/dist ./frontend/dist
RUN useradd --uid 10001 --create-home agmen && mkdir -p /data && chown -R agmen:agmen /data /app
USER agmen
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=3)"
CMD ["python", "launch_agmen.py", "--serve"]
