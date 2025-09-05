# Stage 1: Base on pinned Airflow version
FROM apache/airflow:2.8.1-python3.10

# Stage 2: Switch to non-root (best practice)
USER airflow

# Stage 3: Copy only what we need
# (avoid clutter, only DAGs/plugins/config)
COPY airflow/dags /opt/airflow/dags

# Stage 4: Healthcheck for container
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8080/health || exit 1

# Entrypoint is provided by base image (airflow)
