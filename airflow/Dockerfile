FROM apache/airflow:2.9.0-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

# Install FinHedge backend dependencies so DAGs can import them
COPY --chown=airflow:root requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

ENV PYTHONPATH=/opt/airflow
