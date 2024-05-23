FROM python:3.10.11-slim-bullseye

WORKDIR /app

RUN mkdir -p /app/data

COPY requirements.txt /app

RUN apt-get -y update && \
    apt-get install -y libmagic-mgc libmagic1 && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install -U pip && pip install --no-cache-dir -r requirements.txt && \
    adduser --disabled-password --gecos '' appuser

COPY . .


EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]