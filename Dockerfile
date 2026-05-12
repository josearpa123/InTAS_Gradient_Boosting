# 1. IMAGEN DE BASE: Laboratorio de Simulación
FROM ubuntu:22.04

# 2. METADATA DE LA TESIS
LABEL maintainer="InTAS Research Team"
LABEL version="1.0"
LABEL description="Entorno reproducible para Tesis InTAS: SUMO + OMNeT++ + ML"

# 3. PREPARACIÓN DEL SISTEMA (Dependencias de OMNeT++ y SUMO)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ bison flex perl python3 python3-pip \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
    libqt5opengl5-dev libxml2-dev zlib1g-dev default-jre \
    sumo sumo-tools sumo-gui \
    wget git curl \
    && rm -rf /var/lib/apt/lists/*

# 4. CONFIGURACIÓN DE ENTORNOS
ENV SUMO_HOME=/usr/share/sumo
ENV OMNET_DIR=/opt/omnetpp
WORKDIR /app

# 5. INSTALACIÓN DE DEPENDENCIAS PYTHON (Versiones Exactas)
COPY requirements.txt .
# Crear venv explícito para evitar conflictos PEP 668 en Ubuntu 22.04+
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# 6. COPIA DEL REPOSITORIO LIMPIO
COPY . .

# 7. SCRIPT DE ENTRADA (Reproducción del Pipeline de la Tesis)
# El orquestador usa sys.executable → todos los subprocesos usan el mismo venv.
CMD ["/app/.venv/bin/python", "scripts/run_full_reproduction.py"]
