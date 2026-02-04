# Dockerfile für learningBase_s7_delays
# Basierend auf busybox 
FROM busybox:latest

# Erstelle die erforderlichen Verzeichnisstrukturen
RUN mkdir -p /tmp/learningBase/train && \
    mkdir -p /tmp/learningBase/validation

# Kopiere die Trainingsdaten
COPY ./finaldata/training_data.csv /tmp/learningBase/train/training_data.csv

# Kopiere die Validierungsdaten (Test-Set)
COPY ./finaldata/test_data.csv /tmp/learningBase/validation/test_data.csv

# Setze Arbeitsverzeichnis 
WORKDIR /tmp/learningBase

# Default Command: zeige verfügbare Dateien
CMD ["ls", "-la", "/tmp/learningBase"]

