# Dockerfile für knowledgeBase_sbahn_delay_prediction
# Basierend auf busybox (gemäß Subgoal 6 PDF-Anforderungen)
FROM busybox:latest

# Kopiere Dateien in ein Staging-Verzeichnis (nicht /tmp, da das Volume /tmp überschreibt)
# ANN Model
COPY ./ann_improved_model_new.keras /data/ann_improved_model_new.keras
COPY ./ann_improved_scaler.pkl /data/ann_improved_scaler.pkl
COPY ./ann_improved_feature_names.pkl /data/ann_improved_feature_names.pkl
# OLS Model
COPY ./currentOlsSolution.pkl /data/currentOlsSolution.pkl
# README
COPY README.md /data/README.md

# Beim Start: Erstelle Verzeichnis und kopiere Dateien ins gemountete Volume
CMD mkdir -p /tmp/knowledgeBase && \
    cp /data/* /tmp/knowledgeBase/ && \
    ls -la /tmp/knowledgeBase
