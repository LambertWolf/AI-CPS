FROM python:3.9-slim

WORKDIR /app

# Keras ZUERST installieren, DANN TensorFlow
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    keras==3.10.0 \
    tensorflow==2.16.2 \
    pandas \
    "numpy<2.0.0" \
    scipy \
    scikit-learn \
    statsmodels \
    matplotlib \
    seaborn

# Keras 3 explizit aktivieren UND sicherstellen dass die standalone Version genutzt wird
ENV TF_USE_LEGACY_KERAS=0
ENV KERAS_BACKEND=tensorflow

# Version-Check beim Start hinzufügen
RUN python3 -c "import keras; print(f'Keras: {keras.__version__}')"

COPY ./apply_ai_model.py /app/apply_ai_model.py
COPY ./apply_ols_model.py /app/apply_ols_model.py

CMD ["python", "/app/apply_ai_model.py"]