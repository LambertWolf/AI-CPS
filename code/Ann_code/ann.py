import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

print(f"TensorFlow Version: {tf.__version__}")

# ==================
# DATEN LADEN
# ==================
print("\n" + "=" * 60)
print("DATEN LADEN")
print("=" * 60)

df = pd.read_csv("./finaldata/joint_data_collection.csv")
print(f"Dataset: {df.shape}")

# ==================
# FEATURES VORBEREITEN (wie bei OLS)
# ==================
print("\n" + "=" * 60)
print("FEATURES VORBEREITEN")
print("=" * 60)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

# Cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Rush hour
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

# Weekend
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

print("✓ Zeitliche Features erstellt")

# ==================
# STATION ENCODING (wie bei OLS)
# ==================
print("\n" + "=" * 60)
print("STATION ENCODING")
print("=" * 60)

# Encoder laden (vom OLS Script)
with open("./ols/station_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

station_encoded = encoder.transform(df[["stop_id"]])
station_cols = [f"station_{int(cat)}" for cat in encoder.categories_[0][1:]]
station_df = pd.DataFrame(station_encoded, columns=station_cols, index=df.index)

# Features zusammenführen
X_base = df[["hour", "weekday"]]
X = pd.concat([X_base, station_df], axis=1)
y = df["delay_minutes"]

print(f"✓ Features: {X.shape[1]} (6 Basis + {len(station_cols)} Stationen)")

# ==================
# TRAIN-TEST SPLIT (wie bei OLS)
# ==================
print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")

# ==================
# NORMALISIERUNG (wie bei OLS)
# ==================
print("\n" + "=" * 60)
print("NORMALISIERUNG")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalisiert (Mean=0, Std=1)")

# ==================
# ANN MODELL BAUEN
# ==================
print("\n" + "=" * 60)
print("ANN MODELL ERSTELLEN")
print("=" * 60)

# Random seed für Reproduzierbarkeit
tf.random.set_seed(42)
np.random.seed(42)

# Modell-Architektur
model = keras.Sequential(
    [
        # Input Layer (automatisch basierend auf ersten Daten)
        # Hidden Layer 1: 64 Neuronen
        layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        layers.Dropout(0.2),  # Dropout gegen Overfitting
        # Hidden Layer 2: 32 Neuronen
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        # Hidden Layer 3: 16 Neuronen
        layers.Dense(16, activation="relu"),
        # Output Layer: 1 Neuron (delay prediction)
        layers.Dense(1),
    ]
)

# Modell kompilieren
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mean_squared_error",
    metrics=["mae"],
)

print("\n📊 Modell-Architektur:")
model.summary()

# ==================
# CALLBACKS DEFINIEREN
# ==================
print("\n" + "=" * 60)
print("CALLBACKS VORBEREITEN")
print("=" * 60)

# Early Stopping: Stoppt Training wenn kein Fortschritt mehr
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

# Learning Rate Reduction: Reduziert LR wenn kein Fortschritt
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
)

print("✓ Early Stopping aktiviert (patience=15)")
print("✓ Learning Rate Reduction aktiviert")

# ==================
# MODELL TRAINIEREN
# ==================
print("\n" + "=" * 60)
print("TRAINING STARTEN")
print("=" * 60)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,  # 20% von Training für Validation
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

print("\n✓ Training abgeschlossen!")

# ==================
# VORHERSAGEN
# ==================
print("\n" + "=" * 60)
print("VORHERSAGEN ERSTELLEN")
print("=" * 60)

y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()

print("✓ Vorhersagen erstellt")

# ==================
# PERFORMANCE METRIKEN
# ==================
print("\n" + "=" * 60)
print("PERFORMANCE METRIKEN")
print("=" * 60)

# Training Performance
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test Performance
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n📊 Training Set:")
print(f"  R² Score:  {train_r2:.4f}  (1.0 = perfekt)")
print(f"  MSE:       {train_mse:.2f}")
print(f"  MAE:       {train_mae:.2f} Min (durchschnittlicher Fehler)")

print(f"\n📊 Test Set:")
print(f"  R² Score:  {test_r2:.4f}  (1.0 = perfekt)")
print(f"  MSE:       {test_mse:.2f}")
print(f"  MAE:       {test_mae:.2f} Min (durchschnittlicher Fehler)")

# ==================
# VERGLEICH MIT OLS
# ==================
print("\n" + "=" * 60)
print("VERGLEICH: ANN vs OLS")
print("=" * 60)

# OLS Ergebnisse laden
with open("./ols/currentOlsSolution.pkl", "rb") as f:
    ols_model = pickle.load(f)

# OLS Vorhersagen auf GLEICHEN Daten
import statsmodels.api as sm

X_train_const = sm.add_constant(X_train_scaled)
X_test_const = sm.add_constant(X_test_scaled)

ols_train_pred = ols_model.predict(X_train_const)
ols_test_pred = ols_model.predict(X_test_const)

ols_train_r2 = r2_score(y_train, ols_train_pred)
ols_test_r2 = r2_score(y_test, ols_test_pred)
ols_test_mae = mean_absolute_error(y_test, ols_test_pred)

print("\n📊 OLS (Baseline):")
print(f"  Train R²:  {ols_train_r2:.4f}")
print(f"  Test R²:   {ols_test_r2:.4f}")
print(f"  Test MAE:  {ols_test_mae:.2f} Min")

print("\n📊 ANN (Neural Network):")
print(f"  Train R²:  {train_r2:.4f}")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test MAE:  {test_mae:.2f} Min")

print("\n🎯 Verbesserung:")
r2_improvement = ((test_r2 - ols_test_r2) / ols_test_r2) * 100
mae_improvement = ((ols_test_mae - test_mae) / ols_test_mae) * 100
print(f"  R² Verbesserung:  {r2_improvement:+.1f}%")
print(f"  MAE Verbesserung: {mae_improvement:+.1f}%")

# ==================
# MODELL SPEICHERN
# ==================
print("\n" + "=" * 60)
print("MODELL SPEICHERN")
print("=" * 60)

model.save("./ann_first/ann_model.keras")
print("✓ ANN Modell gespeichert: ann_model.keras")


# ==================
# VISUALISIERUNGEN
# ==================
print("\n" + "=" * 60)
print("VISUALISIERUNGEN ERSTELLEN")
print("=" * 60)

# =====================================
# Plot 1: Training History
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ANN Training History", fontsize=16, fontweight="bold")

# Loss
axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("MSE Loss", fontsize=12)
axes[0].set_title("Model Loss", fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history["mae"], label="Training MAE", linewidth=2)
axes[1].plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("MAE (Minutes)", fontsize=12)
axes[1].set_title("Mean Absolute Error", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./ann_first/ann_training_history.png", dpi=300)
print("✓ Training History gespeichert: ann_training_history.png")
plt.show()

# =====================================
# Plot 2: Predicted vs Actual (ANN)
# =====================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("ANN Model - Predicted vs Actual Delay", fontsize=16, fontweight="bold")

# Training Set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20, color="blue")
axes[0].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "r--",
    lw=2,
    label="Perfect Prediction",
)
axes[0].set_xlabel("Actual Delay (Min)", fontsize=12)
axes[0].set_ylabel("Predicted Delay (Min)", fontsize=12)
axes[0].set_title(f"Training Set (R²={train_r2:.3f})", fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test Set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color="green")
axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction",
)
axes[1].set_xlabel("Actual Delay (Min)", fontsize=12)
axes[1].set_ylabel("Predicted Delay (Min)", fontsize=12)
axes[1].set_title(f"Test Set (R²={test_r2:.3f})", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./ann_first/ann_scatter_plots.png", dpi=300)
print("✓ Scatter Plots gespeichert: ann_scatter_plots.png")
plt.show()

# =====================================
# Plot 3: OLS vs ANN Comparison
# =====================================
fig, ax = plt.subplots(figsize=(10, 6))

models = ["OLS", "ANN"]
train_r2_scores = [ols_train_r2, train_r2]
test_r2_scores = [ols_test_r2, test_r2]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(
    x - width / 2, train_r2_scores, width, label="Training R²", color="steelblue"
)
bars2 = ax.bar(x + width / 2, test_r2_scores, width, label="Test R²", color="coral")

ax.set_xlabel("Model", fontsize=12, fontweight="bold")
ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
ax.set_title("Model Comparison: OLS vs ANN", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

# Werte auf Balken schreiben
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

plt.tight_layout()
plt.savefig("./ann_first/model_comparison.png", dpi=300)
print("✓ Model Comparison gespeichert: model_comparison.png")
plt.show()

print("\n" + "=" * 60)
print("✅ FERTIG! Alle Plots gespeichert.")
print("=" * 60)
