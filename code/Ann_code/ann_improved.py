import os

os.environ["TF_USE_LEGACY_KERAS"] = "0"
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


warnings.filterwarnings("ignore")

print(f"TensorFlow Version: {tf.__version__}")

# ==================
# DATEN LADEN
# ==================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING IMPROVEMENTS")
print("=" * 60)

# WICHTIG: Verwende die BEREINIGTE Datei (aus apply_ols_model.py)
# Diese hat bereits Outlier-Filtering (2% bottom, 2% top entfernt)
df = pd.read_csv(os.path.join(PROJECT_ROOT, "finaldata", "joint_data_collection.csv"))

# Timestamp konvertieren
df["timestamp"] = pd.to_datetime(df["timestamp"])

# WICHTIG: Nach Station und Zeit sortieren für Lag-Features
df = df.sort_values(["stop_id", "timestamp"]).reset_index(drop=True)
print(f"✓ Sorted by station and timestamp")

# ==================
# FEATURE ENGINEERING (NEUE FEATURES - und Improved)
# ==================
print("\n" + "=" * 60)
print("CREATING IMPROVED FEATURES")
print("=" * 60)

# Basis-Features
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

print("\n1️⃣ Lag Features (Previous Delays)")
# Lag Features: Vorherige 3 Verspätungen pro Station
df["delay_lag_1"] = df.groupby("stop_id")["delay_minutes"].shift(1)
df["delay_lag_2"] = df.groupby("stop_id")["delay_minutes"].shift(2)
df["delay_lag_3"] = df.groupby("stop_id")["delay_minutes"].shift(3)

# NaN mit 0 füllen (bedeutet: keine vorherige Information)
lag_cols = ["delay_lag_1", "delay_lag_2", "delay_lag_3"]
df[lag_cols] = df[lag_cols].fillna(0)

print(f"   ✓ Created: delay_lag_1, delay_lag_2, delay_lag_3")
print(f"   ✓ NaN filled with 0 (no prior information)")

print("\n2️⃣ Cyclical Encoding (sin/cos)")
# Hour cyclical encoding
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# Weekday cyclical encoding
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

print(f"   ✓ Hour: hour_sin, hour_cos")
print(f"   ✓ Weekday: weekday_sin, weekday_cos (NEW!)")

print("\n3️⃣ Station Statistics")
# Pro Station: Durchschnitt und Standardabweichung der Verspätungen
station_stats = (
    df.groupby("stop_id")["delay_minutes"].agg(["mean", "std"]).reset_index()
)
station_stats.columns = ["stop_id", "station_mean_delay", "station_std_delay"]

# Merge zurück zum DataFrame
df = df.merge(station_stats, on="stop_id", how="left")

print(f"   ✓ station_mean_delay (average delay per station)")
print(f"   ✓ station_std_delay (delay variability per station)")
print(f"   ✓ Calculated for {df['stop_id'].nunique()} unique stations")

print("\n4️⃣ Rush Hour Indicators")
# Morgen Rush Hour: 7-9 Uhr
df["is_morning_rush"] = df["hour"].isin([7, 8, 9]).astype(int)

# Abend Rush Hour: 16-19 Uhr
df["is_evening_rush"] = df["hour"].isin([16, 17, 18, 19]).astype(int)

# Gesamt Rush Hour
df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)

# Weekend
df["is_weekend"] = (df["weekday"] >= 5).astype(int)

print(f"   ✓ is_morning_rush (7-9 AM)")
print(f"   ✓ is_evening_rush (4-7 PM)")
print(f"   ✓ is_rush_hour (combined)")
print(f"   ✓ is_weekend")

# ==================
# STATION ENCODING
# ==================
print("\n" + "=" * 60)
print("STATION ENCODING")
print("=" * 60)

# Encoder laden (gespeichert von apply_ols_model.py)
encoder_path = os.path.join(PROJECT_ROOT, "ols", "station_encoder.pkl")
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

station_encoded = encoder.transform(df[["stop_id"]])
station_cols = [f"station_{int(cat)}" for cat in encoder.categories_[0][1:]]
station_df = pd.DataFrame(station_encoded, columns=station_cols, index=df.index)

print(f"✓ One-hot encoded {len(station_cols)} stations")

# ==================
# FEATURES ZUSAMMENFÜHREN
# ==================
print("\n" + "=" * 60)
print("FEATURE SUMMARY")
print("=" * 60)

# Alle Features kombinieren
feature_cols = [
    # Basis temporal
    "hour",
    "weekday",
    # Cyclical encoding
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    # Rush hour indicators
    "is_morning_rush",
    "is_evening_rush",
    "is_rush_hour",
    "is_weekend",
    # Lag features
    "delay_lag_1",
    "delay_lag_2",
    "delay_lag_3",
    # Station statistics
    "station_mean_delay",
    "station_std_delay",
]

X_base = df[feature_cols]
X = pd.concat([X_base, station_df], axis=1)
y = df["delay_minutes"]

print(f"\n📊 Feature Breakdown:")
print(f"   Temporal features:      10 (hour, weekday, sin/cos, rush hour, weekend)")
print(f"   Lag features (NEW):      3 (delay_lag_1, delay_lag_2, delay_lag_3)")
print(f"   Station stats (NEW):     2 (mean_delay, std_delay)")
print(f"   One-hot stations:       {len(station_cols)}")
print(f"   ─────────────────────────")
print(f"   TOTAL FEATURES:         {X.shape[1]}")
print(f"\n   OLD model had:          {6 + len(station_cols)} features")
print(f"   NEW features added:     +{3 + 2 + 2} = +7 features")

# ==================
# TRAIN-TEST SPLIT
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
# NORMALISIERUNG
# ==================
print("\n" + "=" * 60)
print("NORMALISIERUNG")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalized (Mean=0, Std=1)")

# ==================
# DATEN SPEICHERN (für Container)
# ==================
print("\n" + "=" * 60)
print("SAVING SPLIT DATA FOR DOCKER CONTAINERS")
print("=" * 60)

pd.DataFrame(X_train_scaled, columns=X_train.columns).assign(
    delay_minutes=y_train.values
).to_csv(os.path.join(PROJECT_ROOT, "ann", "training_data.csv"), index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).assign(
    delay_minutes=y_test.values
).to_csv(os.path.join(PROJECT_ROOT, "ann", "test_data.csv"), index=False)

# Activation Data (1 Eintrag aus Test Set)
pd.DataFrame(X_test_scaled[:1], columns=X_test.columns).assign(
    delay_minutes=y_test.values[:1]
).to_csv(os.path.join(PROJECT_ROOT, "ann", "activation_data.csv"), index=False)

print(f"✓ training_data.csv saved: {X_train_scaled.shape[0]} rows")
print(f"✓ test_data.csv saved: {X_test_scaled.shape[0]} rows")
print("✓ activation_data.csv saved: 1 row (for inference)")

# ==================
# ANN MODELL
# ==================
print("\n" + "=" * 60)
print("ANN MODEL - SAME ARCHITECTURE AS BASELINE")
print("=" * 60)

# Random seed für Reproduzierbarkeit
tf.random.set_seed(42)
np.random.seed(42)
# Keras Model
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

print("\n📊 Model Architecture (UNCHANGED from baseline):")
model.summary()

# ==================
# CALLBACKS
# ==================
print("\n" + "=" * 60)
print("CALLBACKS (UNCHANGED)")
print("=" * 60)

# Early Stopping
early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
)

# Learning Rate Reduction
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
)

print("✓ Early Stopping (patience=15)")
print("✓ Learning Rate Reduction (factor=0.5, patience=5)")

# ==================
# TRAINING
# ==================
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

# Training
history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.2,  # 20% von Training für Validation
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

print("\n✓ Training completed!")

# ==================
# VORHERSAGEN
# ==================
print("\n" + "=" * 60)
print("PREDICTIONS")
print("=" * 60)

y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()

print("✓ Predictions generated")

# ==================
# PERFORMANCE METRIKEN
# ==================
print("\n" + "=" * 60)
print("PERFORMANCE METRICS - NEW MODEL")
print("=" * 60)

# Training Performance
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test Performance
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n📊 NEW MODEL (Improved Features):")
print(f"   Training Set:")
print(f"     R² Score:  {train_r2:.4f}")
print(f"     MAE:       {train_mae:.4f} minutes")
print(f"\n   Test Set:")
print(f"     R² Score:  {test_r2:.4f}")
print(f"     MAE:       {test_mae:.4f} minutes")

# ==================
# VERGLEICH MIT BASELINE MODEL
# ==================
print("\n" + "=" * 60)
print("COMPARISON WITH BASELINE MODEL")
print("=" * 60)

try:
    # Versuche, das alte Modell zu laden
    baseline_model = keras.models.load_model(os.path.join(PROJECT_ROOT, "ann_first", "ann_model.keras"))

    # Alte Features vorbereiten (nur die Features, die das alte Modell hatte)
    # ann.py nutzt nur hour + weekday + stations (Zeile 63: df[["hour", "weekday"]])
    old_feature_cols = [
        "hour",
        "weekday",
    ]
    X_test_old_base = X_test[old_feature_cols]
    X_test_old = pd.concat([X_test_old_base, X_test[station_cols]], axis=1)

    # Scaler für alte Features
    old_scaler = StandardScaler()
    X_train_old_base = X_train[old_feature_cols]
    X_train_old = pd.concat([X_train_old_base, X_train[station_cols]], axis=1)
    X_train_old_scaled = old_scaler.fit_transform(X_train_old)
    X_test_old_scaled = old_scaler.transform(X_test_old)

    # Vorhersagen mit altem Modell
    baseline_test_pred = baseline_model.predict(X_test_old_scaled, verbose=0).flatten()

    # Baseline Performance
    baseline_test_r2 = r2_score(y_test, baseline_test_pred)
    baseline_test_mae = mean_absolute_error(y_test, baseline_test_pred)

    print(f"\n📊 BASELINE MODEL (Old Features):")
    print(f"   Test R²:  {baseline_test_r2:.4f}")
    print(f"   Test MAE: {baseline_test_mae:.4f} minutes")

    print(f"\n📊 NEW MODEL (Improved Features):")
    print(f"   Test R²:  {test_r2:.4f}")
    print(f"   Test MAE: {test_mae:.4f} minutes")

    # Improvement berechnen
    mae_improvement = ((baseline_test_mae - test_mae) / baseline_test_mae) * 100
    r2_improvement = test_r2 - baseline_test_r2

    print(f"\n🎯 IMPROVEMENT:")
    print(
        f"   MAE Reduction:    {mae_improvement:+.2f}% ({baseline_test_mae - test_mae:+.4f} minutes)"
    )
    print(f"   R² Improvement:   {r2_improvement:+.4f}")

    baseline_available = True

except Exception as e:
    print(f"\n⚠ Could not load baseline model: {e}")
    print(f"   Baseline comparison skipped.")
    baseline_available = False
    baseline_test_mae = None

# ==================
# FEATURE IMPORTANCE ANALYSIS
# ==================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

print("\nCalculating permutation importance (this may take a moment)...")


# Custom scoring function für Keras-Modelle (R²-Score)
def keras_r2_score(model, X, y):
    y_pred = model.predict(X, verbose=0).flatten()
    return r2_score(y, y_pred)


# Permutation importance berechnen
perm_importance = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring=keras_r2_score,
)

# Feature names (alle)
feature_names = list(X.columns)

# DataFrame erstellen
feature_importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": perm_importance.importances_mean,
        "std": perm_importance.importances_std,
    }
).sort_values("importance", ascending=False)

# Top 15 Features anzeigen
print("\n🔍 TOP 15 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance_df.head(15).iterrows():
    print(
        f"   {row['feature']:25s}  Importance: {row['importance']:.4f} ± {row['std']:.4f}"
    )

# Feature-Kategorien analysieren
print("\n📈 FEATURE CATEGORY CONTRIBUTIONS:")

lag_features = [col for col in feature_names if "lag" in col]
station_stat_features = ["station_mean_delay", "station_std_delay"]
cyclical_features = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
rush_features = ["is_morning_rush", "is_evening_rush", "is_rush_hour"]
station_onehot = [col for col in feature_names if col.startswith("station_")]

lag_importance = feature_importance_df[
    feature_importance_df["feature"].isin(lag_features)
]["importance"].sum()
station_stat_importance = feature_importance_df[
    feature_importance_df["feature"].isin(station_stat_features)
]["importance"].sum()
cyclical_importance = feature_importance_df[
    feature_importance_df["feature"].isin(cyclical_features)
]["importance"].sum()
rush_importance = feature_importance_df[
    feature_importance_df["feature"].isin(rush_features)
]["importance"].sum()
station_onehot_importance = feature_importance_df[
    feature_importance_df["feature"].isin(station_onehot)
]["importance"].sum()

total_importance = feature_importance_df["importance"].sum()

print(
    f"   Lag Features (NEW):       {lag_importance/total_importance*100:5.1f}%  (importance: {lag_importance:.4f})"
)
print(
    f"   Station Stats (NEW):      {station_stat_importance/total_importance*100:5.1f}%  (importance: {station_stat_importance:.4f})"
)
print(
    f"   Cyclical Encoding:        {cyclical_importance/total_importance*100:5.1f}%  (importance: {cyclical_importance:.4f})"
)
print(
    f"   Rush Hour Indicators:     {rush_importance/total_importance*100:5.1f}%  (importance: {rush_importance:.4f})"
)
print(
    f"   Station One-Hot:          {station_onehot_importance/total_importance*100:5.1f}%  (importance: {station_onehot_importance:.4f})"
)

# ==================
# MODELL SPEICHERN
# ==================
print("\n" + "=" * 60)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 60)

model.save(os.path.join(PROJECT_ROOT, "ann", "ann_improved_model_new.keras"))
print("✓ Model saved: ann_improved_model_new.keras")

with open(os.path.join(PROJECT_ROOT, "ann", "ann_improved_scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: ann_improved_scaler.pkl")

with open(os.path.join(PROJECT_ROOT, "ann", "ann_improved_feature_names.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)
print("✓ Feature names saved: ann_improved_feature_names.pkl")

feature_importance_df.to_csv(os.path.join(PROJECT_ROOT, "ann", "feature_importance.csv"), index=False)
print("✓ Feature importance saved: feature_importance.csv")

# ==================
# VISUALISIERUNGEN
# ==================
print("\n" + "=" * 60)
print("CREATING VISUALIZATIONS")
print("=" * 60)

# Plot 1: Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Improved ANN - Training History", fontsize=16, fontweight="bold")

axes[0].plot(history.history["loss"], label="Training Loss", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("MSE Loss", fontsize=12)
axes[0].set_title("Model Loss", fontsize=13)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["mae"], label="Training MAE", linewidth=2)
axes[1].plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("MAE (Minutes)", fontsize=12)
axes[1].set_title("Mean Absolute Error", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "ann", "improved_training_history.png"), dpi=300)
print("✓ Training history saved: improved_training_history.png")
plt.close()

# Plot 2: Predicted vs Actual
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Improved ANN - Predicted vs Actual", fontsize=16, fontweight="bold")

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
plt.savefig(os.path.join(PROJECT_ROOT, "ann", "improved_scatter_plots.png"), dpi=300)
print("✓ Scatter plots saved: improved_scatter_plots.png")
plt.close()

# Plot 3: Model Comparison (if baseline available)
if baseline_available:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Baseline vs Improved Model Comparison", fontsize=16, fontweight="bold"
    )

    models = ["Baseline\n(6 features)", "Improved\n(+7 features)"]
    mae_values = [baseline_test_mae, test_mae]
    r2_values = [baseline_test_r2, test_r2]

    colors = ["#FF6B6B", "#4ECDC4"]

    bars1 = ax1.bar(models, mae_values, color=colors, edgecolor="black", linewidth=2)
    ax1.set_ylabel("MAE (Minutes)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Test MAE Comparison (Lower is Better)", fontsize=14, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mae_values):
        ax1.text(
            i,
            v + 0.01,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Improvement annotation
    improvement_text = f"{mae_improvement:+.1f}%"
    ax1.text(
        0.5,
        max(mae_values) * 0.9,
        improvement_text,
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="green" if mae_improvement > 0 else "red",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
    )

    bars2 = ax2.bar(models, r2_values, color=colors, edgecolor="black", linewidth=2)
    ax2.set_ylabel("R² Score", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Test R² Comparison (Higher is Better)", fontsize=14, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(r2_values):
        ax2.text(
            i,
            v + 0.005,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "ann", "model_comparison.png"), dpi=300)
    print("✓ Model comparison saved: model_comparison.png")
    plt.close()

# Plot 4: Feature Importance (Top 20)
fig, ax = plt.subplots(figsize=(12, 10))

top_20 = feature_importance_df.head(20)
colors_importance = [
    "#2E86AB" if "lag" in f or "station_mean" in f or "station_std" in f else "#A23B72"
    for f in top_20["feature"]
]

ax.barh(
    range(len(top_20)),
    top_20["importance"],
    xerr=top_20["std"],
    color=colors_importance,
    edgecolor="black",
    linewidth=1.5,
)
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20["feature"])
ax.set_xlabel("Permutation Importance", fontsize=12, fontweight="bold")
ax.set_title(
    "Top 20 Most Important Features\n(Blue = New Features)",
    fontsize=14,
    fontweight="bold",
)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "ann", "feature_importance.png"), dpi=300)
print("✓ Feature importance plot saved: feature_importance.png")
plt.close()

# ==================
# FINAL SUMMARY
# ==================
print("\n" + "=" * 60)
print("✅ FEATURE ENGINEERING IMPROVEMENTS COMPLETE!")
print("=" * 60)

print(f"\n📊 RESULTS SUMMARY:")
print(f"\n   BASELINE (Old Features):")
if baseline_available:
    print(
        f"     Features: 6 temporal + {len(station_cols)} stations = {6 + len(station_cols)} total"
    )
    print(f"     Test MAE: {baseline_test_mae:.4f} minutes")
    print(f"     Test R²:  {baseline_test_r2:.4f}")
else:
    print(f"     [Baseline model not available for comparison]")

print(f"\n   IMPROVED (New Features):")
print(f"     Features: 15 temporal + {len(station_cols)} stations = {X.shape[1]} total")
print(f"     Test MAE: {test_mae:.4f} minutes")
print(f"     Test R²:  {test_r2:.4f}")

if baseline_available:
    print(f"\n   🎯 IMPROVEMENT:")
    print(
        f"     MAE Reduction: {mae_improvement:+.2f}% ({baseline_test_mae - test_mae:+.4f} min better)"
    )
    print(f"     R² Improvement: {r2_improvement:+.4f}")

print(f"\n📁 SAVED FILES:")
print(f"   Models:")
print(f"     • ann_improved_model.keras")
print(f"     • ann_improved_scaler.pkl")
print(f"     • ann_improved_feature_names.pkl")
print(f"   Data:")
print(f"     • feature_importance.csv")
print(f"   Visualizations:")
print(f"     • improved_training_history.png")
print(f"     • improved_scatter_plots.png")
if baseline_available:
    print(f"     • model_comparison.png")
print(f"     • feature_importance.png")

print(f"\n🔍 KEY INSIGHTS:")
print(f"   Most important new features:")
top_new_features = feature_importance_df[
    feature_importance_df["feature"].isin(lag_features + station_stat_features)
].head(5)
for idx, row in top_new_features.iterrows():
    print(f"     • {row['feature']:20s}  (importance: {row['importance']:.4f})")

print("\n" + "=" * 60)
