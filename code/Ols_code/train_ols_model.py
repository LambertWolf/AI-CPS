import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
from scipy.signal import savgol_filter
from sklearn.preprocessing import OneHotEncoder


# ==================
# daten laden
# ==================
df = pd.read_csv("../finaldata/s7_delays.csv")
print(f"Dataset: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst rows:")
print(df.head())
# =================
# Daten Analysieren
# =================
print("\n" + "=" * 60)
print("Verteilung bevor Data cleaining")
print("\n" + "=" * 60)

print(df["delay_minutes"].describe())

print("\n\nPercentiles:")
for p in [50, 75, 90, 95, 98, 99]:
    val = df["delay_minutes"].quantile(p / 100)
    print(f"  {p}% der Fahrten: ≤ {val:.1f} Min")

print("\n\nExtremwerte:")
print(f"  Max delay: {df['delay_minutes'].max()} Min")
print(f"  Min delay: {df['delay_minutes'].min()} Min")

print("\n" + "=" * 60)
print("Perezntilmethode, um viele Verspätungen zu berücksichtigen")
print("=" * 60)
# ==============================================
# 98. Percentile: Behält 96% der mittleren Daten
# ===============================================
lower = df["delay_minutes"].quantile(0.02)  # Bottom 2%
upper = df["delay_minutes"].quantile(0.98)  # Top 2%

before = len(df)
df_clean = df[(df["delay_minutes"] >= lower) & (df["delay_minutes"] <= upper)].copy()
removed = before - len(df_clean)

print(f"\nPercentile-based removal (2% bottom, 2% top):")
print(f"  Lower bound: {lower:.1f} Min")
print(f"  Upper bound: {upper:.1f} Min")
print(f"  Removed: {removed} rows ({removed/before*100:.2f}%)")
print(f"  Remaining: {len(df_clean)} rows")

print("\n" + "=" * 60)
print("Verteilung der Daten nach Bereiningung")
print("=" * 60)
# =============================
# Datenanalyse nach Bereinigung
# =============================
print("\nBasic Statistics:")
print(df_clean["delay_minutes"].describe())

print("\n\nExtreme Values:")
print(f"  Max delay: {df_clean['delay_minutes'].max()} Min")
print(f"  Min delay: {df_clean['delay_minutes'].min()} Min")

# Speichern in csv
df_clean.to_csv("../finaldata/joint_data_collection.csv", index=False)
print(f"\n✓ Saved: joint_data_collection.csv ({len(df_clean)} rows)")
# ========================
# Features hinzufügen
# ========================
print("\n" + "=" * 60)
print("Features hinzufügen")
print("=" * 60)

df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"])

# Ausfiltern der Stunde und des Wochentags
df_clean["hour"] = df_clean["timestamp"].dt.hour
df_clean["weekday"] = df_clean["timestamp"].dt.weekday  # 0=Mo, 6=So

# stop_id Encodieren
# 1. Encoder erstellen
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")

# 2. stop_id encodieren
station_encoded = encoder.fit_transform(df_clean[["stop_id"]])

# 3. Als DataFrame mit Namen
station_cols = [f"station_{int(cat)}" for cat in encoder.categories_[0][1:]]
station_df = pd.DataFrame(station_encoded, columns=station_cols, index=df_clean.index)

# 4. Zusammenführen
X_base = df_clean[["hour", "weekday"]]
X = pd.concat([X_base, station_df], axis=1)
y = df_clean["delay_minutes"]

# 5. Encoder speichern
with open("../ols/station_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print(f"Features vorher: 3")
print(f"Features nachher: {X.shape[1]}")  # Spalten zurück geben


print(f"\n Folgende Features erstellt:")
print(f"  - hour (Stunde des Tages)")
print(f"  - weekday (Wochentag)")
print(f"  - stop_id_encoded (Station)")

# ================
# Train-Test Split
# ================
print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT (80/20)")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(  # ← X und y splitten!
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape} ({len(X_train)/len(df_clean)*100:.1f}%)")
print(f"Test set:     {X_test.shape} ({len(X_test)/len(df_clean)*100:.1f}%)")


# ================
# Normalisieren
# ================

print("\n" + "=" * 60)
print("NORMALISIERUNG (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit NUR auf Train
X_test_scaled = scaler.transform(X_test)  # transform auf Test

print(f"\n✓ Scaler fitted auf Training-Daten")
print(f"✓ Features normalisiert (Mean=0, Std=1)")

print(f"\nVorher (Beispiel):")
print(X_train.head(3))


pd.DataFrame(X_train_scaled, columns=X_train.columns).assign(
    delay_minutes=y_train.values
).to_csv("../finaldata/training_data.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).assign(
    delay_minutes=y_test.values
).to_csv("../finaldata/test_data.csv", index=False)

# Activation Data (1 Eintrag aus Test Set) - ALTE VERSION
# pd.DataFrame(X_test_scaled[:1], columns=X_test.columns).assign(
#     delay_minutes=y_test.values[:1]
# ).to_csv("../finaldata/activation_data.csv", index=False)

# Activation Data - finde einen Eintrag mit delay_minutes = 5
idx_5min = (y_test == 5).values
if idx_5min.any():
    first_5min_idx = idx_5min.argmax()
    pd.DataFrame(X_test_scaled[first_5min_idx:first_5min_idx+1], columns=X_test.columns).assign(
        delay_minutes=y_test.values[first_5min_idx:first_5min_idx+1]
    ).to_csv("../ols/activation_data_ols.csv", index=False)
    print(f"✓ activation_data_ols.csv saved (delay_minutes = 5)")
else:
    # Fallback: nimm den ersten Eintrag
    pd.DataFrame(X_test_scaled[:1], columns=X_test.columns).assign(
        delay_minutes=y_test.values[:1]
    ).to_csv("../ols/activation_data_ols.csv", index=False)
    print(f"✓ activation_data_ols.csv saved (delay_minutes = {y_test.values[0]})")

# ================
# OLS MODELL
# ================
print("\n" + "=" * 60)
print("OLS REGRESSION MODEL")
print("=" * 60)

# Schritt 1: Konstante hinzufügen
# -------------------------------
# Statsmodels braucht eine "Konstante" (Intercept) für die Regression
# Das ist der y-Achsenabschnitt in der Formel: y = b0 + b1*x1 + b2*x2 + ...
X_train_const = sm.add_constant(X_train_scaled)
X_test_const = sm.add_constant(X_test_scaled)

print("\n✓ Konstante hinzugefügt (für Intercept)")

# Schritt 2: OLS Modell trainieren
# ---------------------------------
ols_model = sm.OLS(y_train, X_train_const)
ols_results = ols_model.fit()

print("✓ OLS Modell trainiert")

# Schritt 3: Zusammenfassung anzeigen
# ------------------------------------
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(ols_results.summary())

# Schritt 4: Vorhersagen machen
# ------------------------------
y_train_pred = ols_results.predict(X_train_const)
y_test_pred = ols_results.predict(X_test_const)

print("\n✓ Vorhersagen erstellt")

# Schritt 5: Performance messen
# ------------------------------

# Training Performance
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Test Performance
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)
print(f"\n📊 Training Set:")
print(f"  R² Score:  {train_r2:.4f}  (1.0 = perfekt)")
print(f"  MSE:       {train_mse:.2f}")
print(f"  MAE:       {train_mae:.2f} Min (durchschnittlicher Fehler)")

print(f"\n📊 Test Set:")
print(f"  R² Score:  {test_r2:.4f}  (1.0 = perfekt)")
print(f"  MSE:       {test_mse:.2f}")
print(f"  MAE:       {test_mae:.2f} Min (durchschnittlicher Fehler)")

# Schritt 6: Modell speichern
# ----------------------------

with open("../ols/currentOlsSolution.pkl", "wb") as f:
    pickle.dump(ols_results, f)

print("\n✓ Modell gespeichert: currentOlsSolution.pkl")

# ========================
# VISUALISIERUNGEN
# ========================
print("\n" + "=" * 60)
print("ERSTELLE PLOTS")
print("=" * 60)

# =====================================
# Plot 1: SCATTER PLOTS
# =====================================
fig, axes_scatter = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("OLS Model - Predicted vs Actual Delay", fontsize=16, fontweight="bold")

# Training Set
axes_scatter[0].scatter(y_train, y_train_pred, alpha=0.5, s=20, color="blue")
axes_scatter[0].plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "r--",
    lw=2,
    label="Perfect Prediction",
)
axes_scatter[0].set_xlabel("Actual Delay (Min)", fontsize=12)
axes_scatter[0].set_ylabel("Predicted Delay (Min)", fontsize=12)
axes_scatter[0].set_title(f"Training Set (R²={train_r2:.3f})", fontsize=13)
axes_scatter[0].legend()
axes_scatter[0].grid(True, alpha=0.3)

# Test Set
axes_scatter[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color="green")
axes_scatter[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction",
)
axes_scatter[1].set_xlabel("Actual Delay (Min)", fontsize=12)
axes_scatter[1].set_ylabel("Predicted Delay (Min)", fontsize=12)
axes_scatter[1].set_title(f"Test Set (R²={test_r2:.3f})", fontsize=13)
axes_scatter[1].legend()
axes_scatter[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../ols/scatter_plots.png", dpi=300)
print("✓ Scatter plots gespeichert: scatter_plots.png")
plt.show()
# =====================================
# Plot 2: DIAGNOSTIC PLOTS
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("OLS Diagnostic Plots", fontsize=18, fontweight="bold", y=0.995)

# Residuals berechnen
residuals = (
    (y_test.values - y_test_pred)
    if hasattr(y_test, "values")
    else (y_test - y_test_pred)
)
standardized_residuals = residuals / residuals.std()

# Textbox Properties
props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)

# =====================================
# Plot 1: Residuals vs Fitted (Oben Links)
# =====================================
axes[0, 0].scatter(
    y_test_pred, residuals, alpha=0.4, s=15, color="steelblue", edgecolors="none"
)
axes[0, 0].axhline(y=0, color="red", linestyle="--", lw=2, label="y=0 (Perfect)")
axes[0, 0].set_xlabel("Fitted Values (Predicted Delay)", fontsize=11, fontweight="bold")
axes[0, 0].set_ylabel("Residuals", fontsize=11, fontweight="bold")
axes[0, 0].set_title("Residuals vs Fitted", fontsize=13, fontweight="bold", pad=10)
axes[0, 0].grid(True, alpha=0.25, linestyle="--")
axes[0, 0].legend(loc="upper left", fontsize=9)

textstr = "• to identify non-linearity\n• a roughly horizontal line\n  is an indicator that the\n  residual has a linear pattern"
axes[0, 0].text(
    0.98,
    0.97,
    textstr,
    transform=axes[0, 0].transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
)

# =====================================
# Plot 2: Normal Q-Q Plot (Oben Rechts)
# =====================================
sm.qqplot(
    residuals,
    line="s",
    ax=axes[0, 1],
    markerfacecolor="steelblue",
    markeredgecolor="steelblue",
    markersize=4,
    alpha=0.6,
)
axes[0, 1].set_title("Normal Q-Q", fontsize=13, fontweight="bold", pad=10)
axes[0, 1].set_xlabel("Theoretical Quantiles", fontsize=11, fontweight="bold")
axes[0, 1].set_ylabel("Standardized Residuals", fontsize=11, fontweight="bold")
axes[0, 1].grid(True, alpha=0.25, linestyle="--")

textstr = "• to visually check if residuals\n  are normally distributed\n• points spread along the\n  diagonal line will suggest so"
axes[0, 1].text(
    0.98,
    0.03,
    textstr,
    transform=axes[0, 1].transAxes,
    fontsize=9,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=props,
)

# =====================================
# Plot 3: Scale-Location (Unten Links)
# =====================================
axes[1, 0].scatter(
    y_test_pred,
    np.sqrt(np.abs(standardized_residuals)),
    alpha=0.4,
    s=15,
    color="steelblue",
    edgecolors="none",
)
axes[1, 0].set_xlabel("Fitted Values (Predicted Delay)", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("√|Standardized Residuals|", fontsize=11, fontweight="bold")
axes[1, 0].set_title("Scale-Location", fontsize=13, fontweight="bold", pad=10)
axes[1, 0].grid(True, alpha=0.25, linestyle="--")

sorted_indices = np.argsort(y_test_pred)
x_sorted = y_test_pred[sorted_indices]
y_sorted = np.sqrt(np.abs(standardized_residuals))[sorted_indices]

if len(x_sorted) > 51:
    try:
        y_smooth = savgol_filter(y_sorted, window_length=51, polyorder=3)
        axes[1, 0].plot(
            x_sorted, y_smooth, color="red", linewidth=2, label="Smoothed trend"
        )
        axes[1, 0].legend(loc="upper left", fontsize=9)
    except Exception as e:
        print(f"  ⚠ Smoothing fehlgeschlagen: {e}")

textstr = "• to check homoscedasticity\n  of the residuals\n• a near horizontal red line\n  in the graph would suggest so"
axes[1, 0].text(
    0.98,
    0.97,
    textstr,
    transform=axes[1, 0].transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
)

# =====================================
# Plot 4: Residuals Histogram (Unten Rechts)
# =====================================
axes[1, 1].hist(
    residuals, bins=35, edgecolor="black", alpha=0.7, color="steelblue", linewidth=0.8
)
axes[1, 1].set_xlabel("Residuals", fontsize=11, fontweight="bold")
axes[1, 1].set_ylabel("Frequency", fontsize=11, fontweight="bold")
axes[1, 1].set_title("Residuals Distribution", fontsize=13, fontweight="bold", pad=10)
axes[1, 1].grid(True, alpha=0.25, linestyle="--", axis="y")
axes[1, 1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="Mean=0")
axes[1, 1].legend(loc="upper right", fontsize=9)

textstr = "• to visually check the\n  distribution of residuals\n• normally distributed residuals\n  form a bell-shaped curve"
axes[1, 1].text(
    0.98,
    0.97,
    textstr,
    transform=axes[1, 1].transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=props,
)

# =====================================
# Speichern und anzeigen
# =====================================
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("../ols/diagnostic_plots_second.png", dpi=300, bbox_inches="tight")
print("✓ Professional diagnostic plots gespeichert")
plt.show()
