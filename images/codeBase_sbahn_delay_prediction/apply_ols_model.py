import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import statsmodels.api as sm

print("=" * 60)
print("S-BAHN DELAYS - OLS MODEL INFERENCE")
print("=" * 60)

# ==================
# LADE MODELL
# ==================
print("\n1️⃣ Loading OLS Model...")

try:
    # Lade das trainierte OLS-Modell
    with open("/tmp/knowledgeBase/currentOlsSolution.pkl", "rb") as f:
        ols_model = pickle.load(f)
    print("   ✓ OLS Model loaded: currentOlsSolution.pkl")

except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

# ==================
# LADE ACTIVATION DATA
# ==================
print("\n2️⃣ Loading Activation Data...")

try:
    activation_data = pd.read_csv("/tmp/activationBase/activation_data_ols.csv")
    print(f"   ✓ Data loaded: {activation_data.shape}")
except Exception as e:
    print(f"   ✗ Error loading activation data: {e}")
    exit(1)

# ==================
# VORBEREITUNG
# ==================
print("\n3️⃣ Preparing Data...")

# Entferne die Zielvariable falls vorhanden
if "delay_minutes" in activation_data.columns:
    actual_delay = activation_data["delay_minutes"].values[0]
    X = activation_data.drop(columns=["delay_minutes"])
    print(f"   ✓ Actual delay (ground truth): {actual_delay:.2f} minutes")
else:
    actual_delay = None
    X = activation_data

# Füge Konstante hinzu (für OLS Intercept)
X_const = sm.add_constant(X, has_constant="add")
print(f"   ✓ Data prepared with {X_const.shape[1]} features (including constant)")

# ==================
# VORHERSAGE
# ==================
print("\n4️⃣ Making Prediction...")

try:
    prediction = ols_model.predict(X_const)
    predicted_delay = float(prediction[0])
    print(f"   ✓ Prediction: {predicted_delay:.2f} minutes")
except Exception as e:
    print(f"   ✗ Error during prediction: {e}")
    exit(1)

# ==================
# ERGEBNIS SPEICHERN
# ==================
print("\n5️⃣ Saving Results...")

result = {
    "model": "OLS (Ordinary Least Squares)",
    "predicted_delay_minutes": predicted_delay,
    "status": "success",
    "timestamp": pd.Timestamp.now().isoformat(),
    "input_shape": list(X.shape),
    "features_used": X.shape[1],
}

# Füge Ground Truth hinzu falls verfügbar
if actual_delay is not None:
    result["actual_delay_minutes"] = float(actual_delay)
    result["prediction_error"] = float(predicted_delay - actual_delay)

# Speichere als JSON
output_dir = Path("/tmp/ai_system")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "ols_prediction_result.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"   ✓ Results saved: {output_file}")

# ==================
# FINAL OUTPUT
# ==================
print("\n" + "=" * 60)
print("✅ OLS INFERENCE COMPLETE")
print("=" * 60)
print(f"\n📊 RESULT:")
print(f"   Predicted Delay: {predicted_delay:.2f} minutes")
if actual_delay is not None:
    print(f"   Actual Delay:    {actual_delay:.2f} minutes")
    print(f"   Error:           {predicted_delay - actual_delay:+.2f} minutes")
print(f"   Features Used:   {X.shape[1]}")
print("\n" + "=" * 60)
