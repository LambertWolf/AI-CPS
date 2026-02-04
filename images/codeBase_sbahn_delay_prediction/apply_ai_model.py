import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import json
from pathlib import Path

print("=" * 60)
print("S7 DELAYS - AI MODEL INFERENCE")
print("=" * 60)

# ==================
# LADE MODELL & ARTEFAKTE
# ==================
print("\n1️⃣ Loading Model and Artifacts...")

try:
    # Lade das trainierte Modell
    # safe_mode=False erlaubt das Laden von Modellen mit neueren Keras-Versionen
    model = keras.models.load_model(
        "/tmp/knowledgeBase/ann_improved_model_new.keras", safe_mode=False
    )
    print("   ✓ Model loaded: ann_improved_model.keras")

    # Lade Scaler
    with open("/tmp/knowledgeBase/ann_improved_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("   ✓ Scaler loaded")

    # Lade Feature Namen
    with open("/tmp/knowledgeBase/ann_improved_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print(f"   ✓ Feature names loaded ({len(feature_names)} features)")

except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    exit(1)

# ==================
# LADE ACTIVATION DATA
# ==================
print("\n2️⃣ Loading Activation Data...")

try:
    activation_data = pd.read_csv("/tmp/activationBase/activation_data.csv")
    print(f"   ✓ Data loaded: {activation_data.shape}")
except Exception as e:
    print(f"   ✗ Error loading activation data: {e}")
    exit(1)

# ==================
# VALIDIERE FEATURES
# ==================
print("\n3️⃣ Validating Features...")

missing_features = set(feature_names) - set(activation_data.columns)
if missing_features:
    print(f"   ✗ Missing features: {missing_features}")
    exit(1)

print(f"   ✓ All {len(feature_names)} features present")

# ==================
# VORBEREITUNG
# ==================
print("\n4️⃣ Preparing Data...")

# Entferne die Zielvariable falls vorhanden und speichere Ground Truth
if "delay_minutes" in activation_data.columns:
    actual_delay = activation_data["delay_minutes"].values[0]
    print(f"   ✓ Actual delay (ground truth): {actual_delay:.2f} minutes")
else:
    actual_delay = None

# Selektiere nur die benötigten Features in der richtigen Reihenfolge
X = activation_data[feature_names]

# Normalisiere mit dem trainierten Scaler
X_scaled = scaler.transform(X)
print("   ✓ Data normalized")

# ==================
# VORHERSAGE
# ==================
print("\n5️⃣ Making Prediction...")

try:
    prediction = model.predict(X_scaled, verbose=0)
    predicted_delay = float(prediction[0][0])
    print(f"   ✓ Prediction: {predicted_delay:.2f} minutes")
except Exception as e:
    print(f"   ✗ Error during prediction: {e}")
    exit(1)

# ==================
# ERGEBNIS SPEICHERN
# ==================
print("\n6️⃣ Saving Results...")

result = {
    "model": "ANN (Improved Features)",
    "predicted_delay_minutes": predicted_delay,
    "status": "success",
    "timestamp": pd.Timestamp.now().isoformat(),
    "input_shape": list(X.shape),
    "features_used": len(feature_names),
}

# Füge Ground Truth hinzu falls verfügbar
if actual_delay is not None:
    result["actual_delay_minutes"] = float(actual_delay)
    result["prediction_error"] = float(predicted_delay - actual_delay)

# Speichere als JSON
output_dir = Path("/tmp/ai_system")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "ai_prediction_result.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"   ✓ Results saved: {output_file}")

# ==================
# FINAL OUTPUT
# ==================
print("\n" + "=" * 60)
print("✅ INFERENCE COMPLETE")
print("=" * 60)
print(f"\n📊 RESULT:")
print(f"   Predicted Delay: {predicted_delay:.2f} minutes")
if actual_delay is not None:
    print(f"   Actual Delay:    {actual_delay:.2f} minutes")
    print(f"   Error:           {predicted_delay - actual_delay:+.2f} minutes")
print(f"   Input Records: {X.shape[0]}")
print(f"   Features Used: {len(feature_names)}")
print("\n" + "=" * 60)
