import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_test_scaled, X_train_scaled, cars, scaler, y_test, y_train

# 2. ENCODE STRING LABELS TO INTEGERS FOR XGBOOST (0, 1, 2)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 3. DEFINE BASE MODELS
svm_base = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
xgb_base = XGBClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    eval_metric="mlogloss",
    random_state=42,
)

# 4. CREATE THE HYBRID ENSEMBLE (SVM + XGBOOST)
# Soft voting computes the weighted average of predicted probabilities
ensemble_model = VotingClassifier(
    estimators=[("svm", svm_base), ("xgb", xgb_base)],
    voting="soft",
    weights=[1, 1],  # 50% SVM, 50% XGBoost
)

print("Training Hybrid SVM + XGBoost Classifier...")
ensemble_model.fit(X_train_scaled, y_train_encoded)
print("Training Complete!\n")

# 5. EVALUATE PERFORMANCE METRICS
y_pred_encoded = ensemble_model.predict(X_test_scaled)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("==========================================================")
print("       HYBRID ENSEMBLE (SVM + XGBOOST) REPORT            ")
print("==========================================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================================\n")

# =========================================================================
# 6. INTERACTIVE PRICE PREDICTION FOR USER INPUT
# =========================================================================

TIER_BASE_PRICES = {"Low": 250000, "Medium": 550000, "High": 1000000}

CONDITION_MULTIPLIERS = {
    1: 0.80,  # Poor (-20%)
    2: 0.90,  # Below Average (-10%)
    3: 1.00,  # Good / Fair (Standard Market Value)
    4: 1.10,  # Very Good (+10%)
    5: 1.20,  # Excellent / Like New (+20%)
}


def predict_user_car():
  print("\n==========================================================")
  print("        CUSTOM CAR PRICE RECOMMENDATION SYSTEM            ")
  print("==========================================================")

  try:
    # --- NUMERIC INPUTS ---
    year = float(input("Enter Year (e.g., 2017): "))
    km_driven = float(input("Enter Kilometers Driven (e.g., 45000): "))
    mileage = float(input("Enter Mileage in km/l (e.g., 21.5): "))
    engine = float(input("Enter Engine CC (e.g., 1248): "))
    max_power = float(input("Enter Max Power in bhp (e.g., 85.0): "))
    seats = float(input("Enter Number of Seats (e.g., 5): "))

    # --- MENU CHOICE: FUEL TYPE ---
    print("\nSelect Fuel Type:")
    print("  [1] Diesel")
    print("  [2] Petrol")
    print("  [3] LPG")
    print("  [4] CNG / Other")
    fuel_choice = input("Enter choice (1-4): ").strip()

    # --- MENU CHOICE: TRANSMISSION ---
    print("\nSelect Transmission Type:")
    print("  [1] Manual")
    print("  [2] Automatic")
    trans_choice = input("Enter choice (1-2): ").strip()

    # --- MENU CHOICE: SELLER TYPE ---
    print("\nSelect Seller Type:")
    print("  [1] Individual")
    print("  [2] Dealer")
    print("  [3] Trustmark Dealer")
    seller_choice = input("Enter choice (1-3): ").strip()

    # --- MENU CHOICE: CONDITION RATING ---
    print("\nSelect Physical Condition Rating:")
    print("  1 Star  : Poor (-20%)")
    print("  2 Stars : Below Average (-10%)")
    print("  3 Stars : Good / Fair (Standard Market)")
    print("  4 Stars : Very Good (+10%)")
    print("  5 Stars : Excellent / Like New (+20%)")
    condition_stars = int(input("Enter Rating (1-5): "))

    # --- CONVERT MENU SELECTIONS TO BINARY FLAGS ---
    fuel_Diesel = 1 if fuel_choice == "1" else 0
    fuel_Petrol = 1 if fuel_choice == "2" else 0
    fuel_LPG = 1 if fuel_choice == "3" else 0

    transmission_Manual = 1 if trans_choice == "1" else 0

    seller_Individual = 1 if seller_choice == "1" else 0
    seller_Trustmark = 1 if seller_choice == "3" else 0

    brand_encoded = 2.0

    # --- ASSEMBLE FEATURE ARRAY ---
    input_features = np.array([[
        year,
        km_driven,
        mileage,
        engine,
        max_power,
        seats,
        fuel_Diesel,
        fuel_LPG,
        fuel_Petrol,
        transmission_Manual,
        seller_Individual,
        seller_Trustmark,
        brand_encoded,
    ]])

    # --- SCALE FEATURES & PREDICT VIA ENSEMBLE ---
    input_scaled = scaler.transform(input_features)
    predicted_encoded = ensemble_model.predict(input_scaled)[0]
    predicted_tier = label_encoder.inverse_transform([predicted_encoded])[0]

    # --- CALCULATE CONFIDENCE SCORES ---
    probabilities = ensemble_model.predict_proba(input_scaled)[0]

    base_price = TIER_BASE_PRICES.get(predicted_tier, 500000)
    multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
    final_recommended_price = base_price * multiplier

    # --- DISPLAY PREDICTION SUMMARY ---
    print("\n----------------------------------------------------------")
    print("                 PREDICTION RESULTS                       ")
    print("----------------------------------------------------------")
    print(f"Predicted Market Tier : {predicted_tier}")

    print("Prediction Confidence Breakdown:")
    for class_name, prob in zip(label_encoder.classes_, probabilities):
        print(f"  - {class_name:6s} Tier : {prob * 100:.2f}%")
        
    print(
        f"Condition Rating      : {condition_stars} Star(s) (Multiplier:"
        f" {multiplier:.2f}x)"
    )
    print(f"FINAL RECOMMENDED PRICE: ${final_recommended_price:,.2f}")
    print("==========================================================\n")

  except ValueError:
    print(
        "\n[Error] Invalid input! Please enter numbers for specs and menu"
        " selections."
    )

print("Saving model artifacts...")
# 1. Fit standalone SVM explicitly to ensure it is 100% fitted
svm_base.fit(X_train_scaled, y_train_encoded)
joblib.dump(svm_base, "svm_model.pkl")

# 2. Fit and save ensemble
joblib.dump(ensemble_model, "svm_xgb_model.pkl")

# 3. Save preprocessors
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("All artifacts successfully saved to .pkl files!\n")

# Run interactive user testing
if __name__ == "__main__":
  predict_user_car()