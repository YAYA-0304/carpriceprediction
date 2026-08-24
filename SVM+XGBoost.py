import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, scaler, cars

# 2. ENCODE LABELS (Required for XGBoost: Low->0, Medium->1, High->2)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 3. TRAIN MODELS (SVM + XGBoost)
svm_model = SVC(kernel='rbf', C=1.0, probability=True, decision_function_shape='ovo', random_state=42)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric='mlogloss')

print("Training Ensemble Models (SVM + XGBoost)...")
svm_model.fit(X_train_scaled, y_train_enc)
xgb_model.fit(X_train_scaled, y_train_enc)
print("Training Complete!\n")

# 4. ENSEMBLE EVALUATION (Soft Voting)
svm_probs = svm_model.predict_proba(X_test_scaled)
xgb_probs = xgb_model.predict_proba(X_test_scaled)

ensemble_probs = (svm_probs + xgb_probs) / 2.0
y_pred_enc = np.argmax(ensemble_probs, axis=1)

accuracy = accuracy_score(y_test_enc, y_pred_enc)
precision = precision_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0)
recall = recall_score(y_test_enc, y_pred_enc, average='weighted')
f1 = f1_score(y_test_enc, y_pred_enc, average='weighted')

print("==========================================================")
print("          SVM + XGBOOST ENSEMBLE EVALUATION REPORT        ")
print("==========================================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================================\n")

joblib.dump({"svm": svm_model, "xgb": xgb_model, "le": le}, "svm_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved models to 'svm_xgb_model.pkl' and scaler to 'scaler.pkl'.\n")

# =========================================================================
# 5. INTERACTIVE PRICE PREDICTION (DYNAMIC LOOKUP + CONFIDENCE)
# =========================================================================

CONDITION_MULTIPLIERS = {
    1: 0.80,  # Poor (-20%)
    2: 0.90,  # Below Average (-10%)
    3: 1.00,  # Good / Fair (Standard Market Value)
    4: 1.10,  # Very Good (+10%)
    5: 1.20   # Excellent / Like New (+20%)
}

def predict_user_car():
    print("\n==========================================================")
    print("   SVM + XGBOOST CAR PRICE RECOMMENDATION SYSTEM          ")
    print("==========================================================")
    
    try:
        # --- NUMERIC INPUTS ---
        year = float(input("Enter Year (e.g., 2017): "))
        km_driven = float(input("Enter Kilometers Driven (e.g., 45000): "))
        mileage = float(input("Enter Mileage in km/l (e.g., 21.5): "))
        engine = float(input("Enter Engine CC (e.g., 1248): "))
        max_power = float(input("Enter Max Power in bhp (e.g., 85.0): "))
        seats = float(input("Enter Number of Seats (e.g., 5): "))
        
        # --- MENU CHOICES ---
        print("\nSelect Fuel Type:")
        print("  [1] Diesel | [2] Petrol | [3] LPG | [4] CNG / Other")
        fuel_choice = input("Enter choice (1-4): ").strip()
        
        print("\nSelect Transmission Type:")
        print("  [1] Manual | [2] Automatic")
        trans_choice = input("Enter choice (1-2): ").strip()
        
        print("\nSelect Seller Type:")
        print("  [1] Individual | [2] Dealer | [3] Trustmark Dealer")
        seller_choice = input("Enter choice (1-3): ").strip()
        
        print("\nSelect Physical Condition Rating:")
        print("  1 Star  : Poor (-20%)")
        print("  2 Stars : Below Average (-10%)")
        print("  3 Stars : Good / Fair (Standard Market)")
        print("  4 Stars : Very Good (+10%)")
        print("  5 Stars : Excellent / Like New (+20%)")
        condition_stars = int(input("Enter Rating (1-5): "))

        # --- ONE-HOT BINARY ENCODING ---
        fuel_Diesel = 1 if fuel_choice == "1" else 0
        fuel_Petrol = 1 if fuel_choice == "2" else 0
        fuel_LPG    = 1 if fuel_choice == "3" else 0
        
        transmission_Manual = 1 if trans_choice == "1" else 0
        seller_Individual   = 1 if seller_choice == "1" else 0
        seller_Trustmark    = 1 if seller_choice == "3" else 0
        print("\nEnter Car Brand Name (e.g., Maruti, Hyundai, BMW):")
        brand_input = input("Brand: ").strip().capitalize()

        brand_means = cars.groupby("brand")["Brand_Encoded"].first()

        if brand_input in brand_means.index:
            brand_encoded = brand_means[brand_input]
        else:
            brand_encoded = cars["Brand_Encoded"].median()
        print(f"Unrecognized brand. Assigning generic market weight: {brand_encoded:.2f}")
        
        # --- ASSEMBLE FEATURE ARRAY ---
        input_features = np.array([[
            year, km_driven, mileage, engine, max_power, seats,
            fuel_Diesel, fuel_LPG, fuel_Petrol,
            transmission_Manual, seller_Individual, seller_Trustmark,
            brand_encoded
        ]])
        
        # --- SCALE FEATURES & PREDICT ENSEMBLE PROBABILITIES ---
        input_scaled = scaler.transform(input_features)
        
        p_svm = svm_model.predict_proba(input_scaled)[0]
        p_xgb = xgb_model.predict_proba(input_scaled)[0]
        
        # Combine model probabilities (Soft Voting)
        combined_probs = (p_svm + p_xgb) / 2.0
        predicted_idx = np.argmax(combined_probs)
        predicted_tier = le.inverse_transform([predicted_idx])[0]
        
        # --- DYNAMIC BASELINE LOOKUP FROM DATASET ---
        price_col = "selling_price" if "selling_price" in cars.columns else "Price"
        year_col = "year" if "year" in cars.columns else ("Year" if "Year" in cars.columns else None)

        if year_col and year_col in cars.columns:
            similar_cars = cars[
                (cars["Price_Tier"] == predicted_tier)
                & (cars[year_col].between(year - 2, year + 2))
            ]
        else:
            similar_cars = pd.DataFrame()

        if len(similar_cars) >= 3:
            base_price = float(similar_cars[price_col].median())
        else:
            base_price = float(cars[cars["Price_Tier"] == predicted_tier][price_col].median())

        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier
        
        # --- DISPLAY PREDICTION RESULTS & CONFIDENCE ---
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"Ensemble Predicted Tier : {predicted_tier}")
        print("Ensemble Confidence Breakdown (SVM + XGBoost):")

        for idx, class_name in enumerate(le.classes_):
            print(f"  - {class_name:<6} Tier : {combined_probs[idx] * 100:.2f}%")

        print(f"Condition Rating        : {condition_stars} Star(s) (Multiplier: {multiplier:.2f}x)")
        print(f"Dynamic Base Value      : ${base_price:,.2f}")
        print(f"FINAL RECOMMENDED PRICE : ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")

if __name__ == "__main__":
    predict_user_car()