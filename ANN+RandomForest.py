import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score

from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, y_train_price, y_test_price, scaler, cars, brand_means


le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


ann_clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
rf_clf = RandomForestClassifier(n_estimators=150, random_state=42)

ann_reg = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=42)
rf_reg = RandomForestRegressor(n_estimators=150, random_state=42)

print("Training Hybrid ANN + Random Forest Models...")
ann_clf.fit(X_train_scaled, y_train_enc)
rf_clf.fit(X_train_scaled, y_train_enc)
ann_reg.fit(X_train_scaled, y_train_price)
rf_reg.fit(X_train_scaled, y_train_price)
print("Training Complete!\n")


ann_probs = ann_clf.predict_proba(X_test_scaled)
rf_probs = rf_clf.predict_proba(X_test_scaled)
ensemble_probs = (ann_probs + rf_probs) / 2.0
y_pred_enc = np.argmax(ensemble_probs, axis=1)


ann_price_preds = ann_reg.predict(X_test_scaled)
rf_price_preds = rf_reg.predict(X_test_scaled)
ensemble_price_preds = (ann_price_preds + rf_price_preds) / 2.0

print("==========================================================")
print("      ANN + RANDOM FOREST ENSEMBLE EVALUATION REPORT     ")
print("==========================================================")
print("--- Classification (Price Tier) ---")
print(f"Accuracy       : {accuracy_score(y_test_enc, y_pred_enc) * 100:.2f}%")
print(f"Precision      : {precision_score(y_test_enc, y_pred_enc, average='weighted', zero_division=0) * 100:.2f}%")
print(f"Recall         : {recall_score(y_test_enc, y_pred_enc, average='weighted') * 100:.2f}%")
print(f"F1-Score       : {f1_score(y_test_enc, y_pred_enc, average='weighted') * 100:.2f}%")
print("\n--- Regression (Selling Price) ---")
print(f"Price MAE      : ${mean_absolute_error(y_test_price, ensemble_price_preds):,.2f}")
print(f"Price R2-Score : {r2_score(y_test_price, ensemble_price_preds):.4f}")
print("==========================================================\n")


joblib.dump({
    "ann_clf": ann_clf, "rf_clf": rf_clf,
    "ann_reg": ann_reg, "rf_reg": rf_reg,
    "le": le
}, "ann_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")


CONDITION_MULTIPLIERS = {1: 0.80, 2: 0.90, 3: 1.00, 4: 1.10, 5: 1.20}

def predict_user_car():
    print("\n==========================================================")
    print(" ANN + RANDOM FOREST CAR PRICE RECOMMENDATION SYSTEM     ")
    print("==========================================================")
    try:
        year = float(input("Enter Year (e.g., 2017): "))
        km_driven = float(input("Enter Kilometers Driven (e.g., 45000): "))
        mileage = float(input("Enter Mileage in km/l (e.g., 21.5): "))
        engine = float(input("Enter Engine CC (e.g., 1248): "))
        max_power = float(input("Enter Max Power in bhp (e.g., 85.0): "))
        seats = float(input("Enter Number of Seats (e.g., 5): "))
        
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
        print("  1 Star  : Poor (-20%)\n  2 Stars : Below Average (-10%)\n  3 Stars : Good / Fair (Standard Market)\n  4 Stars : Very Good (+10%)\n  5 Stars : Excellent / Like New (+20%)")
        condition_stars = int(input("Enter Rating (1-5): "))

        fuel_Diesel = 1 if fuel_choice == "1" else 0
        fuel_Petrol = 1 if fuel_choice == "2" else 0
        fuel_LPG    = 1 if fuel_choice == "3" else 0
        transmission_Manual = 1 if trans_choice == "1" else 0
        seller_Individual   = 1 if seller_choice == "1" else 0
        seller_Trustmark    = 1 if seller_choice == "3" else 0
        
        print("\nEnter Car Brand Name (e.g., Maruti, Hyundai, BMW):")
        brand_input = input("Brand: ").strip().capitalize()

        matched_brand = next((b for b in brand_means.index if b.lower() == brand_input.lower()), None)
        brand_encoded = brand_means[matched_brand] if matched_brand else brand_means.mean()
        
        input_features = np.array([[
            year, km_driven, mileage, engine, max_power, seats,
            fuel_Diesel, fuel_LPG, fuel_Petrol,
            transmission_Manual, seller_Individual, seller_Trustmark,
            brand_encoded
        ]])
        
        input_scaled = scaler.transform(input_features)
        
        
        p_ann = ann_clf.predict_proba(input_scaled)[0]
        p_rf = rf_clf.predict_proba(input_scaled)[0]
        combined_probs = (p_ann + p_rf) / 2.0
        predicted_idx = np.argmax(combined_probs)
        predicted_tier = le.inverse_transform([predicted_idx])[0]
        
        
        price_ann = float(ann_reg.predict(input_scaled)[0])
        price_rf = float(rf_reg.predict(input_scaled)[0])
        base_price = (price_ann + price_rf) / 2.0

        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier
        
      
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"Ensemble Predicted Tier : {predicted_tier}")
        print("Ensemble Confidence Breakdown (ANN + Random Forest):")
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