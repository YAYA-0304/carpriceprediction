import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
import joblib

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, y_train_price, y_test_price, scaler, cars, brand_means

# 2. TRAIN THE SVM CLASSIFIER (Tier) & SVR (Price)
svm_classifier = SVC(kernel='rbf', C=1.0, probability=True, decision_function_shape='ovo', random_state=42)
svm_regressor = SVR(kernel='rbf', C=1000.0, epsilon=0.1)

print("Training Support Vector Machine (SVM) Classifier & SVR Regressor...")
svm_classifier.fit(X_train_scaled, y_train)
svm_regressor.fit(X_train_scaled, y_train_price)
print("Training Complete!\n")

# 3. EVALUATE PERFORMANCE METRICS
y_pred_tier = svm_classifier.predict(X_test_scaled)
y_pred_price = svm_regressor.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_tier)
precision = precision_score(y_test, y_pred_tier, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_tier, average='weighted')
f1 = f1_score(y_test, y_pred_tier, average='weighted')
mae = mean_absolute_error(y_test_price, y_pred_price)
r2 = r2_score(y_test_price, y_pred_price)

print("==========================================================")
print("             SUPPORT VECTOR MACHINE (SVM) REPORT          ")
print("==========================================================")
print(f"Tier Accuracy  : {accuracy * 100:.2f}%")
print(f"Tier Precision : {precision * 100:.2f}%")
print(f"Tier Recall    : {recall * 100:.2f}%")
print(f"Tier F1-Score  : {f1 * 100:.2f}%")
print(f"Price MAE      : ${mae:,.2f}")
print(f"Price R2-Score : {r2:.4f}")
print("==========================================================\n")

joblib.dump({"classifier": svm_classifier, "regressor": svm_regressor}, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved trained SVM models to 'svm_model.pkl' and scaler to 'scaler.pkl'.\n")

# Multipliers based on vehicle physical condition (1 to 5 Stars)
CONDITION_MULTIPLIERS = {
    1: 0.80,  # Poor (-20%)
    2: 0.90,  # Below Average (-10%)
    3: 1.00,  # Good / Fair (Standard Market Value)
    4: 1.10,  # Very Good (+10%)
    5: 1.20   # Excellent / Like New (+20%)
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
        fuel_LPG    = 1 if fuel_choice == "3" else 0
        
        transmission_Manual = 1 if trans_choice == "1" else 0
        
        seller_Individual = 1 if seller_choice == "1" else 0
        seller_Trustmark  = 1 if seller_choice == "3" else 0
        
        print("\nEnter Car Brand Name (e.g., Maruti, Hyundai, BMW):")
        brand_input = input("Brand: ").strip().lower()

        matched_brand = None
        for known_brand in brand_means.index:
            if known_brand.lower() == brand_input.lower():
                matched_brand = known_brand
                break

        if matched_brand is not None:
            brand_encoded = brand_means[matched_brand]
            print(f"-> Recognized Brand! Market Weight: {brand_encoded:.2f}")
        else:
            brand_encoded = brand_means.mean()
            print(f"-> Unrecognized brand '{brand_input}'. Assigning average weight: {brand_encoded:.2f}")
        
        # --- ASSEMBLE FEATURE ARRAY ---
        input_features = np.array([[
            year, km_driven, mileage, engine, max_power, seats,
            fuel_Diesel, fuel_LPG, fuel_Petrol,
            transmission_Manual, seller_Individual, seller_Trustmark,
            brand_encoded
        ]])
        
        # --- SCALE FEATURES & PREDICT ---
        input_scaled = scaler.transform(input_features)

        # 1. Predict Tier (Classification)
        probabilities = svm_classifier.predict_proba(input_scaled)[0]
        class_labels = svm_classifier.classes_
        predicted_tier = class_labels[np.argmax(probabilities)]
        
        # 2. Predict Base Price (SVR Regression)
        base_price = float(svm_regressor.predict(input_scaled)[0])

        # 3. Apply Condition Multiplier
        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier
        
        # --- DISPLAY PREDICTION RESULTS ---
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"Predicted Market Tier : {predicted_tier}")
        print("Prediction Confidence Breakdown:")

        for label, prob in zip(class_labels, probabilities):
            print(f"  - {label:<6} Tier : {prob * 100:.2f}%")

        print(f"Condition Rating      : {condition_stars} Star(s) (Multiplier: {multiplier:.2f}x)")
        print(f"Dynamic Base Value    : ${base_price:,.2f}")
        print(f"FINAL RECOMMENDED PRICE: ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")

if __name__ == "__main__":
    predict_user_car()