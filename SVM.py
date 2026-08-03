import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, scaler, cars

# 2. TRAIN THE SVM CLASSIFIER
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)

print("Training Support Vector Machine (SVM) Classifier...")
svm_model.fit(X_train_scaled, y_train)
print("Training Complete!\n")

# 3. EVALUATE PERFORMANCE METRICS
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("==========================================================")
print("             SUPPORT VECTOR MACHINE (SVM) REPORT          ")
print("==========================================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================================\n")

# =========================================================================
# 4. INTERACTIVE PRICE PREDICTION FOR USER INPUT
# =========================================================================

# Baseline prices assigned to each tier (Adjust these figures if needed)
TIER_BASE_PRICES = {
    "Low": 250000,
    "Medium": 550000,
    "High": 1000000
}

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

        # --- CONVERT MENU SELECTIONS TO BINARY FLAGS (0 or 1) ---
        fuel_Diesel = 1 if fuel_choice == "1" else 0
        fuel_Petrol = 1 if fuel_choice == "2" else 0
        fuel_LPG    = 1 if fuel_choice == "3" else 0
        
        transmission_Manual = 1 if trans_choice == "1" else 0
        
        seller_Individual = 1 if seller_choice == "1" else 0
        seller_Trustmark  = 1 if seller_choice == "3" else 0
        
        # Fallback target average tier encoding (~2.0 for standard average)
        brand_encoded = 2.0 
        
        # --- ASSEMBLE FEATURE ARRAY ---
        input_features = np.array([[
            year, km_driven, mileage, engine, max_power, seats,
            fuel_Diesel, fuel_LPG, fuel_Petrol,
            transmission_Manual, seller_Individual, seller_Trustmark,
            brand_encoded
        ]])
        
        # --- SCALE FEATURES & PREDICT ---
        input_scaled = scaler.transform(input_features)
        predicted_tier = svm_model.predict(input_scaled)[0]
        
        base_price = TIER_BASE_PRICES.get(predicted_tier, 500000)
        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier
        
        # --- DISPLAY PREDICTION SUMMARY ---
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"Predicted Market Tier : {predicted_tier}")
        print(f"Base Market Value     : ${base_price:,.2f}")
        print(f"Condition Rating      : {condition_stars} Star(s) (Multiplier: {multiplier:.2f}x)")
        print(f"FINAL RECOMMENDED PRICE: ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")

# Run interactive user testing
if __name__ == "__main__":
    predict_user_car()