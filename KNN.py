import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, scaler, cars, X

# 2. TRAIN THE KNN CLASSIFIER
# k=5 is chosen as standard balance; weights='distance' implements the Inverse Distance formula
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

print("Training K-Nearest Neighbors (KNN) Classifier...")
knn_model.fit(X_train_scaled, y_train)
print("Training Complete!\n")

# 3. EVALUATE PERFORMANCE METRICS
y_pred = knn_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("==========================================================")
print("               K-NEAREST NEIGHBORS (KNN) REPORT           ")
print("==========================================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================================\n")


# =========================================================================
# 4. INTERACTIVE PRICE PREDICTION FOR USER INPUT
# =========================================================================

# Baseline prices assigned to each tier (Matching the project requirements)
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
    print("        CUSTOM CAR PRICE RECOMMENDATION SYSTEM (KNN)      ")
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
        
        # --- BRAND VALUE ENCODING ---
        print("\nEnter Car Brand Name (e.g., Maruti, Hyundai, BMW):")
        brand_input = input("Brand: ").strip().capitalize()
        
        # Pull the mean value from the dictionary created in loaddata.py
        # Fallback to the overall database median if the brand is unrecognized
        brand_means = cars.groupby("brand")["Brand_Encoded"].first()
        if brand_input in brand_means.index:
            brand_encoded = brand_means[brand_input]
        else:
            brand_encoded = cars["Brand_Encoded"].median()
            print(f"-> Unrecognized brand. Assigning generic market weight: {brand_encoded:.2f}")

        # --- MENU CHOICE: CONDITION RATING ---
        print("\nSelect Physical Condition Rating:")
        print("  1 Star  : Poor (-20%)")
        print("  2 Stars : Below Average (-10%)")
        print("  3 Stars : Good / Fair (Standard Market)")
        print("  4 Stars : Very Good (+10%)")
        print("  5 Stars : Excellent / Like New (+20%)")
        condition_choice = int(input("Enter rating (1-5): "))
        
        if condition_choice not in CONDITION_MULTIPLIERS:
            print("Invalid rating choice. Defaulting to 3 Stars (Good).")
            condition_choice = 3

        # =========================================================================
        # 5. MAP USER INPUT TO ONE-HOT ENCODED COLUMNS
        # =========================================================================
        # Create a base dictionary mapping all features from the original X matrix to 0
        input_data = {col: 0.0 for col in X.columns}
        
        # Assign numeric entry coordinates
        input_data['year'] = year
        input_data['km_driven'] = km_driven
        input_data['mileage(km/ltr/kg)'] = mileage
        input_data['engine'] = engine
        input_data['max_power'] = max_power
        input_data['seats'] = seats
        input_data['Brand_Encoded'] = brand_encoded
        
        # Assign encoded categories based on user choices matching your dummy options
        if fuel_choice == '2' and 'fuel_Petrol' in input_data: input_data['fuel_Petrol'] = 1
        elif fuel_choice == '3' and 'fuel_LPG' in input_data: input_data['fuel_LPG'] = 1
        # Diesel acts as the dropped baseline dummy column if your code dropped first
        
        if trans_choice == '2' and 'transmission_Automatic' in input_data: 
            input_data['transmission_Automatic'] = 1
            
        if seller_choice == '2' and 'seller_type_Dealer' in input_data: input_data['seller_type_Dealer'] = 1
        elif seller_choice == '3' and 'seller_type_Trustmark Dealer' in input_data: input_data['seller_type_Trustmark Dealer'] = 1

        # Transform dictionary entries into a row vector matching feature sequence
        user_df = pd.DataFrame([input_data])[X.columns]
        
        # Scale inputs using parameters fit from training set
        user_scaled = scaler.transform(user_df)
        
        # =========================================================================
        # 6. RUN PREDICTION AND APPLY PRICE PIPELINE
        # =========================================================================
        predicted_tier = knn_model.predict(user_scaled)[0]
        
        base_price = TIER_BASE_PRICES[predicted_tier]
        multiplier = CONDITION_MULTIPLIERS[condition_choice]
        final_predicted_price = base_price * multiplier
        
        print("\n==========================================================")
        print("                 VALUATION RESULT REPORT                  ")
        print("==========================================================")
        print(f"Assigned Market Tier : {predicted_tier}")
        print(f"Tier Base Value      : INR {base_price:,.2f}")
        print(f"Condition Multiplier : {multiplier:.2f}x")
        print(f"Final Estimated Price: INR {final_predicted_price:,.2f}")
        print("==========================================================\n")
        
    except ValueError:
        print("\nError: Please make sure to input valid numbers for vehicle statistics.")
    except Exception as e:
        print(f"\nAn error occurred during prediction pipeline execution: {e}")

# Run user evaluation loop if script is called directly
if __name__ == "__main__":
    predict_user_car()
