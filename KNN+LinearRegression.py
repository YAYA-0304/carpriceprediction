import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_test_scaled, X_train_scaled, cars, scaler, y_test, y_train, X

# 2. ENCODE STRING LABELS TO INTEGERS FOR ENSEMBLE CONSISTENCY (0, 1, 2)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 3. DEFINE BASE MODELS
# KNN using inverse distance weighting and Euclidean metric
knn_base = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

# Logistic Regression (Linear Classification model) configured for multi-class data
lr_base = LogisticRegression(max_iter=1000, multi_class='multinomial', random_state=42)

# 4. CREATE THE HYBRID ENSEMBLE (KNN + LINEAR CLASSIFIER)
# Soft voting computes the weighted average of predicted probabilities
ensemble_model = VotingClassifier(
    estimators=[("knn", knn_base), ("linear_reg", lr_base)],
    voting="soft",
    weights=[1, 1],  # 50% K-Nearest Neighbors, 50% Linear Logistic Regression
)

print("Training Hybrid KNN + Linear Regression Classifier...")
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
print("       HYBRID ENSEMBLE (KNN + LINEAR REG) REPORT         ")
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
    print("     CUSTOM CAR PRICE RECOMMENDATION SYSTEM (HYBRID)       ")
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
        
        # Resolve target numerical encoding matching loaddata.py
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
        condition_stars = int(input("Enter Rating (1-5): "))

        # =========================================================================
        # 7. MAP USER INPUT TO ONE-HOT ENCODED MATRIX
        # =========================================================================
        # Initialize an empty row matching the exact layout of data features
        input_data = {col: 0.0 for col in X.columns}
        
        # Populate basic numerical parameters
        input_data['year'] = year
        input_data['km_driven'] = km_driven
        input_data['mileage(km/ltr/kg)'] = mileage
        input_data['engine'] = engine
        input_data['max_power'] = max_power
        input_data['seats'] = seats
        input_data['Brand_Encoded'] = brand_encoded
        
        # Process the one-hot columns based on the dummy structure inside loaddata.py
        if fuel_choice == '2' and 'fuel_Petrol' in input_data: input_data['fuel_Petrol'] = 1
        elif fuel_choice == '3' and 'fuel_LPG' in input_data: input_data['fuel_LPG'] = 1
        # If choice is '1' (Diesel), both remain 0, representing the dropped baseline feature
        
        if trans_choice == '2' and 'transmission_Automatic' in input_data: 
            input_data['transmission_Automatic'] = 1
            
        if seller_choice == '2' and 'seller_type_Dealer' in input_data: input_data['seller_type_Dealer'] = 1
        elif seller_choice == '3' and 'seller_type_Trustmark Dealer' in input_data: input_data['seller_type_Trustmark Dealer'] = 1

        # Transform dictionary vector into a sorted DataFrame matching training sequence
        user_df = pd.DataFrame([input_data])[X.columns]
        
        # Scale features & predict via ensemble
        input_scaled = scaler.transform(user_df)
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
        print(f"FINAL RECOMMENDED PRICE: INR {final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")
    except Exception as e:
        print(f"\n[Error] System execution issue: {e}")

print("Saving model artifacts...")
# Fit standalone KNN explicitly before saving to avoid downstream state errors
knn_base.fit(X_train_scaled, y_train_encoded)
joblib.dump(knn_base, "knn_model.pkl")

# Save hybrid ensemble and supporting items
joblib.dump(ensemble_model, "knn_linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("All artifacts successfully saved to .pkl files!\n")

# Run interactive user testing loop
if __name__ == "__main__":
    predict_user_car()
