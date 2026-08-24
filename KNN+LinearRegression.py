import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# =========================================================================
# 1. IMPORT PREPROCESSED DATA & SCALER
# =========================================================================
# y_train / y_test are Price_Tier labels (Low/Medium/High) - both models
from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, scaler, cars, brand_means

# =========================================================================
# 2. TRAIN BOTH MODELS
# =========================================================================
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
logreg_model = LogisticRegression(max_iter=1000, random_state=42)

print("Training K-Nearest Neighbors (KNN) Classifier...")
knn_model.fit(X_train_scaled, y_train)
print("Training Complete!\n")

print("Training Logistic Regression...")
logreg_model.fit(X_train_scaled, y_train)
print("Training Complete!\n")

joblib.dump({"knn": knn_model, "lr": logreg_model}, "knn_linear_model.pkl")
print("Saved trained models to 'knn_linear_model.pkl'.\n")

# =========================================================================
# 3. EVALUATE PERFORMANCE METRICS
# =========================================================================
def evaluate(model, X, y_true, name):
    y_pred = model.predict(X)
    return {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
    }

knn_results = evaluate(knn_model, X_test_scaled, y_test, "KNN Classifier")
logreg_results = evaluate(logreg_model, X_test_scaled, y_test, "Logistic Regression")

# --- SOFT-VOTING ENSEMBLE: average both models' predicted class probabilities ---
# Class order must match between the two models - both were fit on the same
# y_train, so classes_ should already align, but we align explicitly to be safe.
knn_classes = list(knn_model.classes_)
logreg_classes = list(logreg_model.classes_)
assert knn_classes == logreg_classes, "Class order mismatch between models"

knn_proba = knn_model.predict_proba(X_test_scaled)
logreg_proba = logreg_model.predict_proba(X_test_scaled)
ensemble_proba = (knn_proba + logreg_proba) / 2
ensemble_pred = np.array(knn_classes)[np.argmax(ensemble_proba, axis=1)]

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')

print("==========================================================")
print("                  MODEL COMPARISON REPORT                 ")
print("==========================================================")
print(f"{'Metric':<12}{'KNN Classifier':<20}{'Logistic Reg':<20}{'Ensemble (avg)':<20}")
print("--------------------------------------------------------------------------")
print(f"{'Accuracy':<12}{knn_results['accuracy']*100:<20.2f}{logreg_results['accuracy']*100:<20.2f}{ensemble_accuracy*100:<20.2f}")
print(f"{'Precision':<12}{knn_results['precision']*100:<20.2f}{logreg_results['precision']*100:<20.2f}{ensemble_precision*100:<20.2f}")
print(f"{'Recall':<12}{knn_results['recall']*100:<20.2f}{logreg_results['recall']*100:<20.2f}{ensemble_recall*100:<20.2f}")
print(f"{'F1-Score':<12}{knn_results['f1']*100:<20.2f}{logreg_results['f1']*100:<20.2f}{ensemble_f1*100:<20.2f}")
print("==========================================================")
print("(all values are percentages)")
print("==========================================================\n")

# =========================================================================
# 4. INTERACTIVE PRICE PREDICTION FOR USER INPUT
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
    print("     KNN + LOGISTIC REGRESSION TIER PREDICTION            ")
    print("==========================================================")

    try:
        # --- NUMERIC INPUTS ---
        year = float(input("Enter Year (e.g., 2017): "))
        km_driven = float(input("Enter Kilometers Driven (e.g., 45000): "))
        mileage = float(input("Enter Mileage in km/l (e.g., 21.5): "))
        engine = float(input("Enter Engine CC (e.g., 1248): "))
        max_power = float(input("Enter Max Power in bhp (e.g., 85.0): "))
        seats = float(input("Enter Number of Seats (e.g., 5): "))

        # --- BRAND NAME LOOKUP ---
        brand_input = input("Enter Car Brand (e.g., Maruti, BMW, Hyundai): ").strip()

        matched_brand = None
        for known_brand in brand_means.index:
            if known_brand.lower() == brand_input.lower():
                matched_brand = known_brand
                break

        if matched_brand is not None:
            brand_encoded = brand_means[matched_brand]
            print(f"  -> Matched brand '{matched_brand}' (Brand_Encoded: {brand_encoded:.3f})")
        else:
            brand_encoded = brand_means.mean()
            print(f"  -> Brand '{brand_input}' not found in training data. "
                  f"Using overall average (Brand_Encoded: {brand_encoded:.3f})")

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

        # --- ASSEMBLE FEATURE ARRAY ---
        input_features = np.array([[
            year, km_driven, mileage, engine, max_power, seats,
            fuel_Diesel, fuel_LPG, fuel_Petrol,
            transmission_Manual, seller_Individual, seller_Trustmark,
            brand_encoded
        ]])

        # --- SCALE FEATURES ---
        input_scaled = scaler.transform(input_features)

        # --- PREDICT WITH BOTH MODELS + SOFT-VOTING ENSEMBLE ---
        knn_proba_user = knn_model.predict_proba(input_scaled)[0]
        logreg_proba_user = logreg_model.predict_proba(input_scaled)[0]
        ensemble_proba_user = (knn_proba_user + logreg_proba_user) / 2

        class_labels = knn_model.classes_
        knn_tier = class_labels[np.argmax(knn_proba_user)]
        logreg_tier = class_labels[np.argmax(logreg_proba_user)]
        ensemble_tier = class_labels[np.argmax(ensemble_proba_user)]

        price_col = "selling_price" if "selling_price" in cars.columns else "Price"
        year_col = "year" if "year" in cars.columns else ("Year" if "Year" in cars.columns else None)

        if year_col and year_col in cars.columns:
            similar_cars = cars[
                (cars["Price_Tier"] == ensemble_tier)
                & (cars[year_col].between(year - 2, year + 2))
            ]
        else:
            similar_cars = pd.DataFrame()

        if len(similar_cars) >= 3:
            base_price = float(similar_cars[price_col].median())
        else:
            base_price = float(cars[cars["Price_Tier"] == ensemble_tier][price_col].median())

        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier

        # --- DISPLAY PREDICTION SUMMARY ---
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"KNN Classifier predicted tier       : {knn_tier}")
        print(f"Logistic Regression predicted tier  : {logreg_tier}")
        print(f"Ensemble (soft-voted) predicted tier: {ensemble_tier}")
        print("\nEnsemble Confidence Breakdown:")
        for label, prob in zip(class_labels, ensemble_proba_user):
            print(f"  - {label:<6} Tier : {prob * 100:.2f}%")
        print(f"\nBase Market Value (by ensemble tier) : ${base_price:,.2f}")
        print(f"Condition Rating                     : {condition_stars} Star(s) (Multiplier: {multiplier:.2f}x)")
        print(f"FINAL RECOMMENDED PRICE              : ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")

# Run interactive user testing
if __name__ == "__main__":
    predict_user_car()