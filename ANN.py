import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, scaler, cars, brand_means, X_train

# 2. TRAIN THE ANN CLASSIFIER
ann_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

print("Training Artificial Neural Network (ANN) Classifier...")
ann_model.fit(X_train_scaled, y_train)
print("Training Complete!\n")

# 3. EVALUATE PERFORMANCE METRICS
y_pred = ann_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("==========================================================")
print("        ARTIFICIAL NEURAL NETWORK (ANN) REPORT           ")
print("==========================================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================================\n")

# Save trained ANN model and scaler
joblib.dump(ann_model, "ann_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved trained ANN model to 'ann_model.pkl' and scaler to 'scaler.pkl'.\n")


# =========================================================================
# 4. INTERACTIVE PRICE PREDICTION FOR USER INPUT
# =========================================================================

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

        if condition_stars not in CONDITION_MULTIPLIERS:
            print("Invalid rating choice. Defaulting to 3 Stars (Good).")
            condition_stars = 3

        # --- BRAND VALUE ENCODING ---
        print("\nEnter Car Brand Name (e.g., Maruti, Hyundai, BMW):")
        brand_input = input("Brand: ").strip()

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
            print(f"-> Unrecognized brand. Assigning generic market weight: {brand_encoded:.2f}")


        # =========================================================================
        # 5. MAP USER INPUT TO ONE-HOT ENCODED COLUMNS
        # =========================================================================

        # Create a base dictionary matching all features used during training
        input_data = {col: 0.0 for col in X_train.columns}

        # Assign numeric values
        input_data['year'] = year
        input_data['km_driven'] = km_driven
        input_data['mileage(km/ltr/kg)'] = mileage
        input_data['engine'] = engine
        input_data['max_power'] = max_power
        input_data['seats'] = seats
        input_data['Brand_Encoded'] = brand_encoded


        # --- FUEL TYPE ---
        if fuel_choice == '1' and 'fuel_Diesel' in input_data:
            input_data['fuel_Diesel'] = 1

        elif fuel_choice == '2' and 'fuel_Petrol' in input_data:
            input_data['fuel_Petrol'] = 1

        elif fuel_choice == '3' and 'fuel_LPG' in input_data:
            input_data['fuel_LPG'] = 1

        # CNG is baseline if its dummy column was dropped


        # --- TRANSMISSION TYPE ---
        if (
            trans_choice == '1'
            and 'transmission_Manual' in input_data
        ):
            input_data['transmission_Manual'] = 1

        # Automatic is baseline if its dummy column was dropped


        # --- SELLER TYPE ---
        if (
            seller_choice == '1'
            and 'seller_type_Individual' in input_data
        ):
            input_data['seller_type_Individual'] = 1

        elif (
            seller_choice == '3'
            and 'seller_type_Trustmark Dealer' in input_data
        ):
            input_data['seller_type_Trustmark Dealer'] = 1

        # Dealer is baseline if its dummy column was dropped


        # Convert dictionary into DataFrame using exact training column order
        user_df = pd.DataFrame([input_data])[X_train.columns]

        # Scale features
        input_scaled = scaler.transform(user_df)


        # =========================================================================
        # 6. RUN ANN PREDICTION
        # =========================================================================

        # Get prediction probabilities
        probabilities = ann_model.predict_proba(input_scaled)[0]

        # ANN class labels
        class_labels = ann_model.classes_

        # Select class with highest probability
        predicted_tier = class_labels[np.argmax(probabilities)]


        # =========================================================================
        # 7. DYNAMIC BASELINE LOOKUP FROM DATASET
        # =========================================================================

        # Identify price and year columns
        price_col = (
            "selling_price"
            if "selling_price" in cars.columns
            else "Price"
        )

        year_col = (
            "year"
            if "year" in cars.columns
            else ("Year" if "Year" in cars.columns else None)
        )

        # Find cars with same predicted tier and similar year (±2 years)
        if year_col and year_col in cars.columns:

            similar_cars = cars[
                (cars["Price_Tier"] == predicted_tier)
                &
                (cars[year_col].between(year - 2, year + 2))
            ]

        else:
            similar_cars = pd.DataFrame()


        # Use median of similar cars if enough samples exist
        if len(similar_cars) >= 3:

            base_price = float(
                similar_cars[price_col].median()
            )

        # Otherwise use overall median of the predicted tier
        else:

            base_price = float(
                cars[
                    cars["Price_Tier"] == predicted_tier
                ][price_col].median()
            )


        # =========================================================================
        # 8. APPLY CONDITION MULTIPLIER
        # =========================================================================

        multiplier = CONDITION_MULTIPLIERS.get(
            condition_stars,
            1.0
        )

        final_recommended_price = (
            base_price * multiplier
        )


        # =========================================================================
        # 9. DISPLAY PREDICTION RESULTS
        # =========================================================================

        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")

        print(
            f"Predicted Market Tier : "
            f"{predicted_tier}"
        )

        print(
            "Prediction Confidence Breakdown:"
        )

        for label, prob in zip(
            class_labels,
            probabilities
        ):

            print(
                f"  - {label:<6} Tier : "
                f"{prob * 100:.2f}%"
            )

        print(
            f"Condition Rating      : "
            f"{condition_stars} Star(s) "
            f"(Multiplier: {multiplier:.2f}x)"
        )

        print(
            f"Dynamic Base Value    : "
            f"${base_price:,.2f}"
        )

        print(
            f"FINAL RECOMMENDED PRICE: "
            f"${final_recommended_price:,.2f}"
        )

        print(
            "==========================================================\n"
        )


    except ValueError:

        print(
            "\n[Error] Invalid input! Please enter numbers "
            "for specs and menu selections."
        )

    except Exception as e:

        print(
            f"\n[Error] System execution issue: {e}"
        )


# Run interactive user testing
if __name__ == "__main__":
    predict_user_car()