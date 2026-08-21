import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. IMPORT PREPROCESSED DATA & SCALER
from loaddata import (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
    cars,
    X
)

# Set random seeds for reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# 2. ENCODE STRING LABELS TO INTEGERS (0, 1, 2)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# 3. BUILD THE ARTIFICIAL NEURAL NETWORK (ANN)
ann_model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(64, activation="relu"),
    Dropout(0.20),

    Dense(32, activation="relu"),

    Dense(len(label_encoder.classes_), activation="softmax")
])

# Compile the ANN model
ann_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. TRAIN THE ANN CLASSIFIER
print("Training Artificial Neural Network (ANN) Classifier...")

ann_model.fit(
    X_train_scaled,
    y_train_encoded,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

print("Training Complete!\n")

# 5. EVALUATE PERFORMANCE METRICS
y_prob = ann_model.predict(X_test_scaled, verbose=0)

# Select the class with the highest probability
y_pred_encoded = np.argmax(y_prob, axis=1)

# Convert integer predictions back to Low / Medium / High
y_pred = label_encoder.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("==========================================================")
print("        ARTIFICIAL NEURAL NETWORK (ANN) REPORT           ")
print("==========================================================")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================================\n")


# =========================================================================
# 6. INTERACTIVE PRICE PREDICTION FOR USER INPUT
# =========================================================================

# Baseline prices assigned to each market tier
TIER_BASE_PRICES = {
    "Low": 250000,
    "Medium": 550000,
    "High": 1000000
}

# Multipliers based on vehicle physical condition
CONDITION_MULTIPLIERS = {
    1: 0.80,  # Poor (-20%)
    2: 0.90,  # Below Average (-10%)
    3: 1.00,  # Good / Fair (Standard Market Value)
    4: 1.10,  # Very Good (+10%)
    5: 1.20   # Excellent / Like New (+20%)
}


def predict_user_car():
    print("\n==========================================================")
    print("        CUSTOM CAR PRICE RECOMMENDATION SYSTEM (ANN)      ")
    print("==========================================================")

    try:
        # --- NUMERIC INPUTS ---
        year = float(input("Enter Year (e.g., 2017): "))
        km_driven = float(
            input("Enter Kilometers Driven (e.g., 45000): ")
        )
        mileage = float(
            input("Enter Mileage in km/l (e.g., 21.5): ")
        )
        engine = float(
            input("Enter Engine CC (e.g., 1248): ")
        )
        max_power = float(
            input("Enter Max Power in bhp (e.g., 85.0): ")
        )
        seats = float(
            input("Enter Number of Seats (e.g., 5): ")
        )

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
        brand_input = input("Brand: ").strip()

        # Create case-insensitive brand mapping
        brand_means = cars.groupby("brand")["Brand_Encoded"].first()

        brand_lookup = {
            str(brand).lower(): value
            for brand, value in brand_means.items()
        }

        if brand_input.lower() in brand_lookup:
            brand_encoded = brand_lookup[brand_input.lower()]
        else:
            brand_encoded = cars["Brand_Encoded"].median()
            print(
                "-> Unrecognized brand. Assigning generic market "
                f"weight: {brand_encoded:.2f}"
            )

        # --- MENU CHOICE: CONDITION RATING ---
        print("\nSelect Physical Condition Rating:")
        print("  1 Star  : Poor (-20%)")
        print("  2 Stars : Below Average (-10%)")
        print("  3 Stars : Good / Fair (Standard Market)")
        print("  4 Stars : Very Good (+10%)")
        print("  5 Stars : Excellent / Like New (+20%)")

        condition_stars = int(
            input("Enter Rating (1-5): ")
        )

        if condition_stars not in CONDITION_MULTIPLIERS:
            print(
                "Invalid rating choice. "
                "Defaulting to 3 Stars (Good)."
            )
            condition_stars = 3

        # =========================================================================
        # 7. MAP USER INPUT TO ONE-HOT ENCODED MATRIX
        # =========================================================================

        # Create an empty row matching the exact training feature structure
        input_data = {
            col: 0.0
            for col in X.columns
        }

        # Assign numeric values
        input_data["year"] = year
        input_data["km_driven"] = km_driven
        input_data["mileage(km/ltr/kg)"] = mileage
        input_data["engine"] = engine
        input_data["max_power"] = max_power
        input_data["seats"] = seats
        input_data["Brand_Encoded"] = brand_encoded

        # --- MAP FUEL TYPE ---
        fuel_mapping = {
            "1": "Diesel",
            "2": "Petrol",
            "3": "LPG",
            "4": "CNG"
        }

        selected_fuel = fuel_mapping.get(fuel_choice)

        if selected_fuel is not None:
            fuel_column = f"fuel_{selected_fuel}"

            # If the column exists, set it to 1.
            # If it does not exist, it is likely the dropped baseline category.
            if fuel_column in input_data:
                input_data[fuel_column] = 1

        # --- MAP TRANSMISSION TYPE ---
        transmission_mapping = {
            "1": "Manual",
            "2": "Automatic"
        }

        selected_transmission = transmission_mapping.get(trans_choice)

        if selected_transmission is not None:
            transmission_column = (
                f"transmission_{selected_transmission}"
            )

            if transmission_column in input_data:
                input_data[transmission_column] = 1

        # --- MAP SELLER TYPE ---
        seller_mapping = {
            "1": "Individual",
            "2": "Dealer",
            "3": "Trustmark Dealer"
        }

        selected_seller = seller_mapping.get(seller_choice)

        if selected_seller is not None:
            seller_column = f"seller_type_{selected_seller}"

            if seller_column in input_data:
                input_data[seller_column] = 1

        # Convert into DataFrame with exact same column order
        user_df = pd.DataFrame(
            [input_data]
        )[X.columns]

        # Scale the user input
        input_scaled = scaler.transform(user_df)

        # =========================================================================
        # 8. RUN ANN PREDICTION AND APPLY PRICE PIPELINE
        # =========================================================================

        probabilities = ann_model.predict(
            input_scaled,
            verbose=0
        )[0]

        predicted_encoded = np.argmax(probabilities)

        predicted_tier = label_encoder.inverse_transform(
            [predicted_encoded]
        )[0]

        base_price = TIER_BASE_PRICES.get(
            predicted_tier,
            500000
        )

        multiplier = CONDITION_MULTIPLIERS.get(
            condition_stars,
            1.0
        )

        final_recommended_price = (
            base_price * multiplier
        )

        # --- DISPLAY PREDICTION SUMMARY ---
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")

        print(
            f"Predicted Market Tier : {predicted_tier}"
        )

        print("Prediction Confidence Breakdown:")

        for class_name, prob in zip(
            label_encoder.classes_,
            probabilities
        ):
            print(
                f"  - {class_name:6s} Tier : "
                f"{prob * 100:.2f}%"
            )

        print(
            f"Condition Rating      : "
            f"{condition_stars} Star(s) "
            f"(Multiplier: {multiplier:.2f}x)"
        )

        print(
            f"FINAL RECOMMENDED PRICE: "
            f"INR {final_recommended_price:,.2f}"
        )

        print(
            "==========================================================\n"
        )

    except ValueError:
        print(
            "\n[Error] Invalid input! Please enter numbers "
            "for specifications and menu selections."
        )

    except Exception as e:
        print(
            f"\n[Error] System execution issue: {e}"
        )


# =========================================================================
# 9. SAVE MODEL ARTIFACTS
# =========================================================================

print("Saving model artifacts...")

# Save ANN model using Keras format
ann_model.save("ann_model.keras")

# Save preprocessors
joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    label_encoder,
    "ann_label_encoder.pkl"
)

print(
    "All ANN artifacts successfully saved!\n"
)


# Run interactive user testing
if __name__ == "__main__":
    predict_user_car()