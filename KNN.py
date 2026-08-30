import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
import joblib


from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, y_train_price, y_test_price, scaler, brand_means


knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')

print("Training K-Nearest Neighbors (KNN) Classifier & Regressor...")
knn_classifier.fit(X_train_scaled, y_train)
knn_regressor.fit(X_train_scaled, y_train_price)
print("Training Complete!\n")

joblib.dump({"classifier": knn_classifier, "regressor": knn_regressor}, "knn_model.pkl")
print("Saved trained KNN models to 'knn_model.pkl'.\n")


y_pred_tier = knn_classifier.predict(X_test_scaled)
y_pred_price = knn_regressor.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_tier)
precision = precision_score(y_test, y_pred_tier, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_tier, average='weighted')
f1 = f1_score(y_test, y_pred_tier, average='weighted')
mae = mean_absolute_error(y_test_price, y_pred_price)
r2 = r2_score(y_test_price, y_pred_price)

print("==========================================================")
print("             KNN PERFORMANCE REPORT                       ")
print("==========================================================")
print("--- Classification (Price Tier) ---")
print(f"Accuracy       : {accuracy * 100:.2f}%")
print(f"Precision      : {precision * 100:.2f}%")
print(f"Recall         : {recall * 100:.2f}%")
print(f"F1-Score       : {f1 * 100:.2f}%")
print("\n--- Regression (Selling Price) ---")
print(f"Price MAE      : ${mae:,.2f}")
print(f"Price R2-Score : {r2:.4f}")
print("==========================================================\n")


CONDITION_MULTIPLIERS = {
    1: 0.80,
    2: 0.90,
    3: 1.00,
    4: 1.10,
    5: 1.20
}

def predict_user_car():
    print("==========================================================")
    print("            KNN CAR PRICE TIER PREDICTION                 ")
    print("==========================================================")

    try:
   
        year = float(input("Enter Year (e.g., 2017): "))
        km_driven = float(input("Enter Kilometers Driven (e.g., 45000): "))
        mileage = float(input("Enter Mileage in km/l (e.g., 21.5): "))
        engine = float(input("Enter Engine CC (e.g., 1248): "))
        max_power = float(input("Enter Max Power in bhp (e.g., 85.0): "))
        seats = float(input("Enter Number of Seats (e.g., 5): "))

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
            print(f"  -> Brand '{brand_input}' not found. Using average (Brand_Encoded: {brand_encoded:.3f})")


        print("\nSelect Fuel Type:\n  [1] Diesel\n  [2] Petrol\n  [3] LPG\n  [4] CNG / Other")
        fuel_choice = input("Enter choice (1-4): ").strip()


        print("\nSelect Transmission Type:\n  [1] Manual\n  [2] Automatic")
        trans_choice = input("Enter choice (1-2): ").strip()


        print("\nSelect Seller Type:\n  [1] Individual\n  [2] Dealer\n  [3] Trustmark Dealer")
        seller_choice = input("Enter choice (1-3): ").strip()

        print("\nSelect Physical Condition Rating:")
        print("  1 Star  : Poor (-20%)")
        print("  2 Stars : Below Average (-10%)")
        print("  3 Stars : Good / Fair (Standard Market)")
        print("  4 Stars : Very Good (+10%)")
        print("  5 Stars : Excellent / Like New (+20%)")
        condition_stars = int(input("Enter Rating (1-5): "))

        fuel_Diesel = 1 if fuel_choice == "1" else 0
        fuel_Petrol = 1 if fuel_choice == "2" else 0
        fuel_LPG    = 1 if fuel_choice == "3" else 0

        transmission_Manual = 1 if trans_choice == "1" else 0

        seller_Individual = 1 if seller_choice == "1" else 0
        seller_Trustmark  = 1 if seller_choice == "3" else 0

       
        input_features = np.array([[
            year, km_driven, mileage, engine, max_power, seats,
            fuel_Diesel, fuel_LPG, fuel_Petrol,
            transmission_Manual, seller_Individual, seller_Trustmark,
            brand_encoded
        ]])

        input_scaled = scaler.transform(input_features)

        knn_proba = knn_classifier.predict_proba(input_scaled)[0]
        class_labels = knn_classifier.classes_
        predicted_tier = class_labels[np.argmax(knn_proba)]

      
        base_price = float(knn_regressor.predict(input_scaled)[0])

      
        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier

      
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"KNN Predicted Tier : {predicted_tier}")
        print("\nConfidence Breakdown:")
        for label, prob in zip(class_labels, knn_proba):
            print(f"  - {label:<6} Tier : {prob * 100:.2f}%")
        print(f"\nML Predicted Base Value    : ${base_price:,.2f}")
        print(f"Condition Multiplier       : {multiplier:.2f}x ({condition_stars} Stars)")
        print(f"FINAL RECOMMENDED PRICE    : ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")

if __name__ == "__main__":
    predict_user_car()