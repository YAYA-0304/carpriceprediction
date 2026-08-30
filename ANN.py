import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
import joblib

from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, y_train_price, y_test_price, scaler, cars, brand_means, X_train


ann_classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)
ann_regressor = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

print("Training ANN Classifier & Regressor...")
ann_classifier.fit(X_train_scaled, y_train)
ann_regressor.fit(X_train_scaled, y_train_price)
print("Training Complete!\n")

y_pred_tier = ann_classifier.predict(X_test_scaled)
y_pred_price = ann_regressor.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred_tier)
precision = precision_score(y_test, y_pred_tier, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred_tier, average='weighted')
f1 = f1_score(y_test, y_pred_tier, average='weighted')

mae = mean_absolute_error(y_test_price, y_pred_price)
r2 = r2_score(y_test_price, y_pred_price)

print("==========================================================")
print("        ARTIFICIAL NEURAL NETWORK (ANN) REPORT           ")
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
joblib.dump({"classifier": ann_classifier, "regressor": ann_regressor}, "ann_models.pkl")
joblib.dump(scaler, "scaler.pkl")


CONDITION_MULTIPLIERS = {
    1: 0.80, 2: 0.90, 3: 1.00, 4: 1.10, 5: 1.20
}

def predict_user_car():
    print("\n==========================================================")
    print("        CUSTOM CAR PRICE RECOMMENDATION SYSTEM            ")
    print("==========================================================")
    try:
        year = float(input("Enter Year (e.g., 2017): "))
        km_driven = float(input("Enter Kilometers Driven (e.g., 45000): "))
        mileage = float(input("Enter Mileage in km/l (e.g., 21.5): "))
        engine = float(input("Enter Engine CC (e.g., 1248): "))
        max_power = float(input("Enter Max Power in bhp (e.g., 85.0): "))
        seats = float(input("Enter Number of Seats (e.g., 5): "))

        print("\nSelect Fuel Type:")
        print("  [1] Diesel\n  [2] Petrol\n  [3] LPG\n  [4] CNG / Other")
        fuel_choice = input("Enter choice (1-4): ").strip()

        print("\nSelect Transmission Type:")
        print("  [1] Manual\n  [2] Automatic")
        trans_choice = input("Enter choice (1-2): ").strip()

        print("\nSelect Seller Type:")
        print("  [1] Individual\n  [2] Dealer\n  [3] Trustmark Dealer")
        seller_choice = input("Enter choice (1-3): ").strip()

        print("\nSelect Physical Condition Rating:")
        print("  1 Star  : Poor (-20%)\n  2 Stars : Below Average (-10%)\n  3 Stars : Good / Fair (Standard Market)\n  4 Stars : Very Good (+10%)\n  5 Stars : Excellent / Like New (+20%)")
        condition_stars = int(input("Enter Rating (1-5): "))
        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)

        print("\nEnter Car Brand Name (e.g., Maruti, Hyundai, BMW):")
        brand_input = input("Brand: ").strip()

        matched_brand = next((b for b in brand_means.index if b.lower() == brand_input.lower()), None)
        brand_encoded = brand_means[matched_brand] if matched_brand else brand_means.mean()

        input_data = {col: 0.0 for col in X_train.columns}
        input_data.update({
            'year': year, 'km_driven': km_driven, 'mileage(km/ltr/kg)': mileage,
            'engine': engine, 'max_power': max_power, 'seats': seats,
            'Brand_Encoded': brand_encoded
        })
        if fuel_choice == '1' and 'fuel_Diesel' in input_data: input_data['fuel_Diesel'] = 1
        elif fuel_choice == '2' and 'fuel_Petrol' in input_data: input_data['fuel_Petrol'] = 1
        elif fuel_choice == '3' and 'fuel_LPG' in input_data: input_data['fuel_LPG'] = 1

        if trans_choice == '1' and 'transmission_Manual' in input_data: input_data['transmission_Manual'] = 1
        if seller_choice == '1' and 'seller_type_Individual' in input_data: input_data['seller_type_Individual'] = 1
        elif seller_choice == '3' and 'seller_type_Trustmark Dealer' in input_data: input_data['seller_type_Trustmark Dealer'] = 1

        user_df = pd.DataFrame([input_data])[X_train.columns]
        input_scaled = scaler.transform(user_df)

       
        probabilities = ann_classifier.predict_proba(input_scaled)[0]
        class_labels = ann_classifier.classes_
        predicted_tier = class_labels[np.argmax(probabilities)]
        
      
        raw_pred_price = float(ann_regressor.predict(input_scaled)[0])
        final_recommended_price = raw_pred_price * multiplier

       
        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"Predicted Market Tier : {predicted_tier}")
        print("Prediction Confidence Breakdown:")
        for label, prob in zip(class_labels, probabilities):
            print(f"  - {label:<6} Tier : {prob * 100:.2f}%")
        print(f"Condition Rating      : {condition_stars} Star(s) (Multiplier: {multiplier:.2f}x)")
        print(f"Dynamic Base Value    : ${raw_pred_price:,.2f}")
        print(f"FINAL RECOMMENDED PRICE: ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")
    except Exception as e:
        print(f"\n[Error] System execution issue: {e}")

if __name__ == "__main__":
    predict_user_car()