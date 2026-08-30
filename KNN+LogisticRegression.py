import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
import joblib


from loaddata import X_train_scaled, X_test_scaled, y_train, y_test, y_train_price, y_test_price, scaler, cars, brand_means


knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
logreg_clf = LogisticRegression(max_iter=1000, random_state=42)

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
lin_reg = LinearRegression()

print("Training KNN & Logistic Regression Classifiers...")
knn_clf.fit(X_train_scaled, y_train)
logreg_clf.fit(X_train_scaled, y_train)

print("Training KNN & Linear Regression Regressors...")
knn_reg.fit(X_train_scaled, y_train_price)
lin_reg.fit(X_train_scaled, y_train_price)
print("Training Complete!\n")

joblib.dump({
    "knn_clf": knn_clf, "logreg_clf": logreg_clf,
    "knn_reg": knn_reg, "lin_reg": lin_reg
}, "knn_linear_model.pkl")
print("Saved trained models to 'knn_linear_model.pkl'.\n")


def evaluate_clf(model, X, y_true, name):
    y_pred = model.predict(X)
    return {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
    }

knn_results = evaluate_clf(knn_clf, X_test_scaled, y_test, "KNN Classifier")
logreg_results = evaluate_clf(logreg_clf, X_test_scaled, y_test, "Logistic Reg")


knn_classes = list(knn_clf.classes_)
knn_proba = knn_clf.predict_proba(X_test_scaled)
logreg_proba = logreg_clf.predict_proba(X_test_scaled)
ensemble_proba = (knn_proba + logreg_proba) / 2
ensemble_pred = np.array(knn_classes)[np.argmax(ensemble_proba, axis=1)]

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_precision = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
ensemble_recall = recall_score(y_test, ensemble_pred, average='weighted')
ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')


knn_price_pred = knn_reg.predict(X_test_scaled)
lin_price_pred = lin_reg.predict(X_test_scaled)
ensemble_price_pred = (knn_price_pred + lin_price_pred) / 2.0

print("==========================================================================")
print("                  MODEL COMPARISON REPORT                                 ")
print("==========================================================================")
print(f"{'Metric':<12}{'KNN Classifier':<20}{'Logistic Reg':<20}{'Ensemble (avg)':<20}")
print("--------------------------------------------------------------------------")
print(f"{'Accuracy':<12}{knn_results['accuracy']*100:<20.2f}{logreg_results['accuracy']*100:<20.2f}{ensemble_accuracy*100:<20.2f}")
print(f"{'Precision':<12}{knn_results['precision']*100:<20.2f}{logreg_results['precision']*100:<20.2f}{ensemble_precision*100:<20.2f}")
print(f"{'Recall':<12}{knn_results['recall']*100:<20.2f}{logreg_results['recall']*100:<20.2f}{ensemble_recall*100:<20.2f}")
print(f"{'F1-Score':<12}{knn_results['f1']*100:<20.2f}{logreg_results['f1']*100:<20.2f}{ensemble_f1*100:<20.2f}")
print("==========================================================================")
print(f"Price Regression Ensemble MAE : ${mean_absolute_error(y_test_price, ensemble_price_pred):,.2f}")
print(f"Price Regression Ensemble R2  : {r2_score(y_test_price, ensemble_price_pred):.4f}")
print("==========================================================================\n")


CONDITION_MULTIPLIERS = {
    1: 0.80, 2: 0.90, 3: 1.00, 4: 1.10, 5: 1.20
}

def predict_user_car():
    print("\n==========================================================")
    print("     KNN + LOGISTIC REGRESSION TIER PREDICTION            ")
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


        knn_proba_user = knn_clf.predict_proba(input_scaled)[0]
        logreg_proba_user = logreg_clf.predict_proba(input_scaled)[0]
        ensemble_proba_user = (knn_proba_user + logreg_proba_user) / 2

        class_labels = knn_clf.classes_
        knn_tier = class_labels[np.argmax(knn_proba_user)]
        logreg_tier = class_labels[np.argmax(logreg_proba_user)]
        ensemble_tier = class_labels[np.argmax(ensemble_proba_user)]

        knn_user_price = float(knn_reg.predict(input_scaled)[0])
        lin_user_price = float(lin_reg.predict(input_scaled)[0])
        base_price = (knn_user_price + lin_user_price) / 2.0


        multiplier = CONDITION_MULTIPLIERS.get(condition_stars, 1.0)
        final_recommended_price = base_price * multiplier


        print("\n----------------------------------------------------------")
        print("                 PREDICTION RESULTS                       ")
        print("----------------------------------------------------------")
        print(f"KNN Classifier predicted tier       : {knn_tier}")
        print(f"Logistic Regression predicted tier  : {logreg_tier}")
        print(f"Ensemble (soft-voted) predicted tier: {ensemble_tier}")
        print("\nEnsemble Confidence Breakdown:")
        for label, prob in zip(class_labels, ensemble_proba_user):
            print(f"  - {label:<6} Tier : {prob * 100:.2f}%")
        print(f"\nBase Market Value (by ensemble reg)  : ${base_price:,.2f}")
        print(f"Condition Rating                     : {condition_stars} Star(s) (Multiplier: {multiplier:.2f}x)")
        print(f"FINAL RECOMMENDED PRICE              : ${final_recommended_price:,.2f}")
        print("==========================================================\n")

    except ValueError:
        print("\n[Error] Invalid input! Please enter numbers for specs and menu selections.")

if __name__ == "__main__":
    predict_user_car()