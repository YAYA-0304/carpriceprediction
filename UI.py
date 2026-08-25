import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# 1. IMPORT DATASET & GRAPH UTILITIES
from loaddata import X_test_scaled, y_test, cars, brand_means
from plot_graph import plot_side_by_side_confusion_matrix

# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="Car Price & Market Tier Prediction System",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 1. LOAD ALL TRAINED MODEL ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    scaler_obj = joblib.load("scaler.pkl")
    
    # Standalone models
    svm_standalone = joblib.load("svm_model.pkl")
    knn_standalone = joblib.load("knn_model.pkl")
    ann_standalone = joblib.load("ann_model.pkl")
    
    # Hybrid models
    svm_xgb_data = joblib.load("svm_xgb_model.pkl")
    ann_rf_data = joblib.load("ann_rf_model.pkl")
    knn_lr_data = joblib.load("knn_linear_model.pkl")
    
    return {
        "scaler": scaler_obj,
        "svm_standalone": svm_standalone,
        "knn_standalone": knn_standalone,
        "ann_standalone": ann_standalone,
        "svm_xgb_data": svm_xgb_data,
        "ann_rf_data": ann_rf_data,
        "knn_lr_data": knn_lr_data
    }

try:
    models = load_all_models()
    scaler = models["scaler"]
except Exception as e:
    st.error(
        f"⚠️ Model artifacts missing: {e}\n"
        "Please run all standalone and hybrid scripts to generate all .pkl files."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 2. PRICING CONSTANTS
# -----------------------------------------------------------------------------
CONDITION_MULTIPLIERS = {
    "1 Star (Poor / -20%)": (1, 0.80),
    "2 Stars (Below Average / -10%)": (2, 0.90),
    "3 Stars (Good / Fair / Standard Market)": (3, 1.00),
    "4 Stars (Very Good / +10%)": (4, 1.10),
    "5 Stars (Excellent / Like New / +20%)": (5, 1.20)
}

# -----------------------------------------------------------------------------
# 3. HEADER & SIDEBAR (HYBRID SELECTION ONLY)
# -----------------------------------------------------------------------------
st.title("🚗 Car Price Tier Prediction & Valuation System")
st.markdown("Automated market tier classification and comparative evaluation across Machine Learning Architectures.")
st.markdown("---")

st.sidebar.header("⚙️ Active Hybrid Model")
selected_architecture = st.sidebar.selectbox(
    "Choose Hybrid Model for Prediction:",
    [
        "Hybrid Ensemble (SVM + XGBoost)",
        "Hybrid Ensemble (KNN + Logistic Regression)",
        "Hybrid Ensemble (ANN + Random Forest)"
    ]
)

tab1, tab2 = st.tabs(["🚀 Interactive Prediction", "📈 Model Evaluation & Comparison Heatmap"])

# =============================================================================
# TAB 1: INTERACTIVE PREDICTION
# =============================================================================
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.subheader("1. Enter Car Details")
        
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Manufacture Year", min_value=1990, max_value=2026, value=2018, step=1)
            km_driven = st.number_input("Kilometers Driven (km)", min_value=0, max_value=500000, value=45000, step=1000)
            mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=21.5, step=0.1)
        
        with c2:
            engine = st.number_input("Engine Capacity (CC)", min_value=600, max_value=6000, value=1248, step=50)
            max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=600.0, value=85.0, step=1.0)
            seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8], index=2)

        st.markdown("---")
        st.subheader("2. Market & Ownership Attributes")
        
        available_brands = sorted(brand_means.index.tolist())
        
        c3, c4 = st.columns(2)
        with c3:
            brand_input = st.selectbox("Car Brand", available_brands)
            fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "LPG", "CNG / Other"])
        with c4:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])

        condition_choice = st.select_slider(
            "Vehicle Physical Condition Rating",
            options=list(CONDITION_MULTIPLIERS.keys()),
            value="3 Stars (Good / Fair / Standard Market)"
        )

        predict_btn = st.button("🚀 Predict Market Tier & Value", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Results")
        st.markdown(f"**Active Hybrid Model:** `{selected_architecture}`")
        
        if predict_btn:
            # One-Hot Binary Flags
            fuel_Diesel = 1 if fuel_type == "Diesel" else 0
            fuel_Petrol = 1 if fuel_type == "Petrol" else 0
            fuel_LPG = 1 if fuel_type == "LPG" else 0
            transmission_Manual = 1 if transmission == "Manual" else 0
            seller_Individual = 1 if seller_type == "Individual" else 0
            seller_Trustmark = 1 if seller_type == "Trustmark Dealer" else 0

            # Encode brand
            brand_encoded = float(brand_means.get(brand_input, brand_means.mean()))

            input_features = np.array([[
                year, km_driven, mileage, engine, max_power, seats,
                fuel_Diesel, fuel_LPG, fuel_Petrol,
                transmission_Manual, seller_Individual, seller_Trustmark,
                brand_encoded
            ]])

            input_scaled = scaler.transform(input_features)

            # Prediction based on selected Hybrid architecture
            if selected_architecture == "Hybrid Ensemble (SVM + XGBoost)":
                svm_ens = models["svm_xgb_data"]["svm"]
                xgb_ens = models["svm_xgb_data"]["xgb"]
                le = models["svm_xgb_data"]["le"]
                
                p_svm = svm_ens.predict_proba(input_scaled)[0]
                p_xgb = xgb_ens.predict_proba(input_scaled)[0]
                probabilities = (p_svm + p_xgb) / 2.0
                class_labels = list(le.classes_)
                predicted_tier = le.inverse_transform([np.argmax(probabilities)])[0]

            elif selected_architecture == "Hybrid Ensemble (ANN + Random Forest)":
                ann_ens = models["ann_rf_data"]["ann"]
                rf_ens = models["ann_rf_data"]["rf"]
                le = models["ann_rf_data"]["le"]
                
                p_ann = ann_ens.predict_proba(input_scaled)[0]
                p_rf = rf_ens.predict_proba(input_scaled)[0]
                probabilities = (p_ann + p_rf) / 2.0
                class_labels = list(le.classes_)
                predicted_tier = le.inverse_transform([np.argmax(probabilities)])[0]

            else:  # Hybrid Ensemble (KNN + Logistic Regression)
                knn_ens = models["knn_lr_data"]["knn"]
                lr_ens = models["knn_lr_data"]["lr"]
                
                p_knn = knn_ens.predict_proba(input_scaled)[0]
                p_lr = lr_ens.predict_proba(input_scaled)[0]
                probabilities = (p_knn + p_lr) / 2.0
                class_labels = list(knn_ens.classes_)
                predicted_tier = class_labels[np.argmax(probabilities)]

            # Dynamic Base Price Lookup from Dataset
            price_col = "selling_price" if "selling_price" in cars.columns else "Price"
            year_col = "year" if "year" in cars.columns else ("Year" if "Year" in cars.columns else None)

            if year_col and year_col in cars.columns:
                similar_cars = cars[
                    (cars["Price_Tier"] == predicted_tier) &
                    (cars[year_col].between(year - 2, year + 2))
                ]
            else:
                similar_cars = pd.DataFrame()

            if len(similar_cars) >= 3:
                base_price = float(similar_cars[price_col].median())
            else:
                base_price = float(cars[cars["Price_Tier"] == predicted_tier][price_col].median())

            stars, multiplier = CONDITION_MULTIPLIERS[condition_choice]
            final_price = base_price * multiplier

            st.success(f"### Predicted Tier: **{predicted_tier} Class**")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Dynamic Base Valuation", f"${base_price:,.2f}")
            with m_col2:
                st.metric("Condition Multiplier", f"{multiplier:.2f}x ({stars}★)")
            
            st.metric("Final Recommended Market Price", f"${final_price:,.2f}")

            st.markdown("#### Model Confidence Distribution")
            prob_df = pd.DataFrame({
                "Market Tier": class_labels,
                "Confidence (%)": [p * 100 for p in probabilities]
            }).set_index("Market Tier")
            st.bar_chart(prob_df)
        else:
            st.info("Fill in the vehicle specifications on the left and click **Predict Market Tier & Value** to see results.")

# =============================================================================
# TAB 2: MODEL EVALUATION & HEATMAP COMPARISON
# =============================================================================
with tab2:
    st.subheader("📊 Model Evaluation & Comparison Heatmap")
    st.markdown("Compare baseline models against their respective Hybrid enhancements on test data.")

    compare_pair = st.selectbox(
        "Select Model Architecture Pair to Compare:",
        [
            "SVM vs. Hybrid (SVM + XGBoost)",
            "KNN vs. Hybrid (KNN + Logistic Regression)",
            "ANN vs. Hybrid (ANN + Random Forest)"
        ]
    )

    if X_test_scaled is None or y_test is None:
        st.warning("⚠️ Test dataset not detected from `loaddata.py`.")
    else:
        if compare_pair == "SVM vs. Hybrid (SVM + XGBoost)":
            base_name = "Standalone SVM"
            hybrid_name = "Hybrid (SVM + XGBoost)"
            le = models["svm_xgb_data"]["le"]
            class_names = list(le.classes_)

            # Standalone
            pred_base = models["svm_standalone"].predict(X_test_scaled)
            
            # Hybrid
            p_svm = models["svm_xgb_data"]["svm"].predict_proba(X_test_scaled)
            p_xgb = models["svm_xgb_data"]["xgb"].predict_proba(X_test_scaled)
            pred_hybrid = le.inverse_transform(np.argmax((p_svm + p_xgb) / 2.0, axis=1))

        elif compare_pair == "KNN vs. Hybrid (KNN + Logistic Regression)":
            base_name = "Standalone KNN"
            hybrid_name = "Hybrid (KNN + Logistic Reg)"
            class_names = list(models["knn_standalone"].classes_)

            # Standalone
            pred_base = models["knn_standalone"].predict(X_test_scaled)

            # Hybrid
            p_knn = models["knn_lr_data"]["knn"].predict_proba(X_test_scaled)
            p_lr = models["knn_lr_data"]["lr"].predict_proba(X_test_scaled)
            comb_proba = (p_knn + p_lr) / 2.0
            pred_hybrid = np.array(class_names)[np.argmax(comb_proba, axis=1)]

        else:  # ANN vs. Hybrid (ANN + Random Forest)
            base_name = "Standalone ANN"
            hybrid_name = "Hybrid (ANN + Random Forest)"
            le = models["ann_rf_data"]["le"]
            class_names = list(le.classes_)

            # Standalone
            pred_base = models["ann_standalone"].predict(X_test_scaled)

            # Hybrid
            p_ann = models["ann_rf_data"]["ann"].predict_proba(X_test_scaled)
            p_rf = models["ann_rf_data"]["rf"].predict_proba(X_test_scaled)
            pred_hybrid = le.inverse_transform(np.argmax((p_ann + p_rf) / 2.0, axis=1))

        # --- Accuracy & Metrics ---
        acc_base = (pred_base == y_test).mean() * 100
        acc_hybrid = (pred_hybrid == y_test).mean() * 100
        gain = acc_hybrid - acc_base

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(f"{base_name} Accuracy", f"{acc_base:.2f}%")
        with m2:
            st.metric(f"{hybrid_name} Accuracy", f"{acc_hybrid:.2f}%")
        with m3:
            st.metric("Ensemble Performance Gain", f"{'+' if gain >= 0 else ''}{gain:.2f}%")

        st.write("---")

        # --- Accuracy Breakdown Heatmap ---
        st.markdown(f"#### 📈 Tier-by-Tier Comparison: {base_name} vs. {hybrid_name}")
        report_base = classification_report(y_test, pred_base, output_dict=True, zero_division=0)
        report_hybrid = classification_report(y_test, pred_hybrid, output_dict=True, zero_division=0)

        base_per_class = [report_base.get(c, {}).get("precision", 0) * 100 for c in class_names]
        hybrid_per_class = [report_hybrid.get(c, {}).get("precision", 0) * 100 for c in class_names]

        rows = class_names + ["Overall Accuracy"]
        base_vals = base_per_class + [acc_base]
        hybrid_vals = hybrid_per_class + [acc_hybrid]
        increments = [h - b for h, b in zip(hybrid_vals, base_vals)]

        df_acc = pd.DataFrame({
            f"{base_name} (%)": base_vals,
            f"{hybrid_name} (%)": hybrid_vals,
            "Increment (+Δ %)": increments
        }, index=rows)

        annot_text = np.array([
            [f"{b:.2f}%", f"{h:.2f}%", f"{'+' if d >= 0 else ''}{d:.2f}%"]
            for b, h, d in zip(base_vals, hybrid_vals, increments)
        ])

        fig_inc, ax_inc = plt.subplots(figsize=(8, 3.2))
        sns.heatmap(
            df_acc, annot=annot_text, fmt="", cmap="Blues",
            linewidths=1, linecolor="white", cbar=True, ax=ax_inc
        )
        ax_inc.set_title(f"Accuracy Comparison & Increment Gain ({compare_pair})", pad=10)
        st.pyplot(fig_inc)

        st.write("---")

        # --- Side-by-Side Confusion Matrix ---
        st.markdown("#### 🔄 Side-by-Side Confusion Matrix")
        fig_cm = plot_side_by_side_confusion_matrix(
            y_true=y_test,
            y_pred_base=pred_base,
            y_pred_hybrid=pred_hybrid,
            class_names=class_names,
            base_name=base_name,
            hybrid_name=hybrid_name
        )
        st.pyplot(fig_cm)