import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, mean_absolute_error, r2_score

# 1. IMPORT DATASET & GRAPH UTILITIES
from loaddata import X_test_scaled, y_test, y_test_price, brand_means
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
    
    # Standalone models (contain both classifier and regressor)
    svm_standalone = joblib.load("svm_model.pkl")
    knn_standalone = joblib.load("knn_model.pkl") if pd.io.common.file_exists("knn_model.pkl") else None
    ann_standalone = joblib.load("ann_models.pkl") if pd.io.common.file_exists("ann_models.pkl") else joblib.load("ann_model.pkl")
    
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
# 3. HEADER & SIDEBAR
# -----------------------------------------------------------------------------
st.title("🚗 Car Price Tier Prediction & Valuation System")
st.markdown("Automated market tier classification and ML continuous price regression across Architectures.")
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

            brand_encoded = float(brand_means.get(brand_input, brand_means.mean()))

            input_features = np.array([[
                year, km_driven, mileage, engine, max_power, seats,
                fuel_Diesel, fuel_LPG, fuel_Petrol,
                transmission_Manual, seller_Individual, seller_Trustmark,
                brand_encoded
            ]])

            input_scaled = scaler.transform(input_features)

            # Prediction via Selected Hybrid Architecture
            if selected_architecture == "Hybrid Ensemble (SVM + XGBoost)":
                svm_clf = models["svm_xgb_data"]["svm_clf"]
                xgb_clf = models["svm_xgb_data"]["xgb_clf"]
                svm_reg = models["svm_xgb_data"]["svm_reg"]
                xgb_reg = models["svm_xgb_data"]["xgb_reg"]
                le = models["svm_xgb_data"]["le"]
                
                # Tier
                p_svm = svm_clf.predict_proba(input_scaled)[0]
                p_xgb = xgb_clf.predict_proba(input_scaled)[0]
                probabilities = (p_svm + p_xgb) / 2.0
                class_labels = list(le.classes_)
                predicted_tier = le.inverse_transform([np.argmax(probabilities)])[0]

                # Exact ML Regression Price
                price_svm = float(svm_reg.predict(input_scaled)[0])
                price_xgb = float(xgb_reg.predict(input_scaled)[0])
                base_price = (price_svm + price_xgb) / 2.0

            elif selected_architecture == "Hybrid Ensemble (ANN + Random Forest)":
                ann_clf = models["ann_rf_data"]["ann_clf"]
                rf_clf = models["ann_rf_data"]["rf_clf"]
                ann_reg = models["ann_rf_data"]["ann_reg"]
                rf_reg = models["ann_rf_data"]["rf_reg"]
                le = models["ann_rf_data"]["le"]
                
                # Tier
                p_ann = ann_clf.predict_proba(input_scaled)[0]
                p_rf = rf_clf.predict_proba(input_scaled)[0]
                probabilities = (p_ann + p_rf) / 2.0
                class_labels = list(le.classes_)
                predicted_tier = le.inverse_transform([np.argmax(probabilities)])[0]

                # Exact ML Regression Price
                price_ann = float(ann_reg.predict(input_scaled)[0])
                price_rf = float(rf_reg.predict(input_scaled)[0])
                base_price = (price_ann + price_rf) / 2.0

            else:  # Hybrid Ensemble (KNN + Logistic Regression)
                knn_clf = models["knn_lr_data"]["knn_clf"]
                logreg_clf = models["knn_lr_data"]["logreg_clf"]
                knn_reg = models["knn_lr_data"]["knn_reg"]
                lin_reg = models["knn_lr_data"]["lin_reg"]
                
                # Tier
                p_knn = knn_clf.predict_proba(input_scaled)[0]
                p_lr = logreg_clf.predict_proba(input_scaled)[0]
                probabilities = (p_knn + p_lr) / 2.0
                class_labels = list(knn_clf.classes_)
                predicted_tier = class_labels[np.argmax(probabilities)]

                # Exact ML Regression Price
                price_knn = float(knn_reg.predict(input_scaled)[0])
                price_lin = float(lin_reg.predict(input_scaled)[0])
                base_price = (price_knn + price_lin) / 2.0

            stars, multiplier = CONDITION_MULTIPLIERS[condition_choice]
            final_price = base_price * multiplier

            st.success(f"### Predicted Tier: **{predicted_tier} Class**")
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("ML Predicted Base Value", f"${base_price:,.2f}")
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
            st.info("Fill in vehicle specifications on the left and click **Predict Market Tier & Value**.")

# =============================================================================
# TAB 2: MODEL EVALUATION & HEATMAP COMPARISON
# =============================================================================
with tab2:
    st.subheader("📊 Model Evaluation & Comparison Heatmap")
    st.markdown("Compare baseline models against Hybrid enhancements on classification accuracy and price regression.")

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

            # Standalone predictions
            svm_base = models["svm_standalone"]
            clf_base = svm_base["classifier"] if isinstance(svm_base, dict) else svm_base
            reg_base = svm_base.get("regressor", None) if isinstance(svm_base, dict) else None

            pred_base = clf_base.predict(X_test_scaled)
            pred_base_price = reg_base.predict(X_test_scaled) if reg_base else None

            # Hybrid predictions
            p_svm = models["svm_xgb_data"]["svm_clf"].predict_proba(X_test_scaled)
            p_xgb = models["svm_xgb_data"]["xgb_clf"].predict_proba(X_test_scaled)
            pred_hybrid = le.inverse_transform(np.argmax((p_svm + p_xgb) / 2.0, axis=1))

            p_reg_svm = models["svm_xgb_data"]["svm_reg"].predict(X_test_scaled)
            p_reg_xgb = models["svm_xgb_data"]["xgb_reg"].predict(X_test_scaled)
            pred_hybrid_price = (p_reg_svm + p_reg_xgb) / 2.0

        elif compare_pair == "KNN vs. Hybrid (KNN + Logistic Regression)":
            base_name = "Standalone KNN"
            hybrid_name = "Hybrid (KNN + Logistic Reg)"
            knn_clf = models["knn_lr_data"]["knn_clf"]
            class_names = list(knn_clf.classes_)

            # Standalone (KNN alone)
            pred_base = knn_clf.predict(X_test_scaled)
            pred_base_price = models["knn_lr_data"]["knn_reg"].predict(X_test_scaled)

            # Hybrid (KNN + Logistic Regression)
            p_knn = knn_clf.predict_proba(X_test_scaled)
            p_lr = models["knn_lr_data"]["logreg_clf"].predict_proba(X_test_scaled)
            pred_hybrid = np.array(class_names)[np.argmax((p_knn + p_lr) / 2.0, axis=1)]

            p_reg_knn = models["knn_lr_data"]["knn_reg"].predict(X_test_scaled)
            p_reg_lin = models["knn_lr_data"]["lin_reg"].predict(X_test_scaled)
            pred_hybrid_price = (p_reg_knn + p_reg_lin) / 2.0

        else:  # ANN vs. Hybrid (ANN + Random Forest)
            base_name = "Standalone ANN"
            hybrid_name = "Hybrid (ANN + Random Forest)"
            le = models["ann_rf_data"]["le"]
            class_names = list(le.classes_)

            # Standalone
            ann_base = models["ann_standalone"]
            clf_base = ann_base["classifier"] if isinstance(ann_base, dict) else ann_base
            reg_base = ann_base.get("regressor", None) if isinstance(ann_base, dict) else None

            pred_base = clf_base.predict(X_test_scaled)
            pred_base_price = reg_base.predict(X_test_scaled) if reg_base else None

            # Hybrid
            p_ann = models["ann_rf_data"]["ann_clf"].predict_proba(X_test_scaled)
            p_rf = models["ann_rf_data"]["rf_clf"].predict_proba(X_test_scaled)
            pred_hybrid = le.inverse_transform(np.argmax((p_ann + p_rf) / 2.0, axis=1))

            p_reg_ann = models["ann_rf_data"]["ann_reg"].predict(X_test_scaled)
            p_reg_rf = models["ann_rf_data"]["rf_reg"].predict(X_test_scaled)
            pred_hybrid_price = (p_reg_ann + p_reg_rf) / 2.0

        # --- Classification Metrics ---
        acc_base = (pred_base == y_test).mean() * 100
        acc_hybrid = (pred_hybrid == y_test).mean() * 100
        gain = acc_hybrid - acc_base

        st.markdown("#### 🎯 Classification Performance (Price Tier)")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(f"{base_name} Accuracy", f"{acc_base:.2f}%")
        with m2:
            st.metric(f"{hybrid_name} Accuracy", f"{acc_hybrid:.2f}%")
        with m3:
            st.metric("Ensemble Accuracy Gain", f"{acc_hybrid:.2f}%", delta=f"{gain:+.2f}%")

        # --- Regression Metrics ---
        if y_test_price is not None:
            st.markdown("#### 💵 Regression Performance (Selling Price Value)")
            r1, r2, r3, r4 = st.columns(4)
            
            mae_hybrid = mean_absolute_error(y_test_price, pred_hybrid_price)
            r2_hybrid = r2_score(y_test_price, pred_hybrid_price)

            if pred_base_price is not None:
                mae_base = mean_absolute_error(y_test_price, pred_base_price)
                r2_base = r2_score(y_test_price, pred_base_price)
                
                # Raw differences: (Hybrid - Base)
                mae_diff = mae_hybrid - mae_base
                r2_diff = r2_hybrid - r2_base
                
                with r1:
                    st.metric(f"{base_name} MAE", f"${mae_base:,.2f}")
                with r2:
                    # Positive change -> Up arrow (Green), Negative change -> Down arrow (Red)
                    mae_delta_str = f"+${mae_diff:,.2f}" if mae_diff >= 0 else f"-${abs(mae_diff):,.2f}"
                    st.metric(f"{hybrid_name} MAE", f"${mae_hybrid:,.2f}", delta=mae_delta_str)
                with r3:
                    st.metric(f"{base_name} R² Score", f"{r2_base:.4f}")
                with r4:
                    st.metric(f"{hybrid_name} R² Score", f"{r2_hybrid:.4f}", delta=f"{r2_diff:+.4f}")
            else:
                with r1:
                    st.metric(f"{hybrid_name} MAE", f"${mae_hybrid:,.2f}")
                with r2:
                    st.metric(f"{hybrid_name} R² Score", f"{r2_hybrid:.4f}")

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