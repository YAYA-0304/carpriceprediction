import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
import streamlit as st

# 1. IMPORT DATASET & GRAPH UTILITIES
from loaddata import X_test_scaled, y_test
from plot_graph import plot_side_by_side_confusion_matrix

# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="Car Price & Market Tier Prediction System",
    page_icon="🚗",
    layout="wide",
)

# -----------------------------------------------------------------------------
# 1. LOAD TRAINED MODEL ARTIFACTS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
  scaler_obj = joblib.load("scaler.pkl")
  svm = joblib.load("svm_model.pkl")
  svm_xgb_data = joblib.load("svm_xgb_model.pkl")
  return svm, svm_xgb_data, scaler_obj


try:
  svm_model, svm_xgb_dict, scaler = load_models()
  svm_ensemble = svm_xgb_dict["svm"]
  xgb_ensemble = svm_xgb_dict["xgb"]
  le = svm_xgb_dict["le"]
except Exception as e:
  st.error(
      "⚠️ Model artifacts (`scaler.pkl`, `svm_model.pkl`, `svm_xgb_model.pkl`)"
      " not found!\nPlease run your training scripts to generate the `.pkl`"
      " files."
  )
  st.stop()

# -----------------------------------------------------------------------------
# 2. PRICING CONSTANTS
# -----------------------------------------------------------------------------
TIER_BASE_PRICES = {"Low": 250000.0, "Medium": 550000.0, "High": 1000000.0}

CONDITION_MULTIPLIERS = {
    "1 Star (Poor / -20%)": (1, 0.80),
    "2 Stars (Below Average / -10%)": (2, 0.90),
    "3 Stars (Good / Fair / Standard Market)": (3, 1.00),
    "4 Stars (Very Good / +10%)": (4, 1.10),
    "5 Stars (Excellent / Like New / +20%)": (5, 1.20),
}

# -----------------------------------------------------------------------------
# 3. HEADER & SIDEBAR ARCHITECTURE SELECTOR
# -----------------------------------------------------------------------------
st.title("🚗 Car Price Tier Prediction & Valuation System")
st.markdown(
    "Automated market tier classification and comparative evaluation across"
    " Standalone and Hybrid Machine Learning Architectures."
)
st.markdown("---")

st.sidebar.header("⚙️ AI Architecture")
selected_architecture = st.sidebar.selectbox(
    "AI MODEL :",
    ["Hybrid Ensemble (SVM + XGBoost)", "Support Vector Machine (SVM OvO)"],
)

tab1, tab2 = st.tabs(
    ["🚀 Interactive Prediction", "📈 Model Evaluation & Comparison Heatmap"]
)

# =============================================================================
# TAB 1: INTERACTIVE PREDICTION
# =============================================================================
with tab1:
  col1, col2 = st.columns([1.2, 1], gap="large")

  with col1:
    st.subheader("1. Enter Car Details")

    c1, c2 = st.columns(2)
    with c1:
      year = st.number_input(
          "Manufacture Year",
          min_value=1990,
          max_value=2026,
          value=2018,
          step=1,
      )
      km_driven = st.number_input(
          "Kilometers Driven (km)",
          min_value=0,
          max_value=500000,
          value=45000,
          step=1000,
      )
      mileage = st.number_input(
          "Mileage (km/l)", min_value=5.0, max_value=40.0, value=21.5, step=0.1
      )

    with c2:
      engine = st.number_input(
          "Engine Capacity (CC)",
          min_value=600,
          max_value=6000,
          value=1248,
          step=50,
      )
      max_power = st.number_input(
          "Max Power (bhp)",
          min_value=30.0,
          max_value=600.0,
          value=85.0,
          step=1.0,
      )
      seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8], index=2)

    st.markdown("---")
    st.subheader("2. Market & Ownership Attributes")

    c3, c4, c5 = st.columns(3)
    with c3:
      fuel_type = st.selectbox(
          "Fuel Type", ["Diesel", "Petrol", "LPG", "CNG / Other"]
      )
    with c4:
      transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    with c5:
      seller_type = st.selectbox(
          "Seller Type", ["Individual", "Dealer", "Trustmark Dealer"]
      )

    condition_choice = st.select_slider(
        "Vehicle Physical Condition Rating",
        options=list(CONDITION_MULTIPLIERS.keys()),
        value="3 Stars (Good / Fair / Standard Market)",
    )

    predict_btn = st.button(
        "🚀 Predict Market Tier & Value",
        type="primary",
        use_container_width=True,
    )

  with col2:
    st.subheader("📊 Prediction Results")
    st.markdown(f"**Active AI Model:** `{selected_architecture}`")

    if predict_btn:
      fuel_Diesel = 1 if fuel_type == "Diesel" else 0
      fuel_Petrol = 1 if fuel_type == "Petrol" else 0
      fuel_LPG = 1 if fuel_type == "LPG" else 0
      transmission_Manual = 1 if transmission == "Manual" else 0
      seller_Individual = 1 if seller_type == "Individual" else 0
      seller_Trustmark = 1 if seller_type == "Trustmark Dealer" else 0
      brand_encoded = 2.0

      input_features = np.array([[
          year,
          km_driven,
          mileage,
          engine,
          max_power,
          seats,
          fuel_Diesel,
          fuel_LPG,
          fuel_Petrol,
          transmission_Manual,
          seller_Individual,
          seller_Trustmark,
          brand_encoded,
      ]])

      input_scaled = scaler.transform(input_features)

      if selected_architecture == "Support Vector Machine (SVM OvO)":
        probabilities = svm_model.predict_proba(input_scaled)[0]
        class_labels = list(svm_model.classes_)
        predicted_tier = class_labels[np.argmax(probabilities)]
      else:
        p_svm = svm_ensemble.predict_proba(input_scaled)[0]
        p_xgb = xgb_ensemble.predict_proba(input_scaled)[0]
        probabilities = (p_svm + p_xgb) / 2.0
        class_labels = list(le.classes_)
        predicted_tier = class_labels[np.argmax(probabilities)]

      stars, multiplier = CONDITION_MULTIPLIERS[condition_choice]
      base_price = TIER_BASE_PRICES.get(predicted_tier, 550000.0)
      final_price = base_price * multiplier

      st.success(f"### Predicted Tier: **{predicted_tier} Class**")

      m_col1, m_col2 = st.columns(2)
      with m_col1:
        st.metric("Base Tier Valuation", f"${base_price:,.2f}")
      with m_col2:
        st.metric("Condition Multiplier", f"{multiplier:.2f}x ({stars}★)")

      st.metric("Final Recommended Market Price", f"${final_price:,.2f}")

      # Confidence Distribution Bar Chart
      st.markdown("#### Model Confidence Distribution")
      prob_df = pd.DataFrame({
          "Market Tier": class_labels,
          "Confidence (%)": [p * 100 for p in probabilities],
      }).set_index("Market Tier")
      st.bar_chart(prob_df)
    else:
      st.info(
          "Fill in the vehicle specifications on the left and click **Predict"
          " Market Tier & Value** to see results."
      )

# =============================================================================
# TAB 2: MODEL EVALUATION & HEATMAP COMPARISON
# =============================================================================
with tab2:
  st.subheader("📊 Model Evaluation & Comparison Heatmap")
  st.markdown(
      "Compare baseline SVM performance against the Hybrid SVM + XGBoost"
      " enhancement on test data."
  )

  if X_test_scaled is None or y_test is None:
    st.warning("⚠️ Test dataset not detected from `loaddata.py`.")
  else:
    class_names = list(le.classes_)

    # Predictions
    y_pred_svm = svm_model.predict(X_test_scaled)
    if isinstance(y_pred_svm[0], (int, np.integer)):
      pred_base = le.inverse_transform(y_pred_svm)
    else:
      pred_base = y_pred_svm

    p_svm_test = svm_ensemble.predict_proba(X_test_scaled)
    p_xgb_test = xgb_ensemble.predict_proba(X_test_scaled)
    comb_test = (p_svm_test + p_xgb_test) / 2.0
    pred_hybrid = le.inverse_transform(np.argmax(comb_test, axis=1))

    # --- Accuracy & Performance Gain Metrics ---
    acc_base = (pred_base == y_test).mean() * 100
    acc_hybrid = (pred_hybrid == y_test).mean() * 100
    gain = acc_hybrid - acc_base

    m1, m2, m3 = st.columns(3)
    with m1:
      st.metric("Standalone SVM Accuracy", f"{acc_base:.2f}%")
    with m2:
      st.metric("Hybrid (SVM + XGBoost) Accuracy", f"{acc_hybrid:.2f}%")
    with m3:
      st.metric(
          "Ensemble Performance Gain",
          f"+{gain:.2f}%" if gain >= 0 else f"{gain:.2f}%",
      )

    st.write("---")

    # --- Accuracy Breakdown & Increment Heatmap ---
    st.markdown("#### 📈 Tier-by-Tier Accuracy & Increment Heatmap")
    report_svm = classification_report(
        y_test, pred_base, output_dict=True, zero_division=0
    )
    report_hybrid = classification_report(
        y_test, pred_hybrid, output_dict=True, zero_division=0
    )

    svm_per_class = [
        report_svm.get(c, {}).get("precision", 0) * 100 for c in class_names
    ]
    hybrid_per_class = [
        report_hybrid.get(c, {}).get("precision", 0) * 100 for c in class_names
    ]

    rows = class_names + ["Overall Accuracy"]
    svm_vals = svm_per_class + [acc_base]
    hybrid_vals = hybrid_per_class + [acc_hybrid]
    increments = [h - s for h, s in zip(hybrid_vals, svm_vals)]

    df_acc = pd.DataFrame(
        {
            "Standalone SVM (%)": svm_vals,
            "Hybrid (SVM + XGB) (%)": hybrid_vals,
            "Increment (+Δ %)": increments,
        },
        index=rows,
    )

    annot_text = np.array(
        [
            [f"{s:.2f}%", f"{h:.2f}%", f"{'+' if d >= 0 else ''}{d:.2f}%"]
            for s, h, d in zip(svm_vals, hybrid_vals, increments)
        ]
    )

    fig_inc, ax_inc = plt.subplots(figsize=(8, 3.2))
    sns.heatmap(
        df_acc,
        annot=annot_text,
        fmt="",
        cmap="Blues",
        linewidths=1,
        linecolor="white",
        cbar=True,
        ax=ax_inc,
    )
    ax_inc.set_title(
        "Accuracy Comparison & Increment Gain Across Classes", pad=10
    )
    st.pyplot(fig_inc)

    st.write("---")

    # --- Side-by-Side Confusion Matrix ---
    st.markdown("#### 🔄 Side-by-Side Confusion Matrix")
    fig_cm = plot_side_by_side_confusion_matrix(
        y_true=y_test,
        y_pred_base=pred_base,
        y_pred_hybrid=pred_hybrid,
        class_names=class_names,
        base_name="Standalone SVM",
        hybrid_name="Hybrid (SVM + XGBoost)",
    )
    st.pyplot(fig_cm)