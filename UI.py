import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Import plotting functions from plot_graph.py
from plot_graph import plot_confusion_matrix, plot_side_by_side_confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="Car Price & Market Tier Prediction System",
    page_icon="🚗",
    layout="wide",
)


# -----------------------------------------------------------------------------
# 1. LOAD TRAINED MODEL ARTIFACTS & TEST DATA
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_artifacts():
  # Model 1: KNN + Linear Regression
  knn_model, knn_lr_model = None, None
  try:
    knn_model = joblib.load("knn_model.pkl")
    knn_lr_model = joblib.load("knn_lr_model.pkl")
  except Exception:
    pass

  # Model 2: SVM + XGBoost
  svm_model, svm_xgb_model = None, None
  try:
    svm_model = joblib.load("svm_model.pkl")
    svm_xgb_model = joblib.load("svm_xgb_model.pkl")
  except Exception:
    pass

  # Model 3: ANN + Random Forest
  ann_model, ann_rf_model = None, None
  try:
    ann_model = joblib.load("ann_model.pkl")
    ann_rf_model = joblib.load("ann_rf_model.pkl")
  except Exception:
    pass

  # Preprocessors
  scaler, label_encoder = None, None
  try:
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
  except Exception:
    pass

  return {
      "KNN + Linear Regression": (knn_model, knn_lr_model),
      "SVM + XGBoost": (svm_model, svm_xgb_model),
      "ANN + Random Forest": (ann_model, ann_rf_model),
      "scaler": scaler,
      "label_encoder": label_encoder,
  }


artifacts = load_all_artifacts()
scaler = artifacts["scaler"]
label_encoder = artifacts["label_encoder"]


# Load test dataset for evaluation
@st.cache_data
def load_test_evaluation_data():
  try:
    from loaddata import X_test_scaled, y_test

    return X_test_scaled, y_test
  except Exception:
    return None, None


X_test_scaled, y_test = load_test_evaluation_data()

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
# 3. UI LAYOUT & TABS
# -----------------------------------------------------------------------------
st.title("🚗 Car Price Tier Prediction & Valuation System")
st.markdown(
    "Automated market tier classification and comparative evaluation across"
    " Hybrid Machine Learning Architectures."
)
st.markdown("---")

if scaler is None or label_encoder is None:
  st.error(
      "⚠️ Core artifacts (`scaler.pkl`, `label_encoder.pkl`) not found!\n\n"
      "Please run your training script to save the `.pkl` files using `joblib`"
      " first."
  )
  st.stop()

# Sidebar: Exactly 3 AI Model Selections
st.sidebar.header("⚙️ AI Architecture")
selected_architecture = st.sidebar.selectbox(
    "AI MODEL :",
    ["SVM + XGBoost", "KNN + Linear Regression", "ANN + Random Forest"],
)

base_model, hybrid_model = artifacts[selected_architecture]

# Tab Navigation
tab1, tab2 = st.tabs(
    ["🚀 Interactive Prediction", "📈 Model Evaluation & Comparison Heatmap"]
)

# =============================================================================
# TAB 1: INTERACTIVE PREDICTION
# =============================================================================
with tab1:
  col1, col2 = st.columns([1.2, 1])

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

    active_predictor = hybrid_model if hybrid_model is not None else base_model

    if predict_btn:
      if active_predictor is None:
        st.warning(
            f"⚠️ Artifacts for `{selected_architecture}` are not loaded. Please"
            " train and save the `.pkl` files."
        )
      else:
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

        try:
          raw_prediction = active_predictor.predict(input_scaled)[0]
          if isinstance(raw_prediction, (int, np.integer)):
            predicted_tier = label_encoder.inverse_transform([raw_prediction])[
                0
            ]
          else:
            predicted_tier = raw_prediction
        except Exception as e:
          st.error(f"Prediction Error: {e}")
          st.stop()

        try:
          probabilities = active_predictor.predict_proba(input_scaled)[0]
          has_proba = True
        except AttributeError:
          has_proba = False

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

        if has_proba:
          st.markdown("#### Model Confidence Distribution")
          prob_df = pd.DataFrame({
              "Market Tier": label_encoder.classes_,
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
  st.subheader(f"📊 Heatmap Analysis: {selected_architecture}")
  st.markdown(
      "Compare baseline performance against hybrid model enhancement on test"
      " data."
  )

  if X_test_scaled is None or y_test is None:
    st.warning("⚠️ Test dataset not detected from `loaddata.py`.")
  else:
    if base_model is None and hybrid_model is None:
      st.warning(
          "⚠️ Neither baseline nor hybrid model is loaded for"
          f" `{selected_architecture}`. Please save their `.pkl` files."
      )

    elif base_model is not None and hybrid_model is not None:
          st.markdown("#### 🔄 Side-by-Side Performance Comparison")

          # 1. Predictions from Hybrid Ensemble
          raw_hybrid = hybrid_model.predict(X_test_scaled)
          if isinstance(raw_hybrid[0], (int, np.integer)):
            pred_hybrid = label_encoder.inverse_transform(raw_hybrid)
          else:
            pred_hybrid = raw_hybrid

          # 2. Predictions from Standalone Base Model (Self-Healing Check)
          try:
            raw_base = base_model.predict(X_test_scaled)
          except Exception:
            # If base model wasn't fitted prior to saving, fit it directly on training data
            from loaddata import X_train_scaled, y_train

            # Encode y_train if necessary
            y_tr = (
                label_encoder.transform(y_train)
                if isinstance(y_train.iloc[0], str)
                else y_train
            )
            base_model.fit(X_train_scaled, y_tr)
            raw_base = base_model.predict(X_test_scaled)

          if isinstance(raw_base[0], (int, np.integer)):
            pred_base = label_encoder.inverse_transform(raw_base)
          else:
            pred_base = raw_base

          base_name_label = selected_architecture.split(" + ")[0]
          hybrid_name_label = selected_architecture

          fig = plot_side_by_side_confusion_matrix(
              y_true=y_test,
              y_pred_base=pred_base,
              y_pred_hybrid=pred_hybrid,
              class_names=label_encoder.classes_,
              base_name=f"Standalone {base_name_label}",
              hybrid_name=f"Hybrid ({hybrid_name_label})",
          )
          st.pyplot(fig)

          # Calculate and display accuracy gain metrics
          acc_base = (pred_base == y_test).mean() * 100
          acc_hybrid = (pred_hybrid == y_test).mean() * 100
          gain = acc_hybrid - acc_base

          m1, m2, m3 = st.columns(3)
          with m1:
            st.metric(
                f"Standalone {base_name_label} Accuracy", f"{acc_base:.2f}%"
            )
          with m2:
            st.metric(
                f"Hybrid ({hybrid_name_label}) Accuracy", f"{acc_hybrid:.2f}%"
            )
          with m3:
            st.metric(
                "Ensemble Performance Gain",
                f"+{gain:.2f}%" if gain >= 0 else f"{gain:.2f}%",
            )

    else:
      single_model = base_model if base_model is not None else hybrid_model
      single_name = (
          "Standalone Model"
          if base_model is not None
          else selected_architecture
      )

      raw_preds = single_model.predict(X_test_scaled)
      pred_labels = (
          label_encoder.inverse_transform(raw_preds)
          if isinstance(raw_preds[0], (int, np.integer))
          else raw_preds
      )

      fig = plot_confusion_matrix(
          y_true=y_test,
          y_pred=pred_labels,
          class_names=label_encoder.classes_,
          title=f"{single_name} Confusion Matrix",
      )
      st.pyplot(fig)