# app.py
# Streamlit frontend (Prediction only). No Train UI.
# Uses backend.py utilities directly (no TestClient, no httpx needed).

import streamlit as st
import backend as be  # our backend module with artifacts and preprocess

st.set_page_config(page_title="Manufacturing PPH Predictor", page_icon="⚙️", layout="wide")
st.title("⚙️ Manufacturing: Parts Per Hour Predictor")
st.caption("Prediction-only UI. Train locally once, then use this to predict.")

art = be.get_artifacts()
if not art.get("model") or not art.get("scaler") or not art.get("feature_cols"):
    st.error(
        "Model artifacts not found. Please train locally first using train_offline.py, "
        "then commit the 'artifacts/' folder to your repo."
    )
    st.stop()

feature_cols = art["feature_cols"]
encoders = art["encoders"] or {}
imputers = art["imputers"] or {"numeric_medians": {}, "categorical_modes": {}}
num_defaults = imputers.get("numeric_medians", {})
cat_defaults = imputers.get("categorical_modes", {})

st.subheader("Provide feature values")
with st.form("predict_form"):
    user_input = {}
    for col in feature_cols:
        if col in encoders:  # categorical
            classes = encoders[col].get("classes", [])
            default_val = cat_defaults.get(col, classes[0] if classes else "")
            # If too many classes or timestamp-like, use text input
            if len(classes) > 50 or col.lower() == "timestamp":
                user_input[col] = st.text_input(col, value=str(default_val))
            else:
                # Ensure default in options
                options = classes if classes else [str(default_val)]
                idx = 0
                user_input[col] = st.selectbox(col, options=options, index=idx)
        else:  # numeric
            default_val = float(num_defaults.get(col, 0.0))
            user_input[col] = st.number_input(col, value=default_val, step=0.1, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        X = be.preprocess_for_inference(user_input)
        X_scaled = art["scaler"].transform(X)
        pred = float(art["model"].predict(X_scaled)[0])
        st.success(f"Predicted Parts_Per_Hour: {pred:.3f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")