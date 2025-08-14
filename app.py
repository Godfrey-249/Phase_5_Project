import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sentence_transformers
import sentence_transformers.util as util
import difflib
import requests

# -------------------- Load Models --------------------

with open("model_structured.pkl", "rb") as f:
    structured_bundle = pickle.load(f)

lifestyle_model = structured_bundle["risk_model"]      # Risk prediction model
severity_model = structured_bundle["severity_model"]   # Severity prediction model
encoders = structured_bundle["encoders"]               # Label encoders
features = structured_bundle["features"]               # Feature list



# Load Structured model bundle
with open("model_structured.pkl", "rb") as f:
    structured_bundle = pickle.load(f)

risk_model = structured_bundle["risk_model"]
severity_model = structured_bundle["severity_model"]
encoders = structured_bundle["encoders"]

# -------------------- Session State --------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- UI --------------------

st.set_page_config(page_title="Mental Health Assistant", layout="centered")
st.title(" Mental Health Prediction Assistant")
st.markdown("Predict mental health concerns from **symptoms** or **lifestyle patterns**.")

# Sidebar choice
mode = st.sidebar.radio(" Choose Prediction Mode", ["Signs-Based", "Lifestyle-Based"])

tab1, tab2 = st.tabs(["Signs-Based", "Lifestyle-Based"])

with tab1:
    st.header(" Signs-based Disorder Prediction")
    user_input = st.text_area("Describe your signs or feelings here (e.g., 'I feel hopeless and anxious'):")

    # Example for signs-based prediction
    if st.button(" Predict Disorder"):
        response = requests.post(
            "http://localhost:8000/predict_signs",
            json={"text": user_input}
        )
        result = response.json()
        st.write(result["predictions"])

        input_embed = nlp_model.encode(user_input)
        similarities = {}

        for disorder, embed in disorder_embeddings.items():
            score = util.cos_sim(input_embed, embed).item()
            similarities[disorder] = score

        sorted_disorders = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]

        st.subheader(" Predicted Disorders")
        for disorder, score in sorted_disorders:
            st.write(f"- **{disorder}**: {score:.2f}")

        top_disorder = sorted_disorders[0][0]
        st.markdown(f"###  Top Concern: **{top_disorder}**")

        # Fuzzy match to handle typos
        disorders_list = [d.strip().lower() for d in df_recs['Disorder'].unique()]
        closest = difflib.get_close_matches(top_disorder.strip().lower(), disorders_list, n=1, cutoff=0.7)

        if closest:
            matched_disorder = closest[0]
            rec_row = df_recs[df_recs['Disorder'].str.strip().str.lower() == matched_disorder]
        else:
            rec_row = pd.DataFrame()  # Empty DataFrame

        if not rec_row.empty:
            st.subheader(" Recommendations")
            with st.expander("Show Recommendations"):
                self_recs = rec_row['Reccomendations; Self'].dropna().values
                prof_recs = rec_row['Reccomendation 2; Proffesional'].dropna().values
                other_recs = rec_row['Other Reccomendation'].dropna().values

                st.markdown("**Self-care:**")
                for i, val in enumerate(self_recs[:3], 1):
                    st.markdown(f"- Option {i}: {val}")

                st.markdown("**Professional Help:**")
                for i, val in enumerate(prof_recs[:2], 1):
                    st.markdown(f"- Option {i}: {val}")

                st.markdown("**Other Recommendations:**")
                for i, val in enumerate(other_recs[:2], 1):
                    st.markdown(f"- Option {i}: {val}")
        else:
            st.error("No recommendations found.")



# =========================
# SAFE PREDICTION FUNCTION
# =========================

def safe_predict(model, sample_df, feature_order):
    try:
        sample_df = sample_df[feature_order]
    except KeyError as e:
        st.error(f"Missing/extra features for prediction: {e}")
        return None
    return model.predict(sample_df)[0], getattr(model, "predict_proba", lambda x: None)(sample_df)

# ---------- TAB 2 ----------
# =========================
# TAB 2: Lifestyle Risk Prediction
# =========================
with tab2:
    st.header("Lifestyle-Based Mental Health Risk Assessment")

    # Load the full structured model bundle
    with open("model_structured.pkl", "rb") as f:
        structured_bundle = pickle.load(f)

    risk_model = structured_bundle["risk_model"]
    severity_model = structured_bundle["severity_model"]
    encoders = structured_bundle["encoders"]
    features = structured_bundle["features"]

    # Inputs
    age = st.number_input("Age", min_value=1, max_value=120, value=25)

    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    occupation = st.selectbox("Occupation", encoders["Occupation"].classes_)
    consultation = st.selectbox("Consultation History", encoders["Consultation_History"].classes_)

    stress = st.selectbox("Stress Level", encoders["Stress_Level"].classes_)
    stress_map = {"Low": 0, "Medium": 1, "High": 2 } 

    sleep_hours = st.number_input("Average Sleep Hours", min_value=0, max_value=10, value=7, step=1)
    physical_activity = st.number_input("Physical Activity Hours", min_value=0, max_value=10, value=1, step=1)
    social_media_hours = st.number_input("Social Media Usage (Hours)", min_value=0, max_value=6, value=2, step=1)

    diet_quality = st.selectbox("Diet Quality", ["Healthy", "Unhealthy", "Average"])
    diet_map = {"Healthy": 2, "Unhealthy": 0, "Average": 1}

    smoke_map = {"Non-Smoker": 0, "Occasional Smoker": 1, "Regular Smoker": 2, "Heavy Smoker": 3}
    smoking = st.selectbox("Smoking Habit", ["Non-Smoker", "Occasional Smoker", "Regular Smoker", "Heavy Smoker"])

    alcohol_map = {"Non-Drinker": 0, "Occasional Drinker": 1, "Regular Drinker": 2, "Heavy Drinker": 3}
    alcohol= st.selectbox("Alcohol Consumption", ["Non-Drinker", "Occasional Drinker", "Regular Drinker", "Heavy Drinker"])

    medication = st.selectbox("Medication Usage", encoders["Medication_Usage"].classes_)
    work_hours = st.number_input("Average Work Hours per week", min_value=0.0, max_value=80.0, value=8.0, step=5.0)

    if st.button("Predict Risk"):
        # Encode categorical features using the trained encoders
        sample_dict = {
            'Age': age,
            'Gender': encoders['Gender'].transform([gender])[0],
            'Occupation': encoders['Occupation'].transform([occupation])[0],
            'Consultation_History': encoders['Consultation_History'].transform([consultation])[0],
            'Stress_Level': stress_map[stress],
            'Sleep_Hours': sleep_hours,
            'Work_Hours': work_hours,
            'Physical_Activity_Hours': physical_activity,
            'Social_Media_Usage': social_media_hours,
            'Diet_Quality': diet_map[diet_quality],
            'Smoking_Habit': smoke_map[smoking],
            'Alcohol_Consumption': alcohol_map[alcohol],
            'Medication_Usage': encoders['Medication_Usage'].transform([medication])[0]
        }

        # Ensure feature order matches training
        sample_df = pd.DataFrame([[sample_dict[col] for col in features]], columns=features)

        # Predict
        risk_out = risk_model.predict(sample_df)[0]
        sev_out = severity_model.predict(sample_df)[0]

        # Display results
        if risk_out == 1:
            st.markdown("**Mental Health Risk:** High Risk (Disorder Likely)")
            if sev_out == 0:
                st.markdown("Severity: Mild symptoms detected. Consider monitoring and self-care.")
            elif sev_out == 1:
                st.markdown("Severity: Moderate symptoms. Professional consultation recommended.")
            else:
                st.markdown("Severity: Severe symptoms. Seek immediate help.")
        else:
            st.markdown("**Mental Health Risk:** Low Risk (No Disorder)")
            st.markdown("No disorder detected at this time. Keep maintaining healthy habits.")
