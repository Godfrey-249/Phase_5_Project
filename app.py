import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------- Load Models --------------------

# Load NLP model bundle
with open("model_nlp.pkl", "rb") as f:
    nlp_bundle = pickle.load(f)

nlp_model = nlp_bundle["model"]
disorder_embeddings = nlp_bundle["embeddings"]
signs_dict = nlp_bundle["signs_dict"]
df_recs = nlp_bundle["recommendations"]

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

# -------------------- NLP MODE --------------------
if mode == "Signs-Based":
    st.header(" Signs-based Disorder Prediction")
    user_input = st.text_area("Describe your signs or feelings here (e.g., 'I feel hopeless and anxious'):")

    if st.button(" Predict Disorder"):
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

        rec_row = df_recs[df_recs['Disorder'].str.lower() == top_disorder.lower()]
        if not rec_row.empty:
            st.subheader(" Recommendations")

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

# -------------------- STRUCTURED MODE --------------------
elif mode == "Lifestyle-Based":
    st.header(" Lifestyle-based Risk Prediction")

    # Input fields
    age = st.slider("Age", 10, 100, 25)
    gender = st.selectbox("Gender", encoders["Gender"].classes_)
    occupation = st.selectbox("Occupation", encoders["Occupation"].classes_)
    consultation = st.selectbox("Consultation History", encoders["Consultation_History"].classes_)
    stress = st.selectbox("Stress Level", encoders["Stress_Level"].classes_)
    sleep = st.slider("Sleep Hours", 4, 10, 6)
    work = st.slider("Work Hours per Week", 30, 80, 40)
    physical = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
    social = st.slider("Social Media Usage (hrs/day)", 0.5, 6.0, 3.0, step=0.5)
    diet = st.selectbox("Diet Quality", encoders["Diet_Quality"].classes_)
    smoking = st.selectbox("Smoking Habit", encoders["Smoking_Habit"].classes_)
    alcohol = st.selectbox("Alcohol Consumption", encoders["Alcohol_Consumption"].classes_)
    medication = st.selectbox("Medication Usage", encoders["Medication_Usage"].classes_)

    if st.button(" Assess Risk"):
        # Encode inputs
        sample = pd.DataFrame([{
            'Age': age,
            'Gender': encoders['Gender'].transform([gender])[0],
            'Occupation': encoders['Occupation'].transform([occupation])[0],
            'Consultation_History': encoders['Consultation_History'].transform([consultation])[0],
            'Stress_Level': encoders['Stress_Level'].transform([stress])[0],
            'Sleep_Hours': sleep,
            'Work_Hours': work,
            'Physical_Activity_Hours': physical,
            'Social_Media_Usage': social,
            'Diet_Quality': encoders['Diet_Quality'].transform([diet])[0],
            'Smoking_Habit': encoders['Smoking_Habit'].transform([smoking])[0],
            'Alcohol_Consumption': encoders['Alcohol_Consumption'].transform([alcohol])[0],
            'Medication_Usage': encoders['Medication_Usage'].transform([medication])[0]
        }])

        prediction = risk_model.predict(sample)[0]
        label = " Low Risk (No Disorder)" if prediction == 0 else " High Risk (Disorder Likely)"
        st.markdown(f"### Prediction: **{label}**")

        severity = severity_model.predict(sample)[0] if prediction == 1 else "N/A"
        if prediction == 1:
            st.markdown(f"### Estimated Severity: **{severity}**")

        # Save to session state history
        st.session_state.history.append({
            "Age": age,
            "Gender": gender,
            "Stress": stress,
            "Sleep": sleep,
            "Risk": label,
            "Severity": severity
        })

    # Show history
    if st.session_state.history:
        st.subheader(" Your Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)

        st.markdown("### Risk Outcome Distribution")
        risk_counts = history_df["Risk"].value_counts()
        st.bar_chart(risk_counts)

        st.download_button(
            label=" Download History as CSV",
            data=history_df.to_csv(index=False),
            file_name="mental_health_prediction_history.csv",
            mime="text/csv"
        )
