import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Models
with open("model_structured.pkl", "rb") as f:
    structured_model = pickle.load(f)
with open("model_nlp.pkl", "rb") as f:
    nlp_model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

recommendations = pd.read_csv("recommendations.csv")
signs_df = pd.read_csv("merged_signs.csv")  # For NLP model

# UI Header
st.title(" Mental Health Risk & Disorder Detection")

tab1, tab2 = st.tabs([" Structured Risk Prediction", " Symptom-Based NLP Prediction"])

# --------------------- STRUCTURED INPUT PREDICTION ---------------------
with tab1:
    st.subheader("Predict Risk & Severity")
    
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    occupation = st.selectbox("Occupation", encoders['Occupation'].classes_)
    consultation = st.selectbox("Previous Consultation", ['Yes', 'No'])
    stress = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
    sleep = st.slider("Hours of Sleep", 0, 12, 6)
    work = st.slider("Work Hours", 0, 16, 8)
    activity = st.slider("Physical Activity Hours", 0, 5, 1)
    social = st.slider("Social Media Usage (hrs/day)", 0, 10, 3)
    diet = st.selectbox("Diet Quality", ['Healthy', 'Average', 'Unhealthy'])
    smoke = st.selectbox("Smoking Habit", ['Non-Smoker', 'Occasional Smoker', 'Regular Smoker', 'Heavy Smoker'])
    alcohol = st.selectbox("Alcohol Consumption", ['Non-Drinker', 'Social Drinker', 'Regular Drinker', 'Heavy Drinker'])
    medication = st.selectbox("On Medication", ['Yes', 'No'])

    if st.button("Predict Risk"):
        sample = pd.DataFrame([{
            'Age': age,
            'Gender': encoders['Gender'].transform([gender])[0],
            'Occupation': encoders['Occupation'].transform([occupation])[0],
            'Consultation_History': encoders['Consultation_History'].transform([consultation])[0],
            'Stress_Level': encoders['Stress_Level'].transform([stress])[0],
            'Sleep_Hours': sleep,
            'Work_Hours': work,
            'Physical_Activity_Hours': activity,
            'Social_Media_Usage': social,
            'Diet_Quality': encoders['Diet_Quality'].transform([diet])[0],
            'Smoking_Habit': encoders['Smoking_Habit'].transform([smoke])[0],
            'Alcohol_Consumption': encoders['Alcohol_Consumption'].transform([alcohol])[0],
            'Medication_Usage': encoders['Medication_Usage'].transform([medication])[0]
        }])

        risk = structured_model['risk'].predict(sample)[0]
        severity = structured_model['severity'].predict(sample)[0]

        st.markdown(f"### Risk of Mental Health Condition: {'Yes' if risk else 'No'}")
        if risk:
            st.markdown(f"### Estimated Severity: `{severity}`")

# --------------------- NLP SYMPTOM-BASED PREDICTION ---------------------
with tab2:
    st.subheader("Describe How You're Feeling")

    user_input = st.text_area("Write your symptoms or feelings", placeholder="e.g. I feel sad and anxious, I can’t sleep...")

    if st.button("Analyze Symptoms"):
        if user_input.strip():
            # Preprocess and predict
            clean_input = nlp_model['vectorizer'].transform([user_input])
            probs = nlp_model['classifier'].predict_proba(clean_input)[0]
            classes = nlp_model['classifier'].classes_

            results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]

            st.markdown("### Top Predicted Disorders:")
            for disorder, prob in results:
                st.markdown(f"- **{disorder}**: {prob:.2f}")

            top_disorder = results[0][0]
            rec_row = recommendations[recommendations['Disorder'].str.lower() == top_disorder.lower()]
            if not rec_row.empty:
                st.markdown(f"### Recommendations for {top_disorder}")
                st.markdown(f"**Self-care:** {rec_row.iloc[0]['Reccomendations; Self']}")
                st.markdown(f"**Professional Help:** {rec_row.iloc[0]['Reccomendation 2; Proffesional']}")
                st.markdown(f"**Other:** {rec_row.iloc[0]['Other Reccomendation']}")
        else:
            st.warning("Please type something about your symptoms.")

