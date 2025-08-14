import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches

# Load NLP Model
with open("model_nlp.pkl", "rb") as f:
    nlp_bundle = pickle.load(f)
nlp_model = nlp_bundle["model"]
disorder_embeddings = nlp_bundle["embeddings"]
signs_dict = pd.read_excel("Final_Data.xlsx", engine='openpyxl')
df_recs = nlp_bundle["recommendations"]
df_signs = nlp_bundle["signs_dict"]

# Load Structured Model
with open("model_structured.pkl", "rb") as f:
    structured_bundle = pickle.load(f)
risk_model = structured_bundle["risk_model"]
severity_model = structured_bundle["severity_model"]
encoders = structured_bundle["encoders"]

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# UI Configuration
st.set_page_config(page_title="🧠 Mental Health Assistant", layout="centered")
st.title("🧠 Mental Health Prediction Assistant")
st.markdown("Predict mental health concerns using **signs/symptoms** or **lifestyle factors**.")

# Styling
st.markdown("""
    <style>
    /* Green Button Styling */
    .stButton>button,
    .stDownloadButton>button,
    .stForm button {
        background-color: #4CAF50 !important;
        color: black !important;
        font-weight: bold;
        border: none;
        border-radius: 6px;
    }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
        font-size: 16px;
        font-family: Arial, sans-serif;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Main Tabs
tab1, tab2 = st.tabs(["💬 Signs-Based Prediction", "📊 Lifestyle-Based Prediction"])

# NLP Model 
with tab1:
    st.subheader("💬 Enter Emotional Signs or Feelings")
    user_input = st.text_area("e.g., *I feel exhausted, anxious, and hopeless*")

    if st.button("🔍 Predict Disorder", key="predict_nlp"):
        if user_input.strip() == "":
            st.warning("Please enter some symptoms or emotional signs.")
        else:
            # Encode input and calculate similarities
            input_embed = nlp_model.encode(user_input)
            similarities = {
                disorder: util.cos_sim(input_embed, embed).item()
                for disorder, embed in disorder_embeddings.items()
            }
            sorted_disorders = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]

            st.success("✅ Prediction Completed!")
            st.markdown("### 🔝 Top Predicted Disorders")
            st.table(pd.DataFrame(sorted_disorders, columns=["Disorder", "Similarity Score"]))

            # Get top disorder
            top_disorder = sorted_disorders[0][0]
            st.markdown(f"### 🩺 Most Likely Concern: **{top_disorder}**")

            # Recommendations
            rec_row = df_recs[df_recs['Disorder'].str.lower() == top_disorder.lower()]
            if not rec_row.empty:
                with st.expander("📝 Recommendations"):
                    self_recs = rec_row['Reccomendations; Self'].dropna().values
                    prof_recs = rec_row['Reccomendation 2; Proffesional'].dropna().values
                    other_recs = rec_row['Other Reccomendation'].dropna().values

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**💡 Self-care**")
                        for val in self_recs[:3]:
                            st.markdown(f"- {val}")
                    with col2:
                        st.markdown("**🧑‍⚕️ Professional**")
                        for val in prof_recs[:2]:
                            st.markdown(f"- {val}")
                    with col3:
                        st.markdown("**🔁 Other**")
                        for val in other_recs[:2]:
                            st.markdown(f"- {val}")
            else:
                st.error("❌ No recommendations found for this disorder.")

            # Match disorder name to signs
            signs_dict.columns = signs_dict.columns.str.strip()
            signs_dict['Disorder'] = signs_dict['Disorder'].str.strip()
            possible_matches = get_close_matches(
                top_disorder.lower().strip(),
                signs_dict['Disorder'].str.lower(),
                n=1,
                cutoff=0.5
            )

            if possible_matches:
                matched_disorder = possible_matches[0]
                link_row = signs_dict[signs_dict['Disorder'].str.lower() == matched_disorder]
                if not link_row.empty:
                    link_col = [col for col in link_row.columns if "link" in col.lower()]
                    if link_col:
                        disorder_link = link_row[link_col[0]].iloc[0]
                        st.markdown(f"[🔗 Learn more about {matched_disorder.title()}]({disorder_link})")
                    else:
                        st.warning("⚠️ Link column not found in data.")
                else:
                    st.warning("⚠️ No exact row found for matched disorder.")
            else:
                st.warning("⚠️ No matching disorder found in data.")

#  Structured Model
with tab2:
    st.subheader("📊 Lifestyle Risk & Severity Estimation")

    with st.form("lifestyle_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 10, 100, 25)
            gender = st.selectbox("Gender", encoders["Gender"].classes_)
            occupation = st.selectbox("Occupation", encoders["Occupation"].classes_)
            consultation = st.selectbox("Consultation History", encoders["Consultation_History"].classes_)
            stress = st.selectbox("Stress Level", encoders["Stress_Level"].classes_)
        with c2:
            sleep = st.slider("Sleep Hours", 4, 10, 6)
            work = st.slider("Work Hours/week", 30, 80, 40)
            physical = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
            social = st.slider("Social Media (hrs/day)", 0.5, 6.0, 3.0, step=0.5)
            diet = st.selectbox("Diet Quality", encoders["Diet_Quality"].classes_)
            smoking = st.selectbox("Smoking Habit", encoders["Smoking_Habit"].classes_)
            alcohol = st.selectbox("Alcohol Consumption", encoders["Alcohol_Consumption"].classes_)
            medication = st.selectbox("Medication Usage", encoders["Medication_Usage"].classes_)

        submitted = st.form_submit_button("🔍 Assess Risk")

    if submitted:
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
        label = "🟢 Low Risk (No Disorder)" if prediction == 0 else "🔴 High Risk (Disorder Likely)"
        
        # Predict severity
        severity = severity_model.predict(sample)[0]
        
        # Map numeric severity to descriptive label
        severity_mapping = {
            0: "Mild",
            1: "Moderate",
            2: "Severe"
        }
        severity_label = severity_mapping.get(severity, "Unknown")

        st.markdown(f"### 🧠 Risk Prediction: **{label}**")

        if prediction == 1:
            st.markdown(f"### 🚨 Estimated Severity: **{severity_label}**")
            st.subheader("Lifestyle Recommendations")
        if label == "🔴 High Risk (Disorder Likely)":
            if severity_label == "Mild":
                st.markdown("**Self-Care Tips:**")
                st.markdown("- Practice daily mindfulness or journaling.")
                st.markdown("- Aim for 7–9 hours of sleep.")
                st.markdown("- Engage in regular physical activity (at least 3 hrs/week).")
                st.markdown("**Consider:**")
                st.markdown("- Reducing caffeine and alcohol.")
                st.markdown("- Talking to a trusted friend.")

            elif severity_label == "Moderate":
                st.markdown("**Lifestyle Adjustments:**")
                st.markdown("- Follow a structured routine for work and rest.")
                st.markdown("- Increase physical activity and limit screen time.")
                st.markdown("- Join peer support groups or therapy sessions.")
                st.markdown("**Consider Professional Help:**")
                st.markdown("- Schedule an appointment with a therapist or counselor.")
                st.markdown("- Seek advice from a wellness coach.")

            elif severity_label == "Severe":
                st.markdown("**Immediate Actions Recommended:**")
                st.markdown("- Contact a licensed mental health professional.")
                st.markdown("- Inform someone close to you for support.")
                st.markdown("- Prioritize rest, nutrition, and avoid isolation.")
                st.markdown("**Professional Help:**")
                st.markdown("- Consider clinical therapy and medication.")
                st.markdown("- Contact emergency support lines if overwhelmed.")
        else:
             st.success("No disorder detected at this time. Keep maintaining healthy habits.")

        # Save to session state history
        st.session_state.history.append({
            "Age": age,
            "Gender": gender,
            "Stress": stress,
            "Sleep": sleep,
            "Risk": label,
            "Severity": severity_label
        })

        # Show past assessments with expander, clear button, chart, and download option
        if st.session_state.history:
            with st.expander("🕘 Your Past Assessments"):
                history_df = pd.DataFrame(st.session_state.history)
                st.dataframe(history_df)

                # Clear history button directly below the table
                if st.button("🗑️ Clear My Past Assessments", key="clear_history"):
                    st.session_state.history.clear()
                    st.rerun()

                # Risk Outcome Distribution chart
                st.markdown("### 📈 Risk Outcome Distribution")
                st.bar_chart(history_df["Risk"].value_counts())

                # Download history button
                st.download_button(
                    label="⬇️ Download History as CSV",
                    data=history_df.to_csv(index=False),
                    file_name="mental_health_prediction_history.csv",
                    mime="text/csv"
                )




    



