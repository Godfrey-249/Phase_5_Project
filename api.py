from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import sentence_transformers

# Load your models (adjust paths as needed)
with open("model_nlp.pkl", "rb") as f:
    nlp_bundle = pickle.load(f)
nlp_model = nlp_bundle["model"]
disorder_embeddings = nlp_bundle["embeddings"]

with open("model_structured.pkl", "rb") as f:
    structured_bundle = pickle.load(f)
risk_model = structured_bundle["risk_model"]
severity_model = structured_bundle["severity_model"]
encoders = structured_bundle["encoders"]

app = FastAPI()

class SignsRequest(BaseModel):
    text: str

class LifestyleRequest(BaseModel):
    Age: int
    Gender: str
    Occupation: str
    Diet_Quality: str
    Smoking_Habit: str
    Alcohol_Consumption: str
    Sleep_Hours: float
    Physical_Activity_Hours: float
    Stress_Level: str
    Social_Media_Usage: float
    Consultation_History: str
    Medication_Usage: str
    Work_Hours: float

@app.post("/predict_signs")
def predict_signs(req: SignsRequest):
    input_embed = nlp_model.encode(req.text)
    similarities = {}
    for disorder, embed in disorder_embeddings.items():
        score = np.dot(input_embed, embed) / (np.linalg.norm(input_embed) * np.linalg.norm(embed))
        similarities[disorder] = score
    sorted_disorders = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
    return {"predictions": sorted_disorders}

@app.post("/predict_lifestyle")
def predict_lifestyle(req: LifestyleRequest):
    sample = np.array([[
        req.Age,
        encoders['Gender'].transform([req.Gender])[0],
        encoders['Occupation'].transform([req.Occupation])[0],
        encoders['Diet_Quality'].transform([req.Diet_Quality])[0],
        encoders['Smoking_Habit'].transform([req.Smoking_Habit])[0],
        encoders['Alcohol_Consumption'].transform([req.Alcohol_Consumption])[0],
        req.Sleep_Hours,
        req.Physical_Activity_Hours,
        encoders['Stress_Level'].transform([req.Stress_Level])[0],
        req.Social_Media_Usage,
        encoders['Consultation_History'].transform([req.Consultation_History])[0],
        encoders['Medication_Usage'].transform([req.Medication_Usage])[0],
        req.Work_Hours
    ]])
    risk = risk_model.predict(sample)[0]
    severity = severity_model.predict(sample)[0]
    return {"risk": int(risk), "severity": int(severity)}