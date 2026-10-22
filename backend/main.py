from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)
    
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
    

#request body will be an object containing some str email field
class EmailRequest(BaseModel):
    email: str
    
@app.get("/")
def home():
    return {"Message": "Running Classifier"}

@app.post("/predict")
def predict_email(data: EmailRequest):
    email_vector = vectorizer.transform([data.email])
    
    probability = model.predict_proba(email_vector)[0]
    
    ham_probability = probability[0]
    spam_probability = probability[1]
    
    if spam_probability > ham_probability:
        prediction = "Spam"
        confidence = spam_probability
    else:
        prediction = "Ham"
        confidence = ham_probability
        
    return {"prediction": prediction, "confidence": round(float(confidence) * 100, 2)}
