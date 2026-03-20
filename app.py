import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# Load model & tokenizer
# =========================
@st.cache_resource
def load_model():
    model_path = "model"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# Label mapping (PASTIKAN sama seperti training)
# =========================
id2label = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# =========================
# Prediction function
# =========================
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
    
    return id2label[pred]

# =========================
# UI
# =========================
st.title("Priority Classifier")

user_input = st.text_area("Enter query:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a query.")
    else:
        result = predict(user_input)
        
        st.success(f"Predicted Priority: **{result}**")