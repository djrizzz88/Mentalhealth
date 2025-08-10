import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model from Hugging Face
MODEL_NAME = "jordan88rali/mental-health-chatbot"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Class labels (must match your training)
label_map = {
    0: "about",
    1: "anxious",
    2: "default",
    3: "depressed",
    4: "goodbye",
    5: "greeting",
    6: "happy",
    7: "help",
    8: "sad",
    9: "stressed",
    10: "suicide",
    11: "thanks"
}

# Streamlit UI
st.title("🧠 Mental Health Chatbot")
st.write("Type your message below and I'll detect your intent.")

user_input = st.text_input("You:", "")

if st.button("Predict") and user_input.strip():
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    st.write(f"**Predicted Intent:** {label_map[predicted_class]}")
