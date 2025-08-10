import streamlit as st
import torch
import json
import random
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# -------------------
# Load model & tokenizer
# -------------------
MODEL_PATH = "chatbot_model"  # This folder must be in the same repo as app.py

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    with open(f"{MODEL_PATH}/label_map.json", "r") as f:
        label_map = json.load(f)
    return tokenizer, model, label_map["label2id"], label_map["id2label"]

tokenizer, model, label2id, id2label = load_model()
model.eval()

# -------------------
# Load intents for responses
# -------------------
with open("simple_intents.json", "r") as f:
    intents_data = json.load(f)

responses_dict = {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}

# -------------------
# Prediction function
# -------------------
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
    intent_label = id2label[str(pred_id)] if isinstance(id2label, dict) else id2label[pred_id]
    return intent_label

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬")
st.title("💬 Mental Health Chatbot")
st.write("Hello! I’m here to chat with you. Please type your message below.")

# Maintain chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input.strip():
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "text": user_input})

    # Predict intent
    intent = predict_intent(user_input)
    # Choose a random response for that intent
    bot_reply = random.choice(responses_dict.get(intent, ["I'm not sure how to respond to that."]))

    # Add bot reply to history
    st.session_state["messages"].append({"role": "bot", "text": bot_reply})

# Display chat history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Bot:** {msg['text']}")
