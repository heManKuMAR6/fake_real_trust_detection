import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model + tokenizer only once (cached)
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./distilbert_model")
    tokenizer = AutoTokenizer.from_pretrained("./distilbert_tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

st.title("📰 Fake News Detection App")
st.write("Enter any news text below and the model will classify it as **Real** or **Fake**.")

# Input text
text = st.text_area("Enter news text here:", height=200)

if st.button("Predict"):
    if len(text.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        # Tokenize
        enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        # Predict
        with torch.no_grad():
            outputs = model(
                enc["input_ids"],
                attention_mask=enc["attention_mask"]
            )
            prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "🟢 REAL NEWS" if prediction == 1 else "🔴 FAKE NEWS"

        st.write("### Prediction:")
        st.write(label)
