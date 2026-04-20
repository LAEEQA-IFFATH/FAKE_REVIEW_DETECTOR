import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Use pre-trained BERT (no local files needed)
MODEL_NAME = "bert-base-uncased"

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️")

st.title("Fake Review Detector (BERT)")
st.markdown("Detect whether a review is **Real or Fake** using AI")

review = st.text_area("Enter a review:")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        real_prob = probs[0][0].item()
        fake_prob = probs[0][1].item()

        st.subheader("Result")

        if fake_prob > real_prob:
            st.error(f"Fake Review ({fake_prob:.2%})")
        else:
            st.success(f"Real Review ({real_prob:.2%})")

        st.subheader("Confidence Breakdown")
        st.write(f"Fake: {fake_prob:.2%}")
        st.write(f"Real: {real_prob:.2%}")
        st.progress(float(max(fake_prob, real_prob)))