import streamlit as st
from transformers import pipeline

st.title("Emotion Detection from Text")

@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model="./emotion_model",
        tokenizer="./emotion_model"
    )

classifier = load_classifier()

text = st.text_area("Enter text")

label_map = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise",
}

if st.button("Detect Emotion"):
    if text:
        result = classifier(text)[0]
        emotion = label_map[result["label"]]
        confidence = result["score"]

        st.success(f"Emotion: **{emotion.upper()}**")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter some text first!")