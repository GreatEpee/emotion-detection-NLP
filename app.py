import streamlit as st
from transformers import pipeline
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

@st.cache_resource
def build_fuzzy_system():
    confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
    intensity = ctrl.Consequent(np.arange(0, 101, 1), 'intensity')

    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
    confidence['medium'] = fuzz.trimf(confidence.universe, [0.2, 0.5, 0.8])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1.0, 1.0])

    intensity['mild'] = fuzz.trimf(intensity.universe, [0, 0, 50])
    intensity['moderate'] = fuzz.trimf(intensity.universe, [25, 50, 75])
    intensity['extreme'] = fuzz.trimf(intensity.universe, [50, 100, 100])

    intensity.defuzzify_method = 'mom'

    rule1 = ctrl.Rule(confidence['low'], intensity['mild'])
    rule2 = ctrl.Rule(confidence['medium'], intensity['moderate'])
    rule3 = ctrl.Rule(confidence['high'], intensity['extreme'])

    intensity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    return intensity_ctrl, intensity

def get_emotion_intensity(intensity_ctrl, intensity_consequent, confidence_score):
    intensity_sim = ctrl.ControlSystemSimulation(intensity_ctrl)
    intensity_sim.input['confidence'] = confidence_score
    intensity_sim.compute()
    
    score = intensity_sim.output['intensity']
    
    mild_mf = fuzz.interp_membership(intensity_consequent.universe, intensity_consequent['mild'].mf, score)
    mod_mf = fuzz.interp_membership(intensity_consequent.universe, intensity_consequent['moderate'].mf, score)
    ext_mf = fuzz.interp_membership(intensity_consequent.universe, intensity_consequent['extreme'].mf, score)
    
    memberships = {'Mild': mild_mf, 'Moderate': mod_mf, 'Extreme': ext_mf}
    label = max(memberships, key=memberships.get)
    
    return score, label

st.title("Emotion Detection from Text")

@st.cache_resource
def load_classifier():
    return pipeline(
        "text-classification",
        model="./emotion_model",
        tokenizer="./emotion_model"
    )

classifier = load_classifier()
intensity_ctrl, intensity_consequent = build_fuzzy_system()

if 'user_text' not in st.session_state:
    st.session_state.user_text = ""

def set_suggested_text(suggested_text):
    st.session_state.user_text = suggested_text

def clear_text():
    st.session_state.user_text = ""

st.write("**Try a suggested sentence:**")
col1, col2, col3 = st.columns(3)

col1.button("I am so glad I came here!", 
            on_click=set_suggested_text, 
            args=("I am so glad I came here!",), 
            use_container_width=True)

col2.button("This is the worst day of my life.", 
            on_click=set_suggested_text, 
            args=("This is the worst day of my life.",), 
            use_container_width=True)

col3.button("Omg, I'm so afraid of my sir!", 
            on_click=set_suggested_text, 
            args=("Omg, I'm so afraid of my sir!",), 
            use_container_width=True)
            
text = st.text_area("Enter text", key="user_text")

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
        
        intensity_score, intensity_label = get_emotion_intensity(intensity_ctrl, intensity_consequent, confidence)

        st.success(f"Emotion: **{emotion.upper()}**")
        st.write(f"Intensity: {intensity_label} {emotion.capitalize()} ({intensity_score:.0f}/100)")
        st.write(f"*(Raw model confidence: {confidence:.2f})*")
        
        st.button("Check Another Sentence", on_click=clear_text)

    else:
        st.warning("Please enter some text first!")