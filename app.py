# app.py
import streamlit as st
from PIL import Image

from model_utils import load_model, predict, MODEL_SPECS

st.set_page_config(page_title="Model Demo", layout="centered")

st.title("E-Commerce Product Classification Demo")
st.write("Choose a model, then provide the required inputs to get a prediction.")

model_key = st.selectbox(
    "Select model",
    list(MODEL_SPECS.keys()),
    index=0
)

@st.cache_resource
def get_model_cached(key: str):
    return load_model(key)

model = get_model_cached(model_key)

st.divider()
st.caption(f"Model selected: **{model_key}** | Required inputs: {', '.join(MODEL_SPECS[model_key]['requires'])}")

text = None
img = None

if "image" in MODEL_SPECS[model_key]["requires"]:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

if "text" in MODEL_SPECS[model_key]["requires"]:
    text = st.text_area("Paste product text/description", height=140, placeholder="e.g., title + description...")

run = st.button("Predict", type="primary", use_container_width=True)

if run:
    if "image" in MODEL_SPECS[model_key]["requires"] and img is None:
        st.error("Please upload an image.")
        st.stop()
    if "text" in MODEL_SPECS[model_key]["requires"] and (text is None or not text.strip()):
        st.error("Please enter text.")
        st.stop()

    with st.spinner("Predicting..."):
        label, conf, top5 = predict(model_key, model, img=img, text=text)

    st.subheader("Prediction")
    st.write(f"**{label}**")
    st.write(f"Confidence: **{conf*100:.2f}%**")

    st.subheader("Top predictions")
    for name, p in top5:
        st.write(f"- {name}: {p*100:.2f}%")
