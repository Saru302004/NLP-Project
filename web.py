import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BlipProcessor, BlipForConditionalGeneration

# --- Page config ---
st.set_page_config(page_title="🫁 Lung Disease Assistant", layout="wide")

# --- Classes ---
classes = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "No Finding", "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# --- Load lung disease model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 15),
    nn.Sigmoid()
)
model.load_state_dict(torch.load("chest_xray_model.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Prediction function ---
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    preds = outputs[0].cpu().numpy()

    threshold = 0.5
    binary_preds = preds > threshold
    predicted_labels = [classes[i] for i, val in enumerate(binary_preds) if val]
    return predicted_labels, preds

# --- Load FLAN-T5 model for VQA ---
@st.cache_resource
def load_flan_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

tokenizer_flan, model_flan = load_flan_model()

def vqa_answer(disease, question):
    prompt = (
        f"The patient has been diagnosed with {disease}. "
        f"Answer the following question in a detailed, professional, and point-wise manner like a medical expert:\n\n"
        f"Question: {question}\n\n"
        f"Please provide at least 5 bullet points that elaborate on the answer with clear explanations, "
        f"including details about the treatment options, causes, risk factors, symptoms, and any other relevant medical information."
    )
    inputs = tokenizer_flan(prompt, return_tensors="pt")
    outputs = model_flan.generate(**inputs, max_new_tokens=400)
    answer = tokenizer_flan.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# --- Load BLIP model for Image Captioning ---
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor_blip, model_blip = load_blip_model()

def generate_image_caption(image):
    image = image.convert('RGB')
    inputs = processor_blip(images=image, return_tensors="pt")
    out = model_blip.generate(**inputs, max_new_tokens=100)
    caption = processor_blip.decode(out[0], skip_special_tokens=True)
    return caption

# --- UI ---
st.markdown(
    "<h1 style='text-align: center; color: #E83845;'>🫁 Lung Disease Detection & VQA Assistant</h1>",
    unsafe_allow_html=True,
)

st.markdown("---")

sidebar, main = st.columns([1.2, 3])

# --- Sidebar Upload ---
with sidebar:
    st.header("📤 Upload MRI Image")
    uploaded_file = st.file_uploader("Select an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Preview", width=320)
    else:
        st.warning("👆 Please upload a Lung MRI image to continue.", icon="⚠️")

# --- Main prediction + VQA + Captioning ---
with main:
    if uploaded_file:
        st.subheader("🖼️ Image Description :")
        caption = generate_image_caption(img)
        st.info(caption)

        st.markdown("---")
        st.subheader("🦠 Predicted Diseases:")
        labels, raw_scores = predict_image(img)

        if labels:
            st.success(", ".join(labels))
        else:
            st.info("No disease confidently detected.")

        st.subheader("📊 Confidence Scores:")
        confidence_df = pd.DataFrame({
            'Disease': classes,
            'Confidence': np.round(raw_scores, 4)
        }).sort_values(by="Confidence", ascending=False)

        st.dataframe(confidence_df, use_container_width=True)

        st.markdown("---")
        st.subheader("💬 Ask a Question (VQA)")
        question = st.text_input("Ask about the MRI...", placeholder="e.g., How is Pneumonia treated?")

        if st.button("🔍 Get Answer"):
            if question.strip() == "":
                st.warning("Type a valid question first!")
            else:
                if labels:
                    # Pick the top confident disease to answer about
                    main_disease = confidence_df.iloc[0]['Disease']
                    answer = vqa_answer(main_disease, question)
                    st.markdown(f"### 🧠 Answer:\n\n{answer}")
                else:
                    st.warning("No disease detected to answer questions about.")
    else:
        st.info("Waiting for image upload...", icon="🕐")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ by sushil and saravanan.")
