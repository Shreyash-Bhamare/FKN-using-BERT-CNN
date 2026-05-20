import streamlit as st
import torch
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from transformers import BertTokenizer
from torchvision import transforms
import os
from thm import HybridModel  # Ensure this is available

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = HybridModel()
model.load_state_dict(torch.load("hybrid_model_2.pt", map_location=torch.device('cpu')))
model.eval()

CATEGORY_MAPPING = {0: "Real", 1: "Fake"}

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Streamlit UI Design ---
st.set_page_config(page_title="Fake News Detector", page_icon="🕵️‍♂️", layout="centered")
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #d63384;
            font-weight: 700;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #6c757d;
        }
        .result {
            font-size: 1.8rem;
            text-align: center;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 BERT + CNN Powered Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a news article URL and let my model determine its authenticity.</div>', unsafe_allow_html=True)

url = st.text_input("🔗 Paste the URL of a news article:", placeholder="https://example.com/news-article")

if st.button("🚀 Analyze Article"):
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text content
            title = soup.title.string if soup.title else ""
            description = ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag and "content" in desc_tag.attrs:
                description = desc_tag["content"]

            combined_text = title + " [SEP] " + description

            # Tokenize
            encoding = tokenizer(
                combined_text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Extract first image
            image = None
            img_tag = soup.find("img")
            if img_tag and "src" in img_tag.attrs:
                img_url = img_tag["src"]
                if not img_url.startswith("http"):
                    img_url = requests.compat.urljoin(url, img_url)

                img_response = requests.get(img_url)
                image = Image.open(BytesIO(img_response.content)).convert("RGB")
                transformed_img = image_transform(image).unsqueeze(0)
            else:
                transformed_img = torch.zeros((1, 3, 224, 224))

            # Prediction
            with torch.no_grad():
                output = model(input_ids, attention_mask, transformed_img)
                pred = torch.argmax(output, dim=1).item()
                st.markdown(f"<div class='result'>🧾 **Prediction**: {CATEGORY_MAPPING[pred]}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error processing the URL: {e}")
    else:
        st.warning("⚠️ Please enter a valid URL.")
