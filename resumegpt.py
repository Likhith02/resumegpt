import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import re
import os

@st.cache_data
def load_data():
    df = pd.read_csv("Curated_Resume_Bullet_Dataset.csv")

    return df

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9., ]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_bullets(text):
    return [line.strip() for line in re.split(r'[\n\r\.]+', text) if len(line.strip().split()) > 4]

st.title("ResumeGPT (Data-Powered Edition) 💼✨")
st.markdown("Paste your resume below. We'll suggest better bullet points based on top real resumes.")

df = load_data()
df["Cleaned_Resume"] = df["Bullets"].apply(clean_text)

resume_lines = []
for _, row in df.iterrows():
    lines = split_into_bullets(row["Cleaned_Resume"])
    resume_lines.extend([line for line in lines if len(line.split()) >= 5])

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()
resume_embeddings = model.encode(resume_lines, convert_to_tensor=True)

user_resume = st.text_area("Paste your resume:", height=300)
final_suggestions = []

if st.button("🚀 Improve My Resume") and user_resume:
    user_lines = split_into_bullets(clean_text(user_resume))
    user_embeddings = model.encode(user_lines, convert_to_tensor=True)

    st.markdown("---")
    st.subheader("🔁 Suggestions Based on Real Resumes")

    for i, user_line in enumerate(user_lines):
        similarities = util.pytorch_cos_sim(user_embeddings[i], resume_embeddings)[0]
        top_indices = np.argsort(-similarities)[:3]  # top 3

        st.markdown(f"**📝 Your line:** `{user_line}`")
        suggestions = []

        for rank, idx in enumerate(top_indices):
            suggestion = resume_lines[idx]
            score = round(similarities[idx].item() * 100, 2)
            suggestions.append(suggestion)
            st.markdown(f"- ✅ **{suggestion}**  ")
            st.caption(f"Similarity: {score}%")

        final_suggestions.append(suggestions[0])  
        st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("📄 Final Improved Resume")
    full_text = "\n".join(final_suggestions)
    st.text_area("Copy this:", full_text, height=300)
    st.download_button("📥 Download as TXT", full_text, file_name="Improved_Resume.txt")
