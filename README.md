# 🧠 ResumeGPT (Data-Powered Edition)

**ResumeGPT** is an AI-powered resume improvement assistant that uses a curated dataset of high-quality bullet points from top-tier resumes. Unlike typical GPT-based tools, ResumeGPT suggests improvements based on real-world data – no OpenAI API key required.

---

## 🚀 Features

- 📝 Paste your current resume and get improved bullet points
- 📊 Suggestions powered by similarity search using `sentence-transformers`
- 📚 Data sourced from real resumes categorized by role (e.g., Data Scientist, Product Manager)
- ✅ No API keys or internet required — runs locally
- 📦 Built with Python + Streamlit for a fun, modern UX

---

## 📁 Dataset Structure

The dataset is stored in `Curated_Resume_Bullet_Dataset.csv` with the following columns:

| Column    | Description                                    |
|-----------|------------------------------------------------|
| Category  | Job domain (e.g., ACCOUNTANT, DATA SCIENTIST) |
| Bullets   | Resume bullet points (action-oriented lines)  |

---

## 💡 How It Works

1. Your resume is split into bullet points.
2. Each point is embedded using a sentence transformer.
3. The most similar real-world bullet points from the dataset are retrieved.
4. You get top 3 suggestions with similarity scores.
5. Final improved resume can be copied or downloaded.

---

## 🛠 Installation

```bash
git clone https://github.com/Likhith02/resumegpt-data.git
cd resumegpt-data
pip install -r requirements.txt
streamlit run resumegpt.py


