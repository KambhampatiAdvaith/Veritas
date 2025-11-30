# 🧠 Veritas Agent — Deepfake Detection Assistant

Veritas Agent is an AI-powered forensic system that helps users verify the authenticity of videos. It provides a credibility score and natural-language explanations rather than just returning “real” or “fake.”
“Veritas” means truth - and this system exists to verify it. 🔍✨

---

### Why It Matters

Deepfakes are becoming more accessible and harder to detect. Existing tools are often technical, inaccessible, or provide no reasoning. Veritas Agent bridges the gap with a user-friendly forensic assistant.

---

### How It Works

1. User uploads or selects a demo video  
2. Frames and faces are extracted  
3. Deepfake models (MesoNet/XceptionNet) run classification  
4. An AI assistant explains results in a readable format  

---

### Tech Stack

- **Streamlit** (Frontend)
- **OpenCV + MTCNN** (Face/Frame processing)
- **PyTorch (MesoNet/XceptionNet)** (Deepfake Models)
- **Scikit-learn** (Intent classification)
- **Groq / Hugging Face / OpenAI / Gemini** (LLM explanation)

---

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

