import streamlit as st
import time
import os
import joblib
from groq import Groq
import veritas_vision

st.set_page_config(
    page_title="Veritas",
    layout="wide",
    page_icon="🛡️"
)

GROQ_API_KEY=""

try:
    client = Groq(api_key=GROQ_API_KEY)
    api_status = "ONLINE"
except Exception as e:
    api_status = "OFFLINE"
    st.error(f"⚠️ Groq API Error: {e}. Chat will be disabled.")

@st.cache_resource
def load_nlp_model():
    try:
        return joblib.load('intent_classifier.pkl')
    except:
        return None

intent_pipeline = load_nlp_model()

def deepcheck_chat_reply(messages, context=None) -> str:
    if api_status == "OFFLINE" or not client:
        return "⚠️ System Error: Groq API Key is invalid or missing."
    system_prompt = (
        "You are Veritas Agent, a cold and analytical forensic intelligence system. "
        "Your tone is factual, concise, and objective. "
        "You analyze deepfake credibility and explain forensic indicators clearly."
    )
    if context:
        system_prompt += f"\n\n[ACTIVE SCAN DATA]\nVerdict: {context['label']}\nConfidence: {context['score']:.1f}%\n"
        system_prompt += f"Frames Scanned: {context['faces_analyzed']}\n"
        system_prompt += "Explain these findings technically (mention artifacts, frame inconsistency, or XceptionNet confidence)."
    formatted_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=formatted_messages,
            temperature=0.3,
            max_tokens=500,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Groq Inference Error: {e}"

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500&family=Orbitron:wght@600;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root { --bg1: #050516; --bg2: #120624; --accent: #00f7ff; }
[data-testid="stAppViewContainer"] > .main {
    background: radial-gradient(circle at 10% 0%, #34175f 0, #050516 50%, #050516 100%);
    background-size: 240% 240%;
    animation: bgMove 28s ease-in-out infinite;
    color: #f5f5ff !important;
}
@keyframes bgMove { 0% { background-position: 0% 0%; } 50% { background-position: 80% 80%; } 100% { background-position: 0% 0%; } }
[data-testid="stSidebar"] { background: rgba(5, 5, 22, 0.92) !important; backdrop-filter: blur(14px); border-right: 1px solid rgba(0, 247, 255, 0.35); color: #e9efff !important; }
.stButton>button {
    background: linear-gradient(90deg, #c03bff, #00f7ff) !important;
    border: 1px solid rgba(0, 247, 255, 0.6) !important;
    color: #050516 !important;
    border-radius: 999px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    box-shadow: 0 0 20px rgba(0, 247, 255, 0.9);
    transition: all 0.22s ease;
}
.stButton>button:hover { transform: translateY(-1px) scale(1.03); box-shadow: 0 0 28px rgba(0, 247, 255, 1); }
[data-testid="stFileUploaderDropzone"] {
    background: radial-gradient(circle at 0 0, rgba(192,59,255,0.28), rgba(5,5,22,0.95));
    border-radius: 18px !important;
    border: 1px solid rgba(0, 247, 255, 0.5) !important;
}
.status-pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 11px; font-family: 'Montserrat', sans-serif; letter-spacing: 1px; text-transform: uppercase; margin: 0 4px; }
.status-ok { background: rgba(0, 247, 132, 0.12); border: 1px solid rgba(0, 247, 132, 0.85); color: #a7ffda; }
.status-mode { background: rgba(0, 247, 255, 0.1); border: 1px solid rgba(0, 247, 255, 0.7); color: #c1fbff; }
.scanner-wrapper { margin-top: 10px; display: flex; flex-direction: column; align-items: center; }
.scanner-ring { width: 100px; height: 100px; border-radius: 50%; border: 1px solid rgba(0, 247, 255, 0.7); display: flex; align-items: center; justify-content: center; position: relative; box-shadow: 0 0 25px rgba(0, 247, 255, 0.5); }
.scanner-core { width: 64px; height: 64px; border-radius: 50%; border: 2px solid rgba(0, 247, 255, 0.9); box-shadow: 0 0 22px rgba(0, 247, 255, 0.9); position: relative; overflow: hidden; }
.scanner-sweep { position: absolute; width: 100%; height: 100%; background: conic-gradient(from 0deg, rgba(0,247,255,0.95), transparent 40%); animation: sweepRotate 1.6s linear infinite; }
@keyframes sweepRotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.scanner-label { margin-top: 8px; font-size: 11px; font-family: 'Montserrat', sans-serif; color: #9df4ff; text-transform: uppercase; letter-spacing: 1px; }
.ai-running { font-family: 'Montserrat', sans-serif; font-size: 12px; color: #cdefff; margin-top: 8px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

header_html = """
<div style='text-align:center; margin-top: -10px; margin-bottom: 5px;'>
    <h1 style="font-family: 'Orbitron', sans-serif; font-size: 46px; font-weight: 700; color: #e9efff; letter-spacing: 4px; text-shadow: 0px 0px 18px rgba(0,247,255,0.9);">
        VERITAS AGENT
    </h1>
    <p style="font-family: 'Montserrat', sans-serif; font-size: 14px; color: #b6c8ff; margin-top: -6px;">
        Neural forensics console for video authenticity & deepfake credibility assessment
    </p>
    <div style="margin-top: 6px;">
        <span class="status-pill status-ok">SYSTEM ONLINE</span>
        <span class="status-pill status-mode">DEEPFAKE ANALYSIS MODE</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,247,255,0.9), transparent); margin-top:6px; margin-bottom:22px;'>", unsafe_allow_html=True)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.markdown("""
<div style="font-family:Montserrat; font-size:14px; line-height:1.7; color:#e9efff;">
<span style="font-size:16px; font-weight:600; color:#7ef7ff;"> Veritas Agent HUD</span><br><br>
1️⃣ <b>Upload</b> a short video (mp4 / mov / avi).<br>
2️⃣ <b>(Optional)</b> enter a question about the clip.<br>
3️⃣ Hit <b>Run Deepfake Analysis</b> to start the engine.<br>
4️⃣ Review the <b>credibility score</b> and summary on the right.<br>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<h3 style='font-family:Montserrat; color:#e9efff; text-transform:uppercase; letter-spacing:1px;'>Upload Module</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Video received! Buffer locked.")
        st.write("Video preview:")
        st.video(uploaded)

with col2:
    st.markdown("<h3 style='font-family:Montserrat; color:#e9efff; text-transform:uppercase; letter-spacing:1px;'>Analysis Engine</h3>", unsafe_allow_html=True)

    question = st.text_input("Ask something about this video (optional):")
    analyze_btn = st.button("Run Deepfake Analysis")

    if analyze_btn:
        if uploaded is None:
            st.warning("Please upload a video before analyzing.")
        else:
            scanner_placeholder = st.empty()
            with scanner_placeholder.container():
                st.markdown("""
                <div class="scanner-wrapper">
                    <div class="scanner-ring"><div class="scanner-core"><div class="scanner-sweep"></div></div></div>
                    <div class="scanner-label">SCANNING VIDEO FRAMES</div>
                    <div class="ai-running">AI engine running<span class="ai-dots">...</span></div>
                </div>
                """, unsafe_allow_html=True)

            try:
                result = veritas_vision.analyze_video("temp_video.mp4")
                st.session_state.analysis_result = result
                time.sleep(1)
            except Exception as e:
                st.error(f"Critical System Failure: {e}")
            
            scanner_placeholder.empty()

            if st.session_state.analysis_result:
                res = st.session_state.analysis_result
                if "error" in res:
                    st.error(res['error'])
                else:
                    st.success("Analysis complete!")
                    result_box = st.container()
                    with result_box:
                        st.markdown("#### 📊 Credibility Report")
                        col_a, col_b = st.columns(2)
                        score_color = "inverse" if res['label'] == "FAKE" else "normal"
                        with col_a:
                            st.metric("Fake Probability", f"{res['fake_probability']:.4f}")
                        with col_b:
                            st.metric("Verdict", res['label'], delta="-High Risk" if res['label'] == "FAKE" else "+Authentic")
                        st.write(f"**Forensic Details:** Scanned {res['faces_analyzed']} frames using Custom XceptionNet.")
                        if res['label'] == "FAKE":
                            st.error("🚨 **ALERT:** Manipulation artifacts detected. High likelihood of Deepfake.")
                        else:
                            st.success("✅ **CLEAN:** No manipulation signatures found.")

st.markdown("## 💬 Veritas Agent Assistant")

for msg in st.session_state.chat_history:
    role = msg["role"]
    content = msg["content"]
    align = "flex-start" if role == "user" else "flex-end"
    color = "rgba(0,0,0,0.35)" if role == "user" else "rgba(255,255,255,0.10)"
    label = "Operator" if role == "user" else "Veritas Agent"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:{align}; margin-bottom:6px;">
          <div style="background:{color}; padding:8px 12px; border-radius:12px; max-width:70%; font-size:14px; border:1px solid rgba(255,255,255,0.15);">
            <div style="font-size:11px; opacity:0.7; margin-bottom:2px;"><b>{label}</b></div>
            <div>{content}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

user_msg = st.text_input("Ask Veritas Agent something:", key="chat_input")
col_chat_send, col_chat_clear = st.columns([1, 1])
send_clicked = col_chat_send.button("Send")
clear_clicked = col_chat_clear.button("Clear Chat")

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

if send_clicked and user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    intent = "general"
    if intent_pipeline:
        try:
            intent = intent_pipeline.predict([user_msg])[0]
        except:
            pass
    
    if intent == "analyze_video" and not st.session_state.analysis_result:
        reply_text = "I detect you want to analyze a video. Please upload it in the sidebar and click 'Run Deepfake Analysis'."
    else:
        current_score = None
        current_label = None
        if st.session_state.analysis_result and "fake_probability" in st.session_state.analysis_result:
            current_score = st.session_state.analysis_result['fake_probability'] * 100
            current_label = st.session_state.analysis_result['label']

        messages_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
        reply_text = deepcheck_chat_reply(messages_api, context=st.session_state.analysis_result)

    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
    st.rerun()

st.markdown("---")
st.caption("⚠ Prototype console. Built by Team Veritas.")
