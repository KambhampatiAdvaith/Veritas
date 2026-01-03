import torch
import timm
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import os

# --- CONFIGURATION ---
MODEL_PATH = 'veritas_production_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD RESOURCES ---
print(f"🔌 Loading Veritas Vision on {DEVICE}...")

# 1. Face Detector
# keep_all=False ensures we get the single main face
mtcnn = MTCNN(
    image_size=299, margin=40, keep_all=False, 
    select_largest=True, post_process=False, device=DEVICE
)

# 2. Deepfake Detector
model = timm.create_model('legacy_xception', pretrained=False, num_classes=1)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Custom Weights Loaded!")
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}")
else:
    print(f"⚠️ WARNING: '{MODEL_PATH}' not found.")

model.to(DEVICE)
model.eval()

# 3. Exact Training Transform (The "Safe Pipeline")
# This ensures inference data matches training data 100%
val_transforms = transforms.Compose([
    transforms.ToTensor(),               # Converts 0-255 to 0-1
    transforms.Normalize([0.5]*3, [0.5]*3) # Normalizes to -1 to 1
])

def analyze_video(video_path, frame_skip=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    fake_probs = []
    frames_processed = 0
    frame_count = 0
    
    print(f"\n🔍 Analyzing: {os.path.basename(video_path)}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        if frame_count % frame_skip == 0:
            try:
                # Convert BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Detect Face
                face_tensor = mtcnn(pil_img)
                
                if face_tensor is not None:
                    # --- NORMALIZATION FIX ---
                    # MTCNN returns 0-255 tensor (if post_process=False)
                    # We convert to PIL Image first to use standard transforms
                    face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype('uint8')
                    face_pil = Image.fromarray(face_np)
                    
                
                    input_tensor = val_transforms(face_pil).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(input_tensor)
                        prob = torch.sigmoid(output).item()
                        
                        
                        fake_prob = 1.0 - prob
                        
                        fake_probs.append(fake_prob)
                        frames_processed += 1
                        
                        

            except Exception as e:
                pass
        frame_count += 1
    
    cap.release()

    if frames_processed == 0:
        return {"error": "No faces detected in video."}


    avg_fake_prob = sum(fake_probs) / len(fake_probs)
    
 
    THRESHOLD = 0.40
    
    if avg_fake_prob > THRESHOLD:
        label = "FAKE"
        confidence = avg_fake_prob * 100
    else:
        label = "REAL"
        confidence = (1.0 - avg_fake_prob) * 100

    print(f"📊 FINAL CALC: Avg Fake Score = {avg_fake_prob:.4f} -> {label}")

    return {
        "label": label,
        "score": confidence,
        "fake_probability": avg_fake_prob,
        "faces_analyzed": frames_processed
    }