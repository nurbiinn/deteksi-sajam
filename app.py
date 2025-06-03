import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
from io import BytesIO

# --- Konfigurasi ---
st.set_page_config(page_title="Deteksi Sajam", layout="centered")
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
models = {
    'YOLOv8': YOLO('model/yolov8.pt'),
    'YOLOv9': YOLO('model/yolov9.pt'),
    'YOLOv10': YOLO('model/yolov10.pt')
}
allowed_labels = ['pisau', 'pistol']

# --- Fungsi Deteksi ---
def predict_with_yolo(model, img_pil, save_path):
    results = model(img_pil)
    boxes = results[0].boxes
    names = results[0].names

    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id].lower()
        conf = float(box.conf[0])
        if label in allowed_labels:
            detections.append({
                'label': label,
                'confidence': round(conf * 100, 2)
            })

    # Simpan hasil prediksi
    results[0].save(save_path)
    return detections

# --- UI Streamlit ---
st.title("🔍 Deteksi Senjata Tajam Menggunakan YOLO")

st.markdown("Upload gambar dan pilih model deteksi yang diinginkan.")

# Pilih model
model_choice = st.selectbox("Pilih model YOLO:", list(models.keys()))

# Upload file
uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    if st.button("Deteksi Sekarang"):
        filename = uploaded_file.name
        result_filename = "result_" + filename
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)

        # Jalankan deteksi
        detections = predict_with_yolo(models[model_choice], image, result_path)

        # Tampilkan hasil
        st.subheader("🧠 Hasil Deteksi:")
        if detections:
            for det in detections:
                st.write(f"- {det['label'].title()} ({det['confidence']}%)")
        else:
            st.write("Tidak ada senjata tajam terdeteksi.")

        # Tampilkan gambar hasil
        result_image = Image.open(result_path)
        st.image(result_image, caption="Hasil Deteksi", use_column_width=True)