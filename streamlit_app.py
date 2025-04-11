import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import os
import requests

# URL ของโมเดลบน GitHub (เปลี่ยนเป็นลิงก์ไฟล์จริง)
MODEL_URL = "https://github.com/username/repo/raw/main/model.pt"
MODEL_PATH = "model.pt"

# โหลดโมเดลจาก GitHub ถ้าไม่มีไฟล์
if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.write("✅ Model downloaded!")

# โหลดโมเดล YOLO
model = YOLO(MODEL_PATH)

st.title("💰 ระบบนับเหรียญจากภาพ")
st.write("อัปโหลดภาพเหรียญ แล้วให้ AI ช่วยนับจำนวนและคำนวณมูลค่ารวม")

# อัปโหลดไฟล์ภาพ
uploaded_file = st.file_uploader("📷 อัปโหลดภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # อ่านภาพ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # รันโมเดล YOLO
    results = model(image)

    # แสดงผลลัพธ์
    st.image(image, channels="BGR")
