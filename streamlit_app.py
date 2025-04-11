import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# โหลดโมเดล (แก้ path ให้ตรงกับโมเดลของคุณ)
model = YOLO("model.pt")

st.title("🪙 ตรวจจับเหรียญจากภาพ")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แปลงภาพเป็น OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("🔍 กำลังประมวลผล..."):
        # รันโมเดล
        results = model(image_np)[0]

        # วาดกล่องผลลัพธ์
        image_with_boxes = results.plot()  # ได้เป็น NumPy image ที่มีกรอบแล้ว

        # แสดงภาพที่มีกรอบ
        st.image(image_with_boxes, caption="🔍 ผลลัพธ์หลังตรวจจับ", channels="BGR", use_column_width=True)

        # นับเหรียญ
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy()

        value_map = {
            0: 1,
            1: 2,
            2: 5,
            3: 10
        }
        total_value = sum([value_map.get(int(c), 0) for c in class_ids])
        coin_count = len(class_ids)

        st.success(f"🔢 ตรวจพบ {coin_count} เหรียญ")
        st.success(f"💰 รวมมูลค่า: {total_value} บาท")

        # แสดงแต่ละชนิด
        for coin_class, count in zip(*np.unique(class_ids, return_counts=True)):
            coin_value = value_map.get(int(coin_class), "?")
            st.write(f"เหรียญ {coin_value} บาท: {count} เหรียญ")
