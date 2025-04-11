import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# โหลดโมเดล
model = YOLO("model.pt")  # เปลี่ยนเป็น path ที่ถูกต้อง

st.title("🪙 ตรวจจับเหรียญจากภาพ")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพ
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปที่อัปโหลด', use_column_width=True)

    # ทำงานกับโมเดล
    with st.spinner("🔍 กำลังประมวลผล..."):
        image_np = np.array(image)
        results = model(image_np)[0]  # รัน YOLO
        
        # แสดงผลลัพธ์
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy()
        
        # นับเหรียญและคำนวณมูลค่า
        coin_count = len(class_ids)
        value_map = {
            0: 1,  # 1 บาท
            1: 2,  # 2 บาท
            2: 5,
            3: 10
        }
        total_value = sum([value_map.get(int(c), 0) for c in class_ids])
        
        # แสดงผล
        st.success(f"🔢 ตรวจพบ {coin_count} เหรียญ")
        st.success(f"💰 รวมมูลค่า: {total_value} บาท")

        # แยกตามประเภท
        for coin_class, count in zip(*np.unique(class_ids, return_counts=True)):
            st.write(f"เหรียญ {value_map.get(int(coin_class), '?')} บาท: {count} เหรียญ")
