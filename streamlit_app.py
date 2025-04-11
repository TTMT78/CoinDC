import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO
import torchvision.transforms as T
import numpy as np
import cv2

# === CONFIG ===
coin_labels = ['1฿', '2฿', '5฿', '10฿', '0.5฿', '0.25฿']
coin_values = [1, 2, 5, 10, 0.5, 0.25]

# === MODEL SELECTION ===
model_option = st.selectbox("เลือกโมเดล", ["YOLOv8", "Faster R-CNN"])
confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)

# Load model based on selection
if model_option == "YOLOv8":
    model = YOLO("models/yolo.pt")  # เปลี่ยนเป็น path ของโมเดลคุณ
elif model_option == "Faster R-CNN":
    model = load_faster_rcnn_model("models/FasterRCNN.pth")

# === UPLOAD ===
uploaded_file = st.file_uploader("อัปโหลดรูปเหรียญ", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    if st.button("ตรวจจับเหรียญ"):
        if model_option == "YOLOv8":
            results = model(img_array)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            scores = results.boxes.conf.cpu().numpy()
        else:
            transform = T.Compose([T.ToTensor()])
            with torch.no_grad():
                predictions = model([transform(image)])[0]
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            classes = predictions['labels'].cpu().numpy()

        # Filter by confidence
        filtered_boxes = []
        filtered_classes = []
        for box, cls, score in zip(boxes, classes, scores):
            if score >= confidence:
                filtered_boxes.append(box)
                filtered_classes.append(cls)

        # Visualization
        result_img = img_array.copy()
        counter = [0] * 6  # 6 coin types
        for box, cls in zip(filtered_boxes, filtered_classes):
            if cls == 0 or cls > 6:
                continue  # background or unknown
            label = coin_labels[cls - 1]  # subtract background
            value = coin_values[cls - 1]
            counter[cls - 1] += 1

            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        total_value = sum([count * val for count, val in zip(counter, coin_values)])
        st.image(result_img, caption="ผลลัพธ์", use_column_width=True)

        st.markdown("### รายงานผล")
        for label, count in zip(coin_labels, counter):
            st.write(f"- {label}: {count} เหรียญ")
        st.write(f"**รวมมูลค่า: {total_value:.2f} บาท**")
