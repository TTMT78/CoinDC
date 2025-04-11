import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# ---------- CONFIG ----------
CLASS_NAMES = ['1บาท', '2บาท', '5บาท', '10บาท', '50สตางค์', '25สตางค์']
COIN_VALUES = {'1บาท': 1, '2บาท': 2, '5บาท': 5, '10บาท': 10, '50สตางค์': 0.5, '25สตางค์': 0.25}

# ---------- FUNCTIONS ----------
def load_faster_rcnn_model(model_path):
    num_classes = 7  # 6 coins + background
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def draw_boxes(image, boxes, labels):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        draw.rectangle(box.tolist(), outline="red", width=3)
        draw.text((box[0], box[1]), label, fill="red", font=font)
    return image

def predict_yolo(image, model, conf_thres):
    results = model.predict(image, conf=conf_thres)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    labels = [CLASS_NAMES[c] for c in classes]
    return boxes, labels

def predict_faster_rcnn(image, model, conf_thres):
    transform = F.to_tensor(image)
    with torch.no_grad():
        outputs = model([transform])[0]
    boxes = outputs['boxes']
    scores = outputs['scores']
    labels = outputs['labels']
    keep = scores >= conf_thres
    boxes = boxes[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    label_names = [CLASS_NAMES[l - 1] for l in labels]  # Subtract 1 since bg = 0
    return boxes, label_names

def calculate_totals(labels):
    from collections import Counter
    count = Counter(labels)
    total = sum(COIN_VALUES[coin] * qty for coin, qty in count.items())
    return count, total

# ---------- STREAMLIT APP ----------
st.title("🪙 Coin Detector App")
st.markdown("อัปโหลดภาพเหรียญ ระบบจะตรวจจับประเภทและมูลค่าของเหรียญให้อัตโนมัติ")

model_type = st.selectbox("เลือกโมเดลที่ต้องการใช้", ["YOLOv8", "Faster R-CNN"])
conf_thres = st.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

uploaded_file = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    if model_type == "YOLOv8":
        model = YOLO("models/yolov8.pt")
        boxes, labels = predict_yolo(image, model, conf_thres)
    elif model_type == "Faster R-CNN":
        model = load_faster_rcnn_model("models/FasterRCNN.pth")
        boxes, labels = predict_faster_rcnn(image, model, conf_thres)

    if boxes is not None and labels:
        result_img = draw_boxes(image.copy(), boxes, labels)
        st.image(result_img, caption="ผลลัพธ์ที่ตรวจจับได้", use_column_width=True)

        count, total = calculate_totals(labels)
        st.subheader("🔍 สรุปผล")
        for coin, qty in count.items():
            st.write(f"{coin}: {qty} เหรียญ")
        st.write(f"💰 รวมมูลค่า: **{total:.2f} บาท**")
    else:
        st.warning("ไม่พบเหรียญในภาพหรือ confidence ต่ำเกินไป")
