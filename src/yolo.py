import streamlit as st
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image

# =============================
# Functions
# =============================

def detect_license_plate_traditional(image_np):
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    for contour in cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5 and w > 100 and h > 20:
                return img_rgb, edges, img_rgb[y:y+h, x:x+w]
    return img_rgb, edges, None

def detect_license_plate_yolov8(image_np):
    model = YOLO('yolov8n.pt')  # Small YOLOv8 model
    results = model(image_np)

    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            return img_rgb, img_rgb[y1:y2, x1:x2]
    return img_rgb, None

def extract_plate_number(license_plate_img):
    gray_plate = cv2.cvtColor(license_plate_img, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray_plate, config='--psm 8')
    return text.strip()

# =============================
# Streamlit UI
# =============================
# --- Streamlit UI ---
st.title("🚗 License Plate Detection")
st.write("Upload an image and detect license plates using traditional or YOLOv8 method.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = np.array(Image.open(uploaded_file).convert('RGB'))
    st.image(image, caption='Uploaded Image', use_container_width=True)

    method = st.radio("Select Detection Method:", ["Traditional", "YOLOv8"])

    if st.button("Detect License Plate"):
        if method == "Traditional":
            img_rgb, edges, license_plate = detect_license_plate_traditional(image)
            if license_plate is None:
                st.warning("License Plate Not Found with Traditional Method! Trying YOLOv8...")
                img_rgb, license_plate = detect_license_plate_yolov8(image)
                edges = np.zeros_like(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
        else:
            img_rgb, license_plate = detect_license_plate_yolov8(image)
            edges = np.zeros_like(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))

        plate_number = None
        if license_plate is not None:
            plate_number = extract_plate_number(license_plate)
            st.success(f"Detected License Plate Number: {plate_number}")
        else:
            st.error("License Plate could not be detected!")

        # Display results
        st.image(img_rgb, caption="Original Image / YOLO Detection", use_container_width=True)
        if method == "Traditional":
            st.image(edges, caption="Edge Detection", use_container_width=True)
        if license_plate is not None:
            st.image(license_plate, caption="Cropped License Plate", use_container_width=True)
