import streamlit as st
import cv2
import numpy as np
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(page_title="Leaf Tracker Pro", layout="centered")

# --- Google Sheets Connection ---
# Make sure to add your Sheet URL in the Streamlit Cloud Secrets!
conn = st.connection("gsheets", type=GSheetsConnection)

st.title("🌿 Plant Data to Google Sheets")

# --- Initialize Memory ---
if 'total_area' not in st.session_state:
    st.session_state.total_area = 0.0
if 'leaf_count' not in st.session_state:
    st.session_state.leaf_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Plant Info ---
plant_name = st.text_input("Enter Plant Name/ID:", placeholder="e.g. Tomato_Plant_01")

# --- Metrics ---
col1, col2 = st.columns(2)
col1.metric("Leaves Found", st.session_state.leaf_count)
col2.metric("Total Area", f"{st.session_state.total_area:.2f} sq in")

# --- Image Processing ---
uploaded_file = st.file_uploader("Capture leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Process image (Greyscale -> Blur -> Edge)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    target_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            target_contour = approx
            break

    if target_contour is not None:
        # Perspective Correction Logic
        pts = target_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (400, 300))

        # Green Mask
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([90, 255, 255]))
        
        current_area = (cv2.countNonZero(mask) / (400 * 300)) * 12.0
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption=f"Area: {current_area:.2f} sq in")

        if st.button("✅ Add Leaf to Plant", use_container_width=True):
            st.session_state.total_area += current_area
            st.session_state.leaf_count += 1
            st.session_state.history.append(round(current_area, 3))
            st.rerun()

st.divider()

# --- Upload to Google Sheets ---
if st.session_state.leaf_count > 0:
    if st.button("📤 Upload Plant Data to Google Sheets", type="primary", use_container_width=True):
        if not plant_name:
            st.error("Please enter a Plant Name before uploading!")
        else:
            try:
                # Prepare data row
                new_data = pd.DataFrame([{
                    "Plant Name": plant_name,
                    "Leaf Count": st.session_state.leaf_count,
                    "Individual Areas": str(st.session_state.history),
                    "Total Area": round(st.session_state.total_area, 3)
                }])
                
                # Append to sheet
                conn.create(data=new_data)
                st.success(f"Data for {plant_name} uploaded successfully!")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# Reset
if st.button("🗑️ Reset Everything", use_container_width=True):
    st.session_state.total_area = 0.0
    st.session_state.leaf_count = 0
    st.session_state.history = []
    st.rerun()
