import streamlit as st
import cv2
import numpy as np
from streamlit_gsheets import GSheetsConnection
import pandas as pd

st.set_page_config(page_title="Leaf Tracker Pro", layout="centered")

# --- Google Sheets Connection ---
conn = st.connection("gsheets", type=GSheetsConnection)

st.title("🌿 Plant Data to Google Sheets")

# --- Initialize Memory ---
if 'total_area' not in st.session_state:
    st.session_state.total_area = 0.0
if 'leaf_count' not in st.session_state:
    st.session_state.leaf_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []

plant_name = st.text_input("Enter Plant Name/ID:", placeholder="e.g. Tomato_Plant_01")

# --- Image Processing ---
uploaded_file = st.file_uploader("Capture leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Step 1: Find the RED Border (Same as before) ---
    lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv_full, lower_red1, upper_red1) + cv2.inRange(hsv_full, lower_red2, upper_red2)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    target_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            target_contour = approx
            break

    if target_contour is not None:
        # --- Step 2: Perspective Warp ---
        pts = target_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        dst = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (400, 300))
        
        st.subheader("Fine-Tune Leaf Detection")
        
        # --- NEW: Hue Adjustment Sliders ---
        # Green is usually 35-85, Yellow is 20-35.
        hue_range = st.slider("Select Hue Range (Yellow to Green)", 0, 180, (20, 90))
        sat_min = st.slider("Minimum Saturation (Filters Grey/Perspex Glare)", 0, 255, 40)

        # --- NEW: Background Deletion Logic ---
        hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([hue_range[0], sat_min, 30])
        upper_bound = np.array([hue_range[1], 255, 255])
        
        mask = cv2.inRange(hsv_warped, lower_bound, upper_bound)
        # Clean up noise and reflections
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        col_img, col_mask = st.columns(2)
        
        # Apply mask to image to "delete background"
        result_img = cv2.bitwise_and(warped, warped, mask=mask)
        
        col_img.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Original Crop")
        col_mask.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Background Removed")

        # --- NEW: Calculate Button ---
        if st.button("📊 Calculate Leaf Area"):
            leaf_pixels = cv2.countNonZero(mask)
            # Filter out tiny specks (shadows/dirt)
            if leaf_pixels < 500:
                st.warning("Detection area too small. Adjust sliders.")
            else:
                current_area = (leaf_pixels / (400 * 300)) * 12.0
                st.session_state.current_calc = current_area
                st.success(f"Calculated Area: {current_area:.3f} sq in")

        # --- Save Data ---
        if 'current_calc' in st.session_state:
            if st.button("✅ Add this Calculation to History", use_container_width=True):
                st.session_state.total_area += st.session_state.current_calc
                st.session_state.leaf_count += 1
                st.session_state.history.append(round(st.session_state.current_calc, 3))
                del st.session_state.current_calc
                st.rerun()
    else:
        st.error("⚠️ Red border not detected!")

# --- Metrics and Upload (Remaining code same as yours) ---
st.divider()
col1, col2 = st.columns(2)
col1.metric("Leaves Found", st.session_state.leaf_count)
col2.metric("Total Area", f"{st.session_state.total_area:.2f} sq in")

# ... [Keep your existing Google Sheets upload and Reset button code here] ...
