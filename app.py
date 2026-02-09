# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Plant Leaf Tracker", layout="wide")

st.title("🌿 Total Plant Leaf Area Tracker")
st.write("Upload photos one by one. The app will keep a running total for the plant.")

# --- Initialize Session State ---
if 'total_area' not in st.session_state:
    st.session_state.total_area = 0.0
if 'leaf_count' not in st.session_state:
    st.session_state.leaf_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Sidebar for Reset ---
if st.sidebar.button("Reset Plant Data"):
    st.session_state.total_area = 0.0
    st.session_state.leaf_count = 0
    st.session_state.history = []
    st.rerun()

# --- Main Stats Display ---
col_stat1, col_stat2 = st.columns(2)
col_stat1.metric("Total Leaves", st.session_state.leaf_count)
col_stat2.metric("Total Area (sq in)", f"{st.session_state.total_area:.2f}")

# --- File Uploader ---
uploaded_file = st.file_uploader("Capture or Upload a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # --- Image Processing ---
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

    if target_contour is None:
        st.error("Error: Could not detect the wood piece. Please ensure it's fully visible.")
    else:
        # Perspective Transform
        pts = target_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]

        dst = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (400, 300))

        # Green Color Masking
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40]) 
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Calculation
        total_wood_pixels = 400 * 300
        leaf_pixels = cv2.countNonZero(mask)
        current_leaf_area = (leaf_pixels / total_wood_pixels) * 12.0

        # --- "Add to Total" Button ---
        # We use the filename as a simple way to prevent double-adding if the app reruns
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption=f"Detected Area: {current_leaf_area:.3f} sq in", width=300)
        
        if st.button("Add this leaf to Total"):
            st.session_state.total_area += current_leaf_area
            st.session_state.leaf_count += 1
            st.session_state.history.append(current_leaf_area)
            st.success(f"Added Leaf #{st.session_state.leaf_count}!")
            # Use st.rerun() to update the metrics at the top immediately
            st.rerun()

# --- History Table ---
if st.session_state.history:
    with st.expander("View Leaf History"):
        for i, area in enumerate(st.session_state.history):
            st.write(f"Leaf {i+1}: {area:.3f} sq in")
