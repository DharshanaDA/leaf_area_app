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
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Step 1: Find the RED Border ---
    # Red is unique because it's at both ends of the HSV spectrum (0-10 and 170-180)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv_full, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_full, lower_red2, upper_red2)
    red_mask = mask_red1 + mask_red2

    # Clean up the red mask
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Find the largest red shape
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    target_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True) # 3% tolerance
        if len(approx) == 4: # Looking for 4 corners
            target_contour = approx
            break

    # --- Step 2: Process if Board is Found ---
    if target_contour is not None:
        # Perspective Correction (Warp)
        pts = target_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1); rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
        
        dst = np.array([[0, 0], [400, 0], [400, 300], [0, 300]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (400, 300))

        # --- Step 3: Find the GREEN Leaf inside the warped board ---
        hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        # Broad green range for better detection
        lower_green = np.array([30, 40, 30]) 
        upper_green = np.array([95, 255, 255])
        leaf_mask = cv2.inRange(hsv_warped, lower_green, upper_green)
        
        # Clean leaf mask
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

        # Calculate Area
        leaf_pixels = cv2.countNonZero(leaf_mask)
        current_area = (leaf_pixels / (400 * 300)) * 12.0 # (pixels / total) * 3x4 board

        # Display Results
        st.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Detected Board Area")
        st.image(leaf_mask, caption="What the computer sees as 'Leaf'")
        st.metric("Detected Leaf Area", f"{current_area:.3f} sq in")

        if st.button("✅ Add Leaf to Plant", use_container_width=True):
            st.session_state.total_area += current_area
            st.session_state.leaf_count += 1
            st.session_state.history.append(round(current_area, 3))
            st.rerun()
    else:
        st.error("⚠️ Red border not detected!")
        st.info("Make sure the entire RED border is visible in the photo.")

st.divider()

# --- Upload to Google Sheets ---
if st.session_state.leaf_count > 0:
    if st.button("📤 Upload Plant Data to Google Sheets", type="primary", use_container_width=True):
        if not plant_name:
            st.error("Please enter a Plant Name before uploading!")
        else:
            try:
                # 1. Fetch existing data (so we don't overwrite everything)
                # Ensure the worksheet name matches your tab exactly (e.g., "Sheet1")
                existing_data = conn.read(worksheet="Sheet1")
                
                # 2. Prepare the new row


                new_row = pd.DataFrame([{
                    "Plant Name": plant_name,
                    "Leaf Count": st.session_state.leaf_count,
                    "Individual Areas": str(st.session_state.history),
                    "Total Area": round(st.session_state.total_area, 3)
                }])
                
                # 3. Combine them
                updated_df = pd.concat([existing_data, new_row], ignore_index=True)
                
                # 4. Use .update to push the new full list
                conn.update(worksheet="Sheet1", data=updated_df)
                
                st.success(f"Data for {plant_name} uploaded successfully!")
            except Exception as e:
                st.error(f"Upload failed: {e}")

# Reset
if st.button("🗑️ Reset Everything", use_container_width=True):
    st.session_state.total_area = 0.0
    st.session_state.leaf_count = 0
    st.session_state.history = []
    st.rerun()
