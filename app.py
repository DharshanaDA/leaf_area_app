import streamlit as st
import cv2
import numpy as np
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from google.oauth2 import service_account

st.set_page_config(page_title="Leaf Tracker Pro", layout="centered")

# --- 1. Google Drive & Sheets Connection Setup ---
# --- 1. Google Drive & Sheets Connection Setup ---
conn = st.connection("gsheets", type=GSheetsConnection)

try:
    # Get credentials directly from the GSheets connection to avoid mismatch
    creds = conn._instance._credentials 
    drive_service = build('drive', 'v3', credentials=creds)
except Exception as e:
    # If the above fails, use your existing secrets method
    creds_info = st.secrets["connections"]["gsheets"]
    creds = service_account.Credentials.from_service_account_info(
        creds_info, 
        scopes=["https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/spreadsheets"]
    )
    drive_service = build('drive', 'v3', credentials=creds)

DRIVE_FOLDER_ID = "0AAkmovKASPt_Uk9PVA"  # <--- REPLACE THIS WITH YOUR FOLDER ID

def upload_to_drive(file_buffer, file_name):
    """Helper function to upload image buffer to Google Drive"""
    media = MediaIoBaseUpload(file_buffer, mimetype='image/jpeg')
    file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
    drive_service.files().create(body=file_metadata, media_body=media, fields='id', supportsAllDrives=True).execute()

st.title("🌿 Leaf Area Calculator")

# --- Initialize Memory ---
if 'total_area' not in st.session_state:
    st.session_state.total_area = 0.0
if 'leaf_count' not in st.session_state:
    st.session_state.leaf_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Plant Info and Photos ---
plant_name = st.text_input("Enter Plant Name/ID:", placeholder="e.g. Tomato_Plant_01")

st.subheader("📸 Plant Overview Photos")
col_side, col_nadir = st.columns(2)
with col_side:
    side_img = st.camera_input("Side View", key="side")
with col_nadir:
    nadir_img = st.camera_input("Nadir (Top) View", key="nadir")

st.divider()

# --- Image Processing (Leaf Area) ---
uploaded_file = st.file_uploader("Capture leaf for area calculation", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    detection_method = st.selectbox(
        "Select Border Detection Method:",
        ("Blue Corner Dots", "Red Border Line")
    )
    
    target_contour = None

    if detection_method == "Red Border Line":
        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv_full, lower_red1, upper_red1) + cv2.inRange(hsv_full, lower_red2, upper_red2)
        cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                target_contour = approx.reshape(4, 2)
                break

    elif detection_method == "Blue Corner Dots":
        lower_blue, upper_blue = np.array([100, 100, 100]), np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv_full, lower_blue, upper_blue)
        cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) >= 4:
            dot_centers = []
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    dot_centers.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            if len(dot_centers) == 4:
                target_contour = np.array(dot_centers, dtype="float32")

    if target_contour is not None:
        pts = target_contour
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        dst = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (300, 300))
        
        st.subheader("Fine-Tune Leaf Detection")
        hue_range = st.slider("Select Hue Range", 0, 180, (25, 90))
        sat_min = st.slider("Minimum Saturation", 0, 255, 40)

        hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_warped, np.array([hue_range[0], sat_min, 30]), np.array([hue_range[1], 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        col_img, col_mask = st.columns(2)
        col_img.image(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), caption="Warped Area")
        col_mask.image(mask, caption="Leaf Mask")

        if st.button("📊 Calculate Leaf Area"):
            leaf_pixels = cv2.countNonZero(mask)
            if leaf_pixels < 300:
                st.warning("Detection area too small.")
            else:
                current_area = (leaf_pixels / (300 * 300)) * 58.0644
                st.session_state.current_calc = current_area
                st.metric("Detected Leaf Area", f"{current_area:.3f} cm²")

        if 'current_calc' in st.session_state:
            if st.button("✅ Add to History", use_container_width=True):
                st.session_state.total_area += st.session_state.current_calc
                st.session_state.leaf_count += 1
                st.session_state.history.append(round(st.session_state.current_calc, 3))
                del st.session_state.current_calc
                st.rerun()
    else:
        st.error("⚠️ Markers not detected.")

# --- Metrics and Combined Upload ---
st.divider()
c1, c2 = st.columns(2)
c1.metric("Leaves Found", st.session_state.leaf_count)
c2.metric("Total Area", f"{st.session_state.total_area:.2f} cm²")

if st.session_state.leaf_count > 0:
    if st.button("📤 Upload Data & Photos", type="primary", use_container_width=True):
        if not plant_name:
            st.error("Please enter a Plant Name!")
        elif not side_img or not nadir_img:
            st.error("Please capture both Side and Nadir photos!")
        else:
            try:
                # 1. Upload Photos to Drive
                with st.spinner("Uploading photos to Drive..."):
                    upload_to_drive(io.BytesIO(side_img.getvalue()), f"{plant_name}_side.jpg")
                    upload_to_drive(io.BytesIO(nadir_img.getvalue()), f"{plant_name}_nadir.jpg")
                
                # 2. Upload Data to Sheets
                with st.spinner("Updating Google Sheet..."):
                    existing_data = conn.read(worksheet="Sheet1")
                    new_row = pd.DataFrame([{
                        "Plant Name": plant_name, 
                        "Leaf Count": st.session_state.leaf_count, 
                        "History": str(st.session_state.history), 
                        "Total Area": round(st.session_state.total_area, 3)
                    }])
                    conn.update(worksheet="Sheet1", data=pd.concat([existing_data, new_row], ignore_index=True))
                
                st.success(f"Success! Data and photos for {plant_name} saved.")
            except Exception as e:
                st.error(f"Upload Failed: {e}")

if st.button("🗑️ Reset All", use_container_width=True):
    st.session_state.total_area, st.session_state.leaf_count, st.session_state.history = 0.0, 0, []
    st.rerun()
