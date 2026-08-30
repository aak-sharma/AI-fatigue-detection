from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import os
import logging
import mediapipe as mp
from scipy.spatial import distance
from collections import deque
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DriveSafe AI | Fatigue Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL UI THEME
# ============================================================================
st.markdown("""
<style>
/* ===== HIGH-CONTRAST LIGHT THEME ===== */
.stApp {
    background: #ffffff !important;
    color: #111827 !important;
}

.block-container {
    max-width: 1500px;
    padding: 1.2rem 2rem 3rem 2rem;
}

/* Global text */
html, body, [class*="css"], .stApp,
.stMarkdown, .stText, .stCaption, p, span, div, label {
    color: #111827;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-weight: 750 !important;
}

/* Hero */
.hero {
    background: #111827 !important;
    padding: 30px 34px;
    border-radius: 18px;
    margin-bottom: 24px;
    border: 1px solid #111827;
    box-shadow: 0 8px 24px rgba(17,24,39,.12);
}
.hero-title {
    color: #ffffff !important;
    font-size: 2.15rem;
    font-weight: 800;
    margin: 0;
}
.hero-subtitle {
    color: #e5e7eb !important;
    margin: 9px 0 0;
    font-size: 1rem;
}
.status-pill {
    display: inline-block;
    margin-top: 16px;
    padding: 7px 13px;
    border-radius: 999px;
    background: #374151 !important;
    color: #ffffff !important;
    font-size: .82rem;
    font-weight: 650;
}

/* Cards */
.card {
    background: #ffffff !important;
    border: 1px solid #d1d5db;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(17,24,39,.06);
    margin-bottom: 14px;
}
.card-title { color: #111827 !important; font-weight: 750; }
.card-subtitle { color: #4b5563 !important; font-size: .88rem; }

/* Metrics */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 12px;
    padding: 13px 15px;
}
div[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
}
div[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 800 !important;
}
div[data-testid="stMetricDelta"] {
    color: #374151 !important;
}

/* Buttons */

.control-label {
    color: #111827 !important;
    font-size: 1.05rem;
    font-weight: 750;
    margin: 18px 0 10px;
}

/* ===== BUTTONS: FORCE HIGH CONTRAST ===== */
.stButton {
    width: 100%;
}
.stButton > button {
    width: 100% !important;
    min-height: 48px !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 750 !important;
    line-height: 1.2 !important;
    color: #111827 !important;
    background: #ffffff !important;
    border: 2px solid #9ca3af !important;
    opacity: 1 !important;
    box-shadow: none !important;
}

/* Streamlit wraps button labels in paragraph/span elements */
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #111827 !important;
    font-weight: 750 !important;
    opacity: 1 !important;
}

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: #111827 !important;
    color: #ffffff !important;
    border: 2px solid #111827 !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div,
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[data-testid="baseButton-primary"] span,
.stButton > button[data-testid="baseButton-primary"] div {
    color: #ffffff !important;
}

/* Hover/focus states */
.stButton > button:hover,
.stButton > button:focus {
    opacity: 1 !important;
    transform: translateY(-1px);
}
.stButton > button:disabled {
    opacity: 0.55 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #374151 !important;
    font-weight: 700 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #111827 !important;
}

/* Inputs, sliders, checkboxes */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #9ca3af !important;
}
input, textarea {
    color: #111827 !important;
    background: #ffffff !important;
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #111827 !important;
}
.stSlider label, .stCheckbox label {
    color: #111827 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f3f4f6 !important;
    border-right: 1px solid #d1d5db;
}
section[data-testid="stSidebar"] *,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #111827 !important;
}


/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton {
    width: 100%;
}
.stDownloadButton > button {
    width: 100% !important;
    min-height: 48px !important;
    padding: 10px 18px !important;
    border-radius: 10px !important;
    background: #111827 !important;
    color: #ffffff !important;
    border: 2px solid #111827 !important;
    font-size: 0.95rem !important;
    font-weight: 750 !important;
    opacity: 1 !important;
}
.stDownloadButton > button p,
.stDownloadButton > button span,
.stDownloadButton > button div {
    color: #ffffff !important;
    font-weight: 750 !important;
}
.stDownloadButton > button:hover {
    background: #374151 !important;
    border-color: #374151 !important;
}

/* Alerts */
.stAlert p, .stAlert span {
    color: #111827 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">👁️ DriveSafe AI</div>
    <div class="hero-subtitle">Real-time driver fatigue & eye-closure detection</div>
    <div class="status-pill">● Computer Vision • MediaPipe Face Mesh • OpenCV</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## DriveSafe AI")
    st.caption("Fatigue Detection System")
    st.markdown("---")
    st.markdown("### System")
    st.markdown("**Detection:** Multi-metric eye analysis")
    st.markdown("**Primary:** Eye openness")
    st.markdown("**Secondary:** Eyelid proximity")
    st.markdown("**Tertiary:** EAR")
    st.markdown("---")
    st.caption("For research / demonstration use")


# ============================================================================
# LOAD MEDIAPIPE FACE MESH
# ============================================================================

@st.cache_resource
def load_face_mesh():
    """Load MediaPipe Face Mesh for accurate eye detection"""
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.85,  # Higher confidence
            min_tracking_confidence=0.85
        )
        logger.info("✅ MediaPipe Face Mesh loaded")
        return face_mesh, mp_face_mesh
    except Exception as e:
        logger.error(f"❌ Error loading Face Mesh: {e}")
        return None, None


face_mesh, mp_face_mesh = load_face_mesh()

if face_mesh is None:
    st.error("❌ Failed to load MediaPipe Face Mesh")
    st.error("Try: `pip install mediapipe`")
    st.stop()

# ============================================================================
# LANDMARK INDICES
# ============================================================================

LEFT_EYE_PRECISE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_PRECISE = [33, 160, 158, 133, 153, 144]

LEFT_EYE_FULL = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_FULL = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ============================================================================
# IMPROVED EYE DETECTION FUNCTIONS
# ============================================================================

def calculate_eye_openness_robust(eye_landmarks):
    """
    Calculate eye openness using multiple methods
    - Primary: Convex hull area (most reliable for closed eyes)
    - Secondary: Vertical distance
    Returns normalized 0-100 score
    """
    if len(eye_landmarks) < 4:
        return 0
    
    try:
        points = np.array(eye_landmarks, dtype=np.int32)
        
        # Method 1: Convex hull area (primary)
        hull = cv2.convexHull(points)
        area = cv2.contourArea(hull)
        
        # Calculate bounding box for normalization
        x, y, w, h = cv2.boundingRect(hull)
        max_possible_area = w * h
        
        # Normalized area ratio (0-1)
        if max_possible_area > 0:
            area_ratio = area / max_possible_area
        else:
            area_ratio = 0
        
        # Method 2: Vertical spread (secondary validation)
        y_coords = eye_landmarks[:, 1]
        y_spread = np.max(y_coords) - np.min(y_coords)
        
        # Empirical calibration:
        # Fully open: area_ratio ~0.7-0.8, y_spread ~30-40 pixels
        # Closed: area_ratio ~0.2-0.3, y_spread ~5-10 pixels
        
        # Combine metrics
        openness_from_area = min(100, max(0, (area_ratio - 0.15) * 200))  # 15-65% area = 0-100%
        openness_from_spread = min(100, max(0, y_spread * 2.5))  # 0-40px = 0-100%
        
        # Weight them
        openness = (openness_from_area * 0.7) + (openness_from_spread * 0.3)
        
        return max(0, min(100, openness))
    
    except Exception as e:
        logger.error(f"Error calculating eye openness: {e}")
        return 0


def calculate_improved_ear(eye_landmarks):
    """Improved EAR calculation with better handling of closed eyes"""
    if len(eye_landmarks) < 6:
        return 0
    
    try:
        landmarks = eye_landmarks[:6]
        
        # Vertical distances
        v1 = distance.euclidean(landmarks[1], landmarks[4])
        v2 = distance.euclidean(landmarks[2], landmarks[5])
        
        # Horizontal distance
        h = distance.euclidean(landmarks[0], landmarks[3])
        
        if h == 0:
            return 0
        
        # Standard EAR formula
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    except:
        return 0


def calculate_eye_closure_score(eye_openness, threshold_open=50):
    """
    Convert eye openness to closure score (0-100)
    0-100: 0 = fully open, 100 = fully closed
    """
    # Inverse of openness
    closure_score = max(0, min(100, 100 - eye_openness))
    return closure_score


def detect_eyelid_proximity(eye_landmarks):
    """
    Detect eyelid proximity (how close top and bottom lids are)
    Returns 0-100 where 100 = completely closed
    """
    if len(eye_landmarks) < 6:
        return 0
    
    try:
        # Top landmarks (indices 1, 2)
        top_lid = np.mean(eye_landmarks[[1, 2]], axis=0)
        
        # Bottom landmarks (indices 4, 5)
        bottom_lid = np.mean(eye_landmarks[[4, 5]], axis=0)
        
        # Distance between lids
        lid_distance = distance.euclidean(top_lid, bottom_lid)
        
        # Empirical: open eyes = 20-30px gap, closed = 0-5px
        # Map to 0-100
        proximity = max(0, min(100, 100 - (lid_distance * 4)))
        return proximity
    
    except:
        return 0


def preprocess_frame(frame):
    """Enhanced preprocessing for better landmark detection"""
    # Apply bilateral filter for noise reduction while preserving edges
    frame = cv2.bilateralFilter(frame, 9, 75, 75)
    
    # Apply CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_frame


def detect_eye_closure_ultra_accurate(results, frame_width, frame_height, baseline_open=None):
    """Calibration-aware multi-signal eye-closure detector."""
    if not results.multi_face_landmarks:
        return False, False, False, 0, 0, 0, 0

    try:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = np.array(
            [[int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)]
             for idx in LEFT_EYE_PRECISE], dtype=np.float32
        )
        right_eye = np.array(
            [[int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)]
             for idx in RIGHT_EYE_PRECISE], dtype=np.float32
        )

        left_openness = calculate_eye_openness_robust(left_eye)
        right_openness = calculate_eye_openness_robust(right_eye)
        avg_openness = (left_openness + right_openness) / 2.0

        left_ear = calculate_improved_ear(left_eye)
        right_ear = calculate_improved_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        left_proximity = detect_eyelid_proximity(left_eye)
        right_proximity = detect_eyelid_proximity(right_eye)
        avg_proximity = (left_proximity + right_proximity) / 2.0

        closure_score = calculate_eye_closure_score(avg_openness)

        # Personalized calibration:
        # baseline_open = (median_open_openness, median_closed_openness)
        if isinstance(baseline_open, (tuple, list)) and len(baseline_open) == 2:
            open_base = float(baseline_open[0])
            closed_base = float(baseline_open[1])
            separation = max(open_base - closed_base, 5.0)

            calibrated_openness = np.clip(
                100.0 * (avg_openness - closed_base) / separation,
                0.0, 100.0
            )

            # 42% is deliberately tolerant of natural eye variation.
            openness_closed = calibrated_openness < 42.0
            eyes_closing = calibrated_openness < 62.0
        else:
            threshold = float(baseline_open) if baseline_open is not None else 20.0
            openness_closed = avg_openness < threshold
            eyes_closing = avg_openness < threshold + 15.0

        # Secondary evidence.
        ear_closed = avg_ear < 0.18
        proximity_closed = avg_proximity > 52.0

        # Primary calibrated signal OR strong agreement from two secondary signals.
        # Require calibrated evidence plus at least one independent signal.
        # A very strong calibrated closure can stand alone; borderline closures
        # must agree with EAR or eyelid proximity.
        strong_calibrated_closure = (
            isinstance(baseline_open, (tuple, list))
            and len(baseline_open) == 2
            and 'calibrated_openness' in locals()
            and calibrated_openness < 25.0
        )

        eyes_closed = (
            strong_calibrated_closure
            or (openness_closed and (ear_closed or proximity_closed))
            or (ear_closed and proximity_closed)
        )

        return (
            bool(eyes_closed),
            bool(eyes_closing),
            True,
            float(avg_openness),
            float(closure_score),
            float(avg_proximity),
            float(avg_ear)
        )

    except Exception as e:
        logger.error(f"Error in eye closure detection: {e}")
        return False, False, False, 0, 0, 0, 0

def draw_eye_landmarks_enhanced(frame, results, frame_width, frame_height, eyes_state):
    """Draw eye landmarks with enhanced visualization"""
    if not results.multi_face_landmarks:
        return frame
    
    try:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Color based on state
        if eyes_state == "closed":
            color = (0, 0, 255)  # Red
        elif eyes_state == "closing":
            color = (0, 165, 255)  # Orange
        else:
            color = (0, 255, 0)  # Green
        
        # Draw eyes
        for idx in LEFT_EYE_PRECISE:
            x = int(landmarks[idx].x * frame_width)
            y = int(landmarks[idx].y * frame_height)
            cv2.circle(frame, (x, y), 2, color, -1)
        
        for idx in RIGHT_EYE_PRECISE:
            x = int(landmarks[idx].x * frame_width)
            y = int(landmarks[idx].y * frame_height)
            cv2.circle(frame, (x, y), 2, color, -1)
    
    except Exception as e:
        logger.error(f"Error drawing landmarks: {e}")
    
    return frame


# ============================================================================
# CALIBRATION WITH CLOSED EYES BASELINE
# ============================================================================

def run_comprehensive_calibration(face_mesh, cap):
    """
    Comprehensive calibration: measure baseline for both open and closed eyes
    """
    st.info("📐 **Comprehensive Calibration** - Two phases")
    
    # PHASE 1: Eyes OPEN
    st.info("Phase 1️⃣: Keep eyes WIDE OPEN for 3 seconds")
    progress_bar = st.progress(0)
    placeholder = st.empty()
    
    open_eyes_metrics = {"openness": [], "proximity": []}
    start_time = time.time()
    
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam")
            return None, None
        
        frame = preprocess_frame(frame)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye = []
            right_eye = []
            
            for idx in LEFT_EYE_PRECISE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_eye.append([x, y])
            
            for idx in RIGHT_EYE_PRECISE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_eye.append([x, y])
            
            left_eye = np.array(left_eye, dtype=np.float32)
            right_eye = np.array(right_eye, dtype=np.float32)
            
            left_open = calculate_eye_openness_robust(left_eye)
            right_open = calculate_eye_openness_robust(right_eye)
            avg_open = (left_open + right_open) / 2
            
            left_prox = detect_eyelid_proximity(left_eye)
            right_prox = detect_eyelid_proximity(right_eye)
            avg_prox = (left_prox + right_prox) / 2
            
            open_eyes_metrics["openness"].append(avg_open)
            open_eyes_metrics["proximity"].append(avg_prox)
        
        elapsed = time.time() - start_time
        progress = elapsed / 3
        progress_bar.progress(min(progress, 0.99))
        
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(display_frame)
    
    progress_bar.progress(1.0)
    placeholder.empty()
    
    if not open_eyes_metrics["openness"]:
        st.error("Could not detect eyes in open phase")
        return None, None
    
    baseline_open_openness = np.mean(open_eyes_metrics["openness"])
    baseline_open_proximity = np.mean(open_eyes_metrics["proximity"])
    
    st.success(f"✅ Phase 1 Complete - Baseline Openness: {baseline_open_openness:.1f}%")
    
    # PHASE 2: Eyes CLOSED
    st.info("Phase 2️⃣: SLOWLY CLOSE your eyes for 3 seconds")
    progress_bar = st.progress(0)
    placeholder = st.empty()
    
    closed_eyes_metrics = {"openness": [], "proximity": []}
    start_time = time.time()
    
    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam")
            return None, None
        
        frame = preprocess_frame(frame)
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye = []
            right_eye = []
            
            for idx in LEFT_EYE_PRECISE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                left_eye.append([x, y])
            
            for idx in RIGHT_EYE_PRECISE:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                right_eye.append([x, y])
            
            left_eye = np.array(left_eye, dtype=np.float32)
            right_eye = np.array(right_eye, dtype=np.float32)
            
            left_open = calculate_eye_openness_robust(left_eye)
            right_open = calculate_eye_openness_robust(right_eye)
            avg_open = (left_open + right_open) / 2
            
            left_prox = detect_eyelid_proximity(left_eye)
            right_prox = detect_eyelid_proximity(right_eye)
            avg_prox = (left_prox + right_prox) / 2
            
            closed_eyes_metrics["openness"].append(avg_open)
            closed_eyes_metrics["proximity"].append(avg_prox)
        
        elapsed = time.time() - start_time
        progress = elapsed / 3
        progress_bar.progress(min(progress, 0.99))
        
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(display_frame)
    
    progress_bar.progress(1.0)
    placeholder.empty()
    
    if not closed_eyes_metrics["openness"]:
        st.error("Could not detect eyes in closed phase")
        return None, None
    
    baseline_closed_openness = np.mean(closed_eyes_metrics["openness"])
    baseline_closed_proximity = np.mean(closed_eyes_metrics["proximity"])
    
    st.success(f"✅ Phase 2 Complete - Baseline Openness: {baseline_closed_openness:.1f}%")
    
    # Calculate thresholds
    openness_threshold = float(np.clip((np.percentile(open_eyes_metrics['openness'], 10) + np.percentile(closed_eyes_metrics['openness'], 90)) / 2, 10, 55))
    
    st.info(f"""
    📊 **Calibration Summary:**
    - Open Eyes Openness: {baseline_open_openness:.1f}%
    - Closed Eyes Openness: {baseline_closed_openness:.1f}%
    - **Alert Threshold: {openness_threshold:.1f}%**
    """)
    
    return openness_threshold, (baseline_open_openness, baseline_closed_openness)


# ============================================================================
# STREAMLIT UI
# ============================================================================

live_tab, evaluation_tab, about_tab = st.tabs([
    "🎥 Live Detection",
    "📈 Model Evaluation",
    "ℹ️ About"
])

with live_tab:
    st.markdown("### Live monitoring")
    st.caption("Use calibration before starting detection for a personalized eye-openness baseline.")

    col1, col2 = st.columns([2.6, 1], gap="large")

    with col1:
        st.markdown('<div class="card"><div class="card-title">Live Video Feed</div><div class="card-subtitle">Camera-based eye state monitoring</div></div>', unsafe_allow_html=True)
        video_placeholder = st.empty()

    with col2:
        st.markdown('<div class="card"><div class="card-title">Real-time Analysis</div><div class="card-subtitle">Current detection signals</div></div>', unsafe_allow_html=True)
        openness_placeholder = st.empty()
        closure_placeholder = st.empty()
        proximity_placeholder = st.empty()
        state_placeholder = st.empty()
        alert_placeholder = st.empty()

    st.markdown("#### Controls")
    button_col1, button_col2, button_col3 = st.columns(3)

    with button_col1:
        start_button = st.button("▶  START RECORDING", type="primary")

    with button_col2:
        stop_button = st.button("■  STOP RECORDING")

    with button_col3:
        calibrate_button = st.button("◉  CALIBRATE")

    st.markdown("#### Detection settings")
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        openness_threshold = st.slider("Alert Threshold — Eye Openness", 10, 40, 20, 1)

    with col_p2:
        alert_delay = st.slider("Alert Delay — Frames", 3, 15, 5, 1)

    with col_p3:
        show_metrics = st.checkbox("Show diagnostic metrics", value=True)

with evaluation_tab:
    st.markdown("### Performance evaluation")
    st.caption("Measure the detector on labeled images that were not used during development.")

with about_tab:
    st.markdown("### About DriveSafe AI")
    st.write(
        "DriveSafe AI is a real-time computer-vision fatigue detection prototype. "
        "It analyzes facial landmarks and combines eye openness, eyelid proximity, "
        "and Eye Aspect Ratio (EAR) with temporal smoothing and calibration."
    )
    a, b, c = st.columns(3)
    a.metric("Primary Signal", "Eye Openness")
    b.metric("Secondary Signal", "Lid Proximity")
    c.metric("Tertiary Signal", "EAR")
    st.info("Tip: Good lighting and a frontal camera position improve landmark detection.")


# ============================================================================
# ============================================================================
# Runtime dependency checks
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )
except ImportError:
    accuracy_score = precision_score = recall_score = f1_score = confusion_matrix = None

# AUTOMATIC WEBCAM EVALUATION
# ============================================================================

def run_automatic_webcam_evaluation(face_mesh, phase_seconds=20):
    """Run a calibrated guided webcam evaluation and save raw results."""
    from datetime import datetime

    if pd is None:
        raise RuntimeError("pandas is not installed. Run: python -m pip install pandas")
    if accuracy_score is None:
        raise RuntimeError("scikit-learn is not installed. Run: python -m pip install scikit-learn")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam. Check camera permissions.")

    session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []

    try:
        # -------------------- CALIBRATION --------------------
        calibration = [(0, "Keep your eyes OPEN naturally."),
                       (1, "Close your eyes completely.")]

        calibration_values = {0: [], 1: []}

        for label, instruction in calibration:
            st.markdown("### Calibration")
            st.info(instruction)
            progress = st.progress(0)
            timer = st.empty()
            start = time.time()

            while time.time() - start < 5:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = preprocess_frame(frame)
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    le = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in LEFT_EYE_PRECISE], dtype=np.float32)
                    re = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in RIGHT_EYE_PRECISE], dtype=np.float32)
                    openness = (
                        calculate_eye_openness_robust(le) +
                        calculate_eye_openness_robust(re)
                    ) / 2.0
                    calibration_values[label].append(float(openness))

                elapsed = time.time() - start
                progress.progress(min(1.0, elapsed / 5.0))
                timer.markdown(f"**{max(0, 5-elapsed):.1f}s remaining**")

            progress.empty()
            timer.empty()

        if len(calibration_values[0]) < 10 or len(calibration_values[1]) < 10:
            raise RuntimeError("Calibration failed. Keep your full face visible.")

        open_base = float(np.median(calibration_values[0]))
        closed_base = float(np.median(calibration_values[1]))

        if open_base <= closed_base + 3:
            raise RuntimeError(
                "Open and closed eye measurements are too similar. "
                "Try better lighting and face the camera directly."
            )

        st.success(
            f"Calibration complete • Open {open_base:.1f}% • "
            f"Closed {closed_base:.1f}%"
        )

        # -------------------- EVALUATION --------------------
        phases = [
            ("Alert — Eyes Open", "Keep your eyes naturally OPEN.", 0),
            ("Fatigue — Eyes Closed", "Close your eyes and keep them CLOSED.", 1),
            ("Alert — Normal Blinking", "Open your eyes and blink naturally.", 0),
        ]

        for phase_index, (name, instruction, actual_label) in enumerate(phases):
            st.markdown(f"### Phase {phase_index + 1} of 3 — {name}")
            st.info(instruction)

            progress = st.progress(0)
            timer = st.empty()
            preview = st.empty()
            phase_start = time.time()
            consecutive_closed = 0
            prediction_history = deque(maxlen=5)

            while time.time() - phase_start < phase_seconds:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = preprocess_frame(frame)
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                (
                    eyes_closed, eyes_closing, face_detected,
                    avg_openness, closure_score, avg_proximity, avg_ear
                ) = detect_eye_closure_ultra_accurate(
                    results, w, h, (open_base, closed_base)
                )

                if face_detected:
                    if eyes_closed:
                        consecutive_closed += 1
                    else:
                        consecutive_closed = max(0, consecutive_closed - 2)

                    # Temporal majority vote reduces false fatigue detections
                    # caused by a single noisy landmark frame or normal blink.
                    prediction_history.append(1 if eyes_closed else 0)
                    predicted_label = int(
                        sum(prediction_history) >=
                        max(2, int(np.ceil(len(prediction_history) * 0.6)))
                    )
                    system_alert = int(consecutive_closed >= 5)
                else:
                    consecutive_closed = 0
                    prediction_history.clear()
                    predicted_label = 0
                    system_alert = 0

                rows.append({
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "phase": name,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "system_alert": system_alert,
                    "face_detected": int(face_detected),
                    "eye_openness": round(float(avg_openness), 3),
                    "closure_score": round(float(closure_score), 3),
                    "lid_proximity": round(float(avg_proximity), 3),
                    "ear": round(float(avg_ear), 5),
                    "consecutive_closed_frames": consecutive_closed,
                })

                preview.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    use_column_width=True
                )

                elapsed = time.time() - phase_start
                progress.progress(min(1.0, elapsed / phase_seconds))
                timer.markdown(f"**Time remaining: {max(0, phase_seconds-elapsed):.1f}s**")

            progress.empty()
            timer.empty()
            preview.empty()

        df = pd.DataFrame(rows)
        valid = df[df["face_detected"] == 1].copy()

        if valid.empty:
            raise RuntimeError("No valid samples with a detected face.")

        y_true = valid["actual_label"].astype(int)
        y_pred = valid["predicted_label"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "valid": len(valid),
            "skipped": len(df) - len(valid),
            "cm": cm,
        }

        output_dir = Path("evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"evaluation_{session_start}.csv"
        df.to_csv(csv_path, index=False)

        return metrics, df, csv_path

    finally:
        cap.release()


with evaluation_tab:
    st.markdown("### Automatic webcam evaluation")
    st.caption(
        "The system guides you through three labeled phases and automatically "
        "records the detector's measurements and predictions."
    )

    st.markdown("""
    <div class="card">
        <div class="card-title">How this test works</div>
        <div class="card-subtitle">
            1. Keep eyes open → 2. Close eyes → 3. Blink normally.
            Each phase provides the ground-truth label automatically.
        </div>
    </div>
    """, unsafe_allow_html=True)

    duration = st.slider(
        "Duration per phase (seconds)",
        min_value=10,
        max_value=60,
        value=20,
        step=5
    )

    st.warning(
        "For a meaningful result, test with a person who was not used while "
        "tuning the thresholds. Keep the camera position and lighting reasonably consistent."
    )

    if st.button("▶  START AUTOMATIC EVALUATION", type="primary"):
        try:
            with st.spinner("Preparing webcam evaluation..."):
                eval_results, eval_df, eval_csv = run_automatic_webcam_evaluation(
                    face_mesh, duration
                )

            st.success("Evaluation completed and raw results were saved.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{eval_results['accuracy'] * 100:.2f}%")
            m2.metric("Precision", f"{eval_results['precision'] * 100:.2f}%")
            m3.metric("Recall", f"{eval_results['recall'] * 100:.2f}%")
            m4.metric("F1 Score", f"{eval_results['f1'] * 100:.2f}%")

            tn, fp, fn, tp = eval_results["cm"].ravel()
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            st.markdown("#### Additional safety metric")
            s1, s2, s3 = st.columns(3)
            s1.metric("Specificity", f"{specificity * 100:.2f}%")
            s2.metric("False Positives", int(fp))
            s3.metric("False Negatives", int(fn))

            st.markdown("#### Confusion Matrix")
            st.dataframe(
                {
                    "": ["Actual Alert", "Actual Fatigued"],
                    "Predicted Alert": [tn, fn],
                    "Predicted Fatigued": [fp, tp],
                },
                hide_index=True
            )

            st.markdown("#### Evaluation summary")
            s1, s2 = st.columns(2)
            s1.metric("Valid samples", eval_results["valid"])
            s2.metric("Skipped / no-face samples", eval_results["skipped"])

            st.download_button(
                "⬇  DOWNLOAD EVALUATION CSV",
                data=eval_df.to_csv(index=False).encode("utf-8"),
                file_name=eval_csv.name,
                mime="text/csv"
            )

            st.caption(
                f"Raw session data is also saved automatically to: {eval_csv}"
            )

            st.info(
                "Use these measured values on your resume only after running the "
                "test on an appropriate unseen evaluation set. These metrics are "
                "for the guided webcam session, not a universal accuracy guarantee."
            )

        except Exception as e:
            st.error(f"Evaluation failed: {e}")

# Session state
if 'recording' not in st.session_state:
    st.session_state.recording = False

if 'calibrated' not in st.session_state:
    st.session_state.calibrated = False

if 'calibration_baselines' not in st.session_state:
    st.session_state.calibration_baselines = None

if 'calibration_threshold' not in st.session_state:
    st.session_state.calibration_threshold = None

if start_button:
    st.session_state.recording = True

if stop_button:
    st.session_state.recording = False

if calibrate_button:
    st.session_state.recording = False
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            threshold, baselines = run_comprehensive_calibration(face_mesh, cap)
            cap.release()
            
            if threshold:
                st.session_state.calibration_threshold = threshold
                st.session_state.calibration_baselines = baselines
                st.session_state.calibrated = True
    except Exception as e:
        st.error(f"Calibration error: {e}")

# ============================================================================
# VIDEO STREAMING - ULTRA ACCURATE
# ============================================================================

if st.session_state.recording:
    st.warning("● **DETECTION ACTIVE** — Monitoring eye closure")
    
    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            st.session_state.recording = False
        else:
            consecutive_closed = 0
            frame_count = 0
            
            openness_history = deque(maxlen=15)
            closure_history = deque(maxlen=15)
            
            # Use calibrated threshold or slider
            threshold = st.session_state.calibration_threshold if st.session_state.calibrated else openness_threshold
            
            if st.session_state.calibrated:
                st.success(f"✅ Using calibrated threshold: {threshold:.1f}%")
            
            while st.session_state.recording:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                
                # Preprocess
                frame = preprocess_frame(frame)
                frame = cv2.flip(frame, 1)
                
                h, w, c = frame.shape
                
                # Process
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                # Detect
                eyes_closed, eyes_closing, face_detected, avg_openness, closure_score, avg_proximity, avg_ear = \
                    detect_eye_closure_ultra_accurate(
                        results, w, h,
                        st.session_state.calibration_baselines
                        if st.session_state.calibrated
                        else threshold
                    )
                
                # Add to history
                openness_history.append(avg_openness)
                closure_history.append(closure_score)
                
                # Smooth
                if len(openness_history) > 3:
                    smoothed_openness = np.mean(list(openness_history)[-3:])
                    smoothed_closure = np.mean(list(closure_history)[-3:])
                else:
                    smoothed_openness = avg_openness
                    smoothed_closure = closure_score
                
                # Use the calibrated multi-signal detector decision.
                final_eyes_closed = bool(eyes_closed)
                final_eyes_closing = bool(eyes_closing)

                # Temporal confirmation reduces false positives from blinks.
                if final_eyes_closed and face_detected:
                    consecutive_closed += 1
                else:
                    consecutive_closed = max(0, consecutive_closed - 2)

                alert_triggered = consecutive_closed >= alert_delay
                
                # Determine visual state
                if not face_detected:
                    eyes_state = "no_face"
                    status_text = "NO FACE DETECTED"
                    color = (0, 0, 255)
                elif alert_triggered:
                    eyes_state = "closed"
                    status_text = "🚨 EYES CLOSED!"
                    color = (0, 0, 255)
                elif final_eyes_closed:
                    eyes_state = "closed"
                    status_text = "❌ EYES CLOSED"
                    color = (0, 0, 255)
                elif final_eyes_closing:
                    eyes_state = "closing"
                    status_text = "⚠️ Eyes Closing"
                    color = (0, 165, 255)
                else:
                    eyes_state = "open"
                    status_text = "✅ Eyes Open"
                    color = (0, 255, 0)
                
                # Draw
                frame = draw_eye_landmarks_enhanced(frame, results, w, h, eyes_state)
                
                # Info box
                cv2.rectangle(frame, (10, 10), (600, 130), color, 2)
                cv2.putText(frame, status_text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.putText(frame, f"Openness: {smoothed_openness:.1f}% | Threshold: {threshold:.1f}%", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Consecutive Frames: {consecutive_closed}/{alert_delay+1}", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Big alert
                if alert_triggered:
                    cv2.rectangle(frame, (10, 10), (w-10, 140), (0, 0, 255), 5)
                    cv2.putText(frame, "EYES CLOSED ALERT!", (w//5, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                # Display
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_frame)
                
                # Stats
                with openness_placeholder.container():
                    st.metric("Eye Openness", f"{smoothed_openness:.1f}%", 
                             delta=f"Threshold: {threshold:.1f}%")
                
                with closure_placeholder.container():
                    st.metric("Closure Score", f"{smoothed_closure:.1f}/100",
                             delta="0=open, 100=closed")
                
                with proximity_placeholder.container():
                    st.metric("Lid Proximity", f"{avg_proximity:.1f}",
                             delta="0=open, 100=closed")
                
                with state_placeholder.container():
                    if final_eyes_closed:
                        st.error(f"❌ **CLOSED** (Frame: {consecutive_closed})")
                    elif final_eyes_closing:
                        st.warning(f"⚠️ **CLOSING**")
                    else:
                        st.success(f"✅ **OPEN**")
                
                if alert_triggered:
                    with alert_placeholder.container():
                        st.error("🚨 **ALERT: EYES CLOSED!**")
                
                frame_count += 1
                time.sleep(0.01)
            
            cap.release()
            st.session_state.recording = False
            st.info("✅ Recording stopped")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Video error: {e}", exc_info=True)
        st.session_state.recording = False

else:
    with video_placeholder.container():
        st.markdown("""
        <div class="card" style="text-align:center; padding:52px 24px; min-height:260px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:3rem;">👁️</div>
            <h2>Ready to monitor</h2>
            <p style="color:#6b7280;">
                Run calibration first, then start detection. The system will
                continuously analyze eye openness and closure patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("🚀 Ultra-Accurate Detection with Multi-Metric Validation")