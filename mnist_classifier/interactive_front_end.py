import ssl, certifi
# Ensure HTTPS downloads use certifi's CA bundle
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from torchvision import transforms
from PIL import Image
from datetime import datetime

# Import the trained model class
from trainmodel_CNN import MNISTClassifier  # Use relative import if trainmodel.py is in the same directory
from db import log_prediction

# Load the trained model checkpoint once
@st.cache_resource
def load_model():
    model = MNISTClassifier()
    model.load_state_dict(torch.load("mnist_best.pth", map_location="cpu"))
    model.eval()
    return model

# Load the trained model checkpoint once
model = load_model()

# Preprocess canvas RGBA image data into a normalized MNIST tensor
def preprocess_canvas_image(img_data):
    # Use the RGB channels (ignore alpha) and convert to grayscale
    rgb = img_data[:, :, :3].astype('uint8')
    gray = Image.fromarray(rgb, mode='RGB').convert('L')
    # Resize to 28x28 using high-quality resampling
    gray = gray.resize((28, 28), Image.LANCZOS)
    # Transform to tensor and normalize to [-1,1]
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return tf(gray)

# Initialize history in session state
if 'records' not in st.session_state:
    st.session_state['records'] = []

# Initialize app stage
if 'stage' not in st.session_state:
    st.session_state['stage'] = 'initial'

# Initialize canvas counter for resetting
if 'canvas_counter' not in st.session_state:
    st.session_state['canvas_counter'] = 0

# Layout: title and expander
st.title("Digit Recognizer")
with st.expander("How it works", expanded=False):
    st.write(
        "1. Draw a digit on the left canvas.\n"
        "2. Click **Classify** to get a prediction.\n"
        "3. Click **Give Feedback** to label your draw.\n"
        "4. View history below."
    )

# Two columns for canvas/classify and results
col1, col2 = st.columns([2, 1])

with col1:
    canvas_key = f"canvas_{st.session_state['canvas_counter']}"
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key=canvas_key,
    )

    # Classify button, only active in 'initial' stage
    if st.session_state['stage'] == 'initial':
        if st.button("Classify"):
            if canvas_result.image_data is None:
                st.warning("Please draw a digit before classifying.")
            else:
                tensor = preprocess_canvas_image(canvas_result.image_data.astype('uint8'))
                with torch.no_grad():
                    logits = model(tensor.unsqueeze(0))
                    probs  = torch.softmax(logits, dim=1)
                    st.session_state['pred'] = probs.argmax(1).item()
                    st.session_state['conf'] = probs.max().item()
                st.session_state['stage'] = 'classified'
    else:
        st.button("Classify", disabled=True)
        # New prediction: clear canvas but keep existing results
        if st.button("New"):
            # Reset app state for new prediction
            st.session_state['stage'] = 'initial'
            # Move counter by 2 to force a fresh remount of the canvas
            st.session_state['canvas_counter'] += 2
            # Immediately rerun to clear the canvas
            try:
                st.experimental_rerun()
            except AttributeError:
                pass  # older Streamlit versions will rerun automatically

with col2:
    # Results display whenever a prediction exists
    if 'pred' in st.session_state:
        st.subheader("Results")
        st.write(f"**Prediction:** {st.session_state['pred']}")
        st.write(f"**Confidence:** {st.session_state['conf']*100:.2f}%")

# After classification, show 'Give Feedback' button
if st.session_state['stage'] == 'classified':
    if st.button("Give Feedback"):
        st.session_state['stage'] = 'feedback'

# Feedback stage: show true-label buttons and submit
if st.session_state['stage'] == 'feedback':
    st.write("\n---\nSelect the true label:")
    true_label = st.radio(
        label="True label (hidden)",
        options=list(range(10)),
        horizontal=True,
        key='true_label_feedback',
        label_visibility="hidden"
    )
    if st.button("Submit true label"):
        feedback = '✅' if true_label == st.session_state['pred'] else '❌'
        # Append to session history
        st.session_state['records'].append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Prediction': st.session_state['pred'],
            'True Label': true_label,
            'Feedback': feedback
        })
        # Log to PostgreSQL
        try:
            log_prediction(st.session_state['pred'], true_label)
        except Exception as e:
            st.error(f"Error logging to database: {e}")
        # Move to history stage
        st.session_state['stage'] = 'history'

# History table at the bottom
st.write("### History")
st.table(st.session_state['records'])
