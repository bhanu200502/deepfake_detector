import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
from streamlit.components.v1 import html
import base64


IMG_SIZE = 128
st.set_page_config(page_title="Deepfake Detector Grad-CAM", layout="wide")
keras.backend.clear_session()
st.markdown("""
<style>
h1 {
    color: #ff4b4b;
    font-size: 48px;
    text-align: center;
}
.metric-box {
    background-color: #f0f0f0;
    padding: 20px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>Deepfake Detector</h1>', unsafe_allow_html=True)
try:
    model = keras.models.load_model("model.h5")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model.h5: {e}")
    st.stop()

def get_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    if not hasattr(model, "outputs") or len(model.outputs) == 0:
        dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
        _ = model(dummy)

    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)
    
    if last_conv_layer_name is None:
        st.error("No convolutional layer found in model")
        return None
        
    last_conv = model.get_layer(last_conv_layer_name)
    idx = model.layers.index(last_conv)
    conv_model = tf.keras.models.Model(model.inputs, last_conv.output)
    conv_shape = last_conv.output.shape[1:]
    classifier_input = tf.keras.Input(shape=conv_shape)
    x = classifier_input
    for layer in model.layers[idx + 1 :]:
        x = layer(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    if isinstance(heatmap, np.ndarray):
        return heatmap
    return heatmap.numpy()
def classify_confidence_level(prediction):
    """Classify prediction into confidence levels."""
    if prediction > 0.8:
        return "STRONG FAKE", "🔴", "#ff4b4b"
    elif prediction > 0.65:
        return "MODERATE FAKE", "🟠", "#ff9500"
    elif prediction > 0.5:
        return "LIGHT FAKE", "🟡", "#ffb700"
    elif prediction > 0.35:
        return "LIGHT REAL", "🟢", "#9ccc65"
    elif prediction > 0.2:
        return "MODERATE REAL", "🟢", "#4caf50"
    else:
        return "STRONG REAL", "🟢", "#00695c"
uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded Image")
    img_array = np.expand_dims(image_rgb.astype(np.float32) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else (1.0 - prediction)
    level, emoji, color = classify_confidence_level(prediction)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{prediction:.4f}")
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")
    with col3:
        st.markdown(f'<div style="text-align:center; font-size:24px; color:{color}; font-weight:bold;">{emoji} {level}</div>', unsafe_allow_html=True)
    alpha = 0.4  
    alpha = st.slider("Grad-CAM Transparency", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
    last_conv_layer_name = get_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(image_rgb, 1-alpha, heatmap_rgb, alpha, 0)
    _, buffer_orig = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    orig_base64 = base64.b64encode(buffer_orig).decode()
    _, buffer_cam = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    cam_base64 = base64.b64encode(buffer_cam).decode()

    html(f"""
    <div style="display:flex; justify-content:center; align-items:center; gap:30px; margin-top:20px;">
        <div>
            <h3 style="text-align:center;">Original Image</h3>
            <img src="data:image/png;base64,{orig_base64}" width="256"/>
        </div>
        <div>
            <h3 style="text-align:center;">Grad-CAM Overlay</h3>
            <img src="data:image/png;base64,{cam_base64}" width="256"/>
        </div>
    </div>
    """, height=350)