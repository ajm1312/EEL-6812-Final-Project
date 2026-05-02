import gradio as gr
import numpy as np
import tensorflow as tf
import joblib
import scipy.fftpack as fftpack
import cv2
import traceback
import sys

tf.compat.v1.disable_eager_execution()

honeypot_model = tf.keras.models.load_model('cifar_model.h5', compile=False)
prn = tf.keras.models.load_model('trained_prn_v2.h5', compile=False)
joint_model = tf.keras.models.load_model('trained_joint_model.h5', compile=False)
svm = joblib.load('svm_detector_trained_final.joblib')

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.keras.backend.get_session()

LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def compute_dct_features(image_tensor, keep_coeffs=8):
    batch_size, height, width, channels = image_tensor.shape
    features = []
    
    for batch in range(batch_size):
        if channels == 3:
            gray_diff = 0.299 * image_tensor[batch, :, :, 0] + 0.587 * image_tensor[batch, :, :, 1] + 0.114 * image_tensor[batch, :, :, 2]
        else:
            gray_diff = image_tensor[batch, :, :, 0]

        dct_result = fftpack.dct(
            fftpack.dct(gray_diff, axis=0, norm='ortho'),
            axis=1, norm='ortho'
        )
        
        log_abs_dct = np.log(np.abs(dct_result) + 1e-8)

        if keep_coeffs is not None:
            log_abs_dct = log_abs_dct[:keep_coeffs, :keep_coeffs]

        features.append(log_abs_dct.flatten())
        
    return np.array(features)

def predict(image, ground_truth):
    sys.stdout.flush()

    img_resized = cv2.resize(image, (32, 32))
    
    if img_resized.shape[-1] == 4:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2RGB)
    
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    global graph, sess
    with graph.as_default(), sess.as_default():
        unprocessed_pred = honeypot_model.predict(img_batch)[0]
        unprocessed_label = LABELS[int(np.argmax(unprocessed_pred))]

        rectified_batch = prn.predict(img_batch)
        diff = img_batch - rectified_batch

        features = compute_dct_features(diff, keep_coeffs=8)
        svm_pred = svm.predict(features)[0]

        if ground_truth and ground_truth != "Unknown":
            undefended_text = f"Model Guessed: {unprocessed_label}"

        # Clean Image Processing
        if svm_pred == 0:
            svm_status = "SVM Classified as Clean Image"
            route_status = "Image sent to Standard Honeypot Classifier"
            
            undefended_confidences = {"N/A (Image is Clean)": 1.0}
            
            defended_text = "N/A - Image is clean (PRN Bypassed)"
            
            confidences = {LABELS[i]: float(unprocessed_pred[i]) for i in range(len(LABELS))}
            defended_class = unprocessed_label

        # Perturbed Image Processing
        else: 
            svm_status = "SVM Classified as Perturbed Image"
            route_status = "Image sent to PRN Defense Classifier"
            
            undefended_confidences = {LABELS[i]: float(unprocessed_pred[i]) for i in range(len(LABELS))}
            
            final_pred_probs = joint_model.predict(img_batch)[0]
            defended_class = LABELS[int(np.argmax(final_pred_probs))]
            confidences = {LABELS[i]: float(final_pred_probs[i]) for i in range(len(LABELS))}
            
            if ground_truth and ground_truth != "Unknown":
                defended_text = f"Model Guessed: {defended_class}"

    return svm_status, route_status, undefended_text, undefended_confidences, defended_text, confidences


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Image(label="Upload an Image"),
        gr.inputs.Dropdown(choices=["Unknown"] + LABELS, default="Unknown", label="Ground Truth")
    ],
    outputs=[
        gr.outputs.Textbox(label="SVM Detector Output"),
        gr.outputs.Textbox(label="Pipeline Routing Action"),
        gr.outputs.Textbox(label="Undefended Honeypot (Baseline)"),
        gr.outputs.Label(num_top_classes=3, label="Baseline Confidence Scores (Adversarial Image)"),
        gr.outputs.Textbox(label="Defended Pipeline (PRN + Honeypot)"),
        gr.outputs.Label(num_top_classes=3, label="Final Output Confidence Scores")
    ],
    title="Universal Adversarial Defense Model Demo",
    description="Image Upload",
    examples=[
        ["clean_ex_1.png", "Automobile"],
        ["clean_ex_2.png", "Airplane"],
        ["dirty_ex_1.png", "Horse"],
        ["dirty_ex_2.png", "Truck"]
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, debug=True)