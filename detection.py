import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the improved trained model for skin cancer detection
try:
    skin_model = load_model('models/skin_cancer_model.h5')
    logging.info("Skin cancer model loaded successfully")
except Exception as e:
    logging.error(f"Error loading skin cancer model: {e}")
    skin_model = None

# Prediction function for skin cancer
def predict_skin_image(image):
    if skin_model is None:
        logging.error("Skin cancer model is not loaded.")
        return np.array([[0, 0]])
    
    try:
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)
        logging.debug(f"Image shape for skin prediction: {img.shape}")

        prediction = skin_model.predict(img)
        logging.debug(f"Skin prediction: {prediction}")
        
        return prediction
    except Exception as e:
        logging.error(f"Error during skin prediction: {e}")
        return np.array([[0, 0]])

# Load the trained models
try:
    leukemia_model = load_model('models/leukemia_cancer_model.h5')
    lung_model = load_model('models/lung_cancer_model.h5')
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    leukemia_model = None
    lung_model = None
    
    # Function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def predict_leukemia_image(image):
    if leukemia_model is None:
        logging.error("Leukemia model is not loaded.")
        return None

    try:
        # Preprocess the image to match the input shape of the model
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)
        logging.debug(f"Image shape for prediction: {img.shape}")

        # Get the model prediction (assuming the output is a single float value)
        prediction = leukemia_model.predict(img)
        logging.debug(f"Leukemia Prediction: {prediction}")

        # The prediction is a single probability value (a float), no need for indexing
        cancerous_prob = float(prediction)  # Directly convert the prediction to a float
        non_cancerous_prob = 1 - cancerous_prob  # Confidence for non-cancerous is the inverse

        return cancerous_prob, non_cancerous_prob
    except Exception as e:
        logging.error(f"Error during leukemia prediction: {e}")
        return None

# Define the class mapping with exact cancer types
index = {
    'lung_aca': 'Lung Adenocarcinoma (Cancerous)',
    'lung_n': 'Lung Benign Tissue (Non-Cancerous)',
    'lung_scc': 'Lung Squamous Cell Carcinoma (Cancerous)'
}

def predict_lung_image(image):
    if lung_model is None:
        logging.error("Lung cancer model is not loaded.")
        return None

    try:
        # Preprocess the image to match the input shape of the model
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img = np.asarray(img) / 255.0
        img = np.expand_dims(img, axis=0)
        logging.debug(f"Image shape for prediction: {img.shape}")

        # Get the model prediction
        prediction = lung_model.predict(img)
        logging.debug(f"Lung Cancer Prediction: {prediction}")
        
        # The prediction returns probabilities for each class
        lung_aca_prob = prediction[0][0]  # Probability of Lung Adenocarcinoma (cancerous)
        lung_n_prob = prediction[0][1]    # Probability of Lung Benign Tissue (non-cancerous)
        lung_scc_prob = prediction[0][2]  # Probability of Lung Squamous Cell Carcinoma (cancerous)

        # Get the class index with the highest probability
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Map the predicted class index to the corresponding cancer type
        if predicted_class_index == 0:  # lung_aca
            predicted_class = 'Lung Adenocarcinoma'
            cancer_status = 'Cancerous'
        elif predicted_class_index == 1:  # lung_n
            predicted_class = 'Lung Benign Tissue'
            cancer_status = 'Non-Cancerous'
        elif predicted_class_index == 2:  # lung_scc
            predicted_class = 'Lung Squamous Cell Carcinoma'
            cancer_status = 'Cancerous'

        return predicted_class, cancer_status, lung_aca_prob, lung_n_prob, lung_scc_prob
    except Exception as e:
        logging.error(f"Error during lung cancer prediction: {e}")
        return None

def app():
    st.markdown('<h1 class="title-font">📸 Detection Page</h1>', unsafe_allow_html=True)

    # Tabs for different cancer types
    tabs = st.tabs(["Leukemia Detection", "Lung Cancer Detection", "Skin Cancer Detection"])

    with tabs[0]:
        st.header("🧪 Leukemia Detection (For Medical Professionals Only)")
        st.markdown("""
                    **🔍 Ready to detect leukemia?** 
                
                    Upload a blood smear image for analysis. Our model will assist in identifying potential leukemia cells in the sample.
                """)
        with st.expander("See More Details"):
            st.markdown("""
                    ### 📂 Upload an Image
                    Select a saved **blood smear image** from your device for evaluation. Ensure the image is clear and well-captured.

                    ### How It Works:
                    - **Upload:** 
                    Upload a blood smear image to analyze for signs of leukemia. 🩸

                    - **Analyze:** 
                    Our deep learning model processes the image and identifies abnormal cells potentially indicating leukemia.🔬

                    - **Get Insights:** 
                    Get immediate feedback on the presence of leukemia cells. 💡

                    Ready to upload? Let's detect leukemia together! 🚀
                    """)
        
        
        uploaded_file = st.file_uploader("Choose a leukemia image...", type=["jpg", "jpeg", "png", "bmp"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((500, 500))
            img_base64 = image_to_base64(image)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <img src="data:image/png;base64,{img_base64}" width="500" style='border-radius: 15px' 'box-shadow: 0 4px 8px 0 rgba(0, 0, 0.2, 0.2);'/>
                    <div style='margin-top: 10px; font-size: 16px; color: #666;'>🖼️ Uploaded Image</div>
                </div>
                """, unsafe_allow_html=True)

            prediction = predict_leukemia_image(image)

            if prediction is not None:
                cancerous_prob, non_cancerous_prob = prediction

                # Display confidence for both cancerous and non-cancerous
                st.markdown(f"""
                            <div class='dmain'>
                                <p><strong>Cancerous Probability:<strong> {cancerous_prob * 100:.2f}%</p>
                                <p><strong>Non-Cancerous Probability:<strong> {non_cancerous_prob * 100:.2f}%</p>
                            </div>
                                """, unsafe_allow_html=True)

                threshold = 0.5

                if cancerous_prob > threshold:
                    st.markdown(f"""
                        <div class="cprob">
                            <h3 style='color: #eb1948;'>🚨 Leukemia Detected</h3>
                            <p style='color: #0F0F0F;'>Detection: Cancerous (Confidence: {cancerous_prob * 100:.2f}%)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.error("Please consult with a healthcare professional for further tests and treatments. 🩺📅")
                else:
                    st.markdown(f"""
                        <div class="ncprob">
                            <h3 style='color: #00f731;'>✅ No Leukemia Detected</h3>
                            <p style='color: #0F0F0F;'>Detection: Non-Cancerous (Confidence: {non_cancerous_prob * 100:.2f}%)</p>
                            <p style='color: #0F0F0F;'>Advice: Keep monitoring your health regularly. 📊🩸</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("Keep monitoring your health regularly. 📊🩸")

    with tabs[1]:
        st.header("🫁 Lung Cancer Detection")
        st.markdown("""
                    **🔍 Concerned about lung cancer?** 

                    Upload a tissue image to check for potential signs of lung cancer.
                    """)
        with st.expander("See More Details"):
            st.markdown("""
                ### 📂 Upload an Image
                Select a saved **tissue image** (histopathological medical image) from your device. Ensure the image is of good quality to maximize the accuracy of the results.
        
                ### How It Works:
                - **Upload:** 
                Upload a tissue image for analysis. 🧬
                
                - **Analyze:** 
                Our model evaluates the tissue image for potential lung cancer symptoms.🔍
                
                - **Get Insights:** 
                Receive instant analysis and insights on the condition of the lung tissue. 💡
                
                Ready to get started? Upload your tissue image below and let’s begin! 🚀
            """)
                    
        uploaded_file = st.file_uploader("Choose a lung image...", type=["jpg", "jpeg", "png"])
    
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((500, 500))
            img_base64 = image_to_base64(image)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <img src="data:image/png;base64,{img_base64}" width="500" style='border-radius: 15px' 'box-shadow: 0 4px 8px 0 rgba(0, 0, 0.2, 0.2);'/>
                    <div style='margin-top: 10px; font-size: 16px; color: #666;'>🖼️ Uploaded Image</div>
                </div>
                """, unsafe_allow_html=True)
    
            prediction = predict_lung_image(image)
    
            if prediction is not None:
                predicted_class, cancer_status, lung_aca_prob, lung_n_prob, lung_scc_prob = prediction
    
                st.markdown(f"""
                            <div class='dmain'>
                                <p><strong>Lung Adenocarcinoma (Cancerous):</strong> {lung_aca_prob * 100:.2f}%</p>
                                <p><strong>Lung Squamous Cell Carcinoma (Cancerous):</strong> {lung_scc_prob * 100:.2f}%</p>
                                <p><strong>Lung Benign Tissue (Non-Cancerous):</strong> {lung_n_prob * 100:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
    
                # Display result with custom message
                if cancer_status == "Cancerous":
                    st.markdown(f"""
                        <div class="cprob">
                            <h3 style='color: #eb1948;'>🚨 Lung Cancer Detected ({cancer_status})</h3>
                            <p style='color: #0F0F0F;'>Cancer Type: {predicted_class}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.error("Please seek immediate medical attention for diagnosis and treatment options. 🚑⚠️")
                else:
                    st.markdown(f"""
                        <div class="ncprob">
                            <h3 style='color: #00f731;'>✅ No Lung Cancer Detected ({cancer_status})</h3>
                            <p style='color: #0F0F0F;'>Tissue Type: {predicted_class}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.success("Maintain a healthy lifestyle and consider regular check-ups. 🥗💪")
                
    with tabs[2]:
        st.header("📸 Skin Cancer Detection")
        st.markdown("""
                **🔍 Ready to check for skin cancer?** 

                Upload a skin image to get started. Our advanced model will analyze the image and provide insights into potential skin conditions.
                """)
        with st.expander("See More Details"):
            st.markdown("""
                        ### 📂 Upload an Image
                        Select a saved **skin lesion image** from your device for evaluation. Ensure the image is clear and well-lit for the best results.
    
                        ### How It Works:
                        - **Upload:** 
                        Easily upload an image for analysis. Make sure the area of interest is visible and clear for accurate results. 📸
                        
                        - **Analyze:** 
                        Our advanced model will process the image and analyze it for potential skin conditions.🔍
                        
                        - **Get Insights:** 
                        Receive immediate feedback on your skin health based on the analysis.💡
                        
    
                        Ready to begin? Upload an image below and let’s get started! 🚀
                        """)
        

        uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = image.resize((500, 500))
            img_base64 = image_to_base64(image)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <img src="data:image/png;base64,{img_base64}" width="500" style='border-radius: 15px' 'box-shadow: 0 4px 8px 0 rgba(0, 0, 0.2, 0.2);'/>
                    <div style='margin-top: 10px; font-size: 16px; color: #666;'>🖼️ Uploaded Image</div>
                </div>
                """, unsafe_allow_html=True)

            prediction = predict_skin_image(image)

            if prediction is not None:
                benign_prob = prediction[0][0]
                malignant_prob = prediction[0][1]
                st.markdown(f"""
                            <div class='dmain'>
                                <p><strong>Benign Probability:</strong> {benign_prob * 100:.2f}%</p>
                                <p><strong>Malignant Probability:</strong> {malignant_prob * 100:.2f}%</p>
                            </div>
                                """, unsafe_allow_html=True)
                    
                threshold = 0.5
                
                if malignant_prob > threshold:
                    st.markdown(f"""
                        <div class="cprob">
                            <h3 style='color: #eb1948;'>🚨 Skin Cancer Detected</h3>
                            <p style='color: #0F0F0F;'>Detection: Malignant (Confidence: {malignant_prob * 100:.2f}%)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.error("Please consult a dermatologist immediately for a thorough examination. 🧴🔍")
                else:
                    st.markdown(f"""
                        <div class="ncprob">
                            <h3 style='color: #00f731;'>✅ No Skin Cancer Detected</h3>
                            <p style='color: #0F0F0F;'>Detection: Benign (Confidence: {benign_prob * 100:.2f}%)</p>
                    """, unsafe_allow_html=True)
                    st.success("Continue regular skin checks and maintain good skincare practices. 🧖‍♀️🧴")

if __name__ == "__main__":
    app()


