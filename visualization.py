import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
import json

def load_training_history(file_path):
    try:
        with open(file_path, 'r') as file:
            history = json.load(file)
        return history
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None

def load_test_accuracy(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data['test_accuracy']
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except KeyError:
        st.error(f"Key 'test_accuracy' not found in {file_path}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None

def plot_training_accuracy(history):
    if history is None:
        return
    epochs = range(1, len(history['accuracy']) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(epochs), y=history['accuracy'], mode='lines+markers', name='Training Accuracy', line=dict(color='#eb1948')))
    fig.add_trace(go.Scatter(x=list(epochs), y=history['val_accuracy'], mode='lines+markers', name='Validation Accuracy', line=dict(color='#b71338')))
    fig.update_layout(title='Training and Validation Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy')
    st.plotly_chart(fig)

def plot_training_loss(history):
    if history is None:
        return
    epochs = range(1, len(history['loss']) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(epochs), y=history['loss'], mode='lines+markers', name='Training Loss', line=dict(color='#eb1948')))
    fig.add_trace(go.Scatter(x=list(epochs), y=history['val_loss'], mode='lines+markers', name='Validation Loss', line=dict(color='#b71338')))
    fig.update_layout(title='Training and Validation Loss', xaxis_title='Epochs', yaxis_title='Loss')
    st.plotly_chart(fig)

def plot_confusion_matrix(cm, classes):
    if cm is None:
        return
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=['#eb1948', '#b71338'])
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='True')
    st.plotly_chart(fig)

def app():
    st.markdown('<h1 class="title-font">📊 Visualize and Analyze Model\'s Performance</h1>', unsafe_allow_html=True)
    st.markdown("""
    **🔍 Dive into the details of our model's performance:**

    - **Test Accuracy:** Check the final accuracy on the test dataset. 🎯
    - **Training and Validation Accuracy:** See how your model improves with each epoch. 📈
    - **Loss Metrics:** Track the progress of loss reduction throughout training. 📉
    - **Confusion Matrix:** Understand your model's classification results with a visual confusion matrix. 🧩
    """)
    
    tab1, tab2, tab3 = st.tabs(["Leukemia", "Lung Cancer", "Skin Cancer"])

    with tab1:
        st.markdown('<h2 class="sub-title">Leukemia Model Performance</h2>', unsafe_allow_html=True)
        with st.expander("See More Details"):
            st.markdown("""
                        ### 📂 Dataset Details:
                        - **Total Images:** 5,040 labeled images of blood cells 🧬.
                        - **Categories:** Cancerous 🩸 and Non-cancerous 🟢 blood cells.
                        - **Source:** [C-NMC-2019 - The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/c-nmc-2019/) 🌐.
                        
                        ### 🧠 How It Helps:

                        - **Gain Insights:** Identify trends and performance issues to optimize your model. 🌟
                        - **Analyze Trends:** Detect patterns that can guide your model improvement efforts. 🔍
                        - **Enhance Performance:** Use the insights to fine-tune and enhance your model’s accuracy. ⚙️
                        - **Interact with Visuals:** Check out the interactive plots and matrices below to get a comprehensive view of our model's performance. 📊
                        """)
            
        model_leukemia = load_model('models/leukemia_cancer_model.h5')
        history_leukemia = load_training_history('json_files/leukemia/training_history.json')
        test_accuracy_leukemia = load_test_accuracy('json_files/leukemia/test_accuracy.json')
        if test_accuracy_leukemia is not None:
            st.markdown(f'<p class="custom-font">Leukemia Test Accuracy: {test_accuracy_leukemia:.4f} ({test_accuracy_leukemia * 100:.2f}%)</p>', unsafe_allow_html=True)
        plot_training_accuracy(history_leukemia)
        plot_training_loss(history_leukemia)
        cm_leukemia = np.array([[392, 96], [23, 497]])
        classes_leukemia = ['Non-Cancerous', 'Cancerous']
        plot_confusion_matrix(cm_leukemia, classes_leukemia)
        

    with tab2:
        st.markdown('<h2 class="sub-title">Lung Cancer Model Performance</h2>', unsafe_allow_html=True)
        with st.expander("See More Details"):
            st.markdown("""
                        ### 📂 Dataset Details:
                        - **Total Images:** 15,000 histopathological images 🧬.
                        - **Classes:** Lung Adenocarcinoma 🫁, Lung Squamous Cell Carcinoma 🫁, and Benign 🟢.
                        - **Source:** [Lung Cancer Detection Dataset on Kaggle](https://www.kaggle.com/code/mohamedsameh0410/lung-cancer-detection-with-cnn-efficientnetb3/input) 🌐.
                        
                        ### 🧠 How It Helps:

                        - **Gain Insights:** Identify trends and performance issues to optimize your model. 🌟
                        - **Analyze Trends:** Detect patterns that can guide your model improvement efforts. 🔍
                        - **Enhance Performance:** Use the insights to fine-tune and enhance your model’s accuracy. ⚙️
                        - **Interact with Visuals:** Check out the interactive plots and matrices below to get a comprehensive view of our model's performance. 📊
                        """)
            
        model_lung = load_model('models/lung_cancer_model.h5')
        history_lung = load_training_history('json_files/lung cancer/training_history.json')
        test_accuracy_lung = load_test_accuracy('json_files/lung cancer/test_accuracy_and_training_history.json')
        if test_accuracy_lung is not None:
            st.markdown(f'<p class="custom-font">Lung Cancer Test Accuracy: {test_accuracy_lung:.4f} ({test_accuracy_lung * 100:.2f}%)</p>', unsafe_allow_html=True)
        plot_training_accuracy(history_lung)
        plot_training_loss(history_lung)
        cm_lung = np.array([[515, 0, 0], [0, 492, 0], [0, 0, 493]])
        classes_lung = ['lung_aca', 'lung_n', 'lung_scc']
        plot_confusion_matrix(cm_lung, classes_lung)

    with tab3:
        st.markdown('<h2 class="sub-title">Skin Cancer Model Performance</h2>', unsafe_allow_html=True)
        with st.expander("See More Details"):
            st.markdown("""
                        ### 📂 Dataset Details:
                        - **Total Images:** 10,000 images 🧬.
                        - **Classes:** Malignant (Melanoma) 🩸 and Benign 🟢.
                        - **Training Set:** 9,600 images
                        - **Evaluation Set:** 1,000 images
                        - **Source:** [Melanoma Skin Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) 🌐.
                        
                        ### 🧠 How It Helps:

                        - **Gain Insights:** Identify trends and performance issues to optimize your model. 🌟
                        - **Analyze Trends:** Detect patterns that can guide your model improvement efforts. 🔍
                        - **Enhance Performance:** Use the insights to fine-tune and enhance your model’s accuracy. ⚙️
                        - **Interact with Visuals:** Check out the interactive plots and matrices below to get a comprehensive view of our model's performance. 📊
                        """)
        model_skin = load_model('models/skin_cancer_model.h5')
        test_accuracy_skin = load_test_accuracy('json_files/skin cancer/test_accuracy.json')
        if test_accuracy_skin is not None:
            test_accuracy_percentage = test_accuracy_skin * 100
            st.markdown(f'<p class="custom-font">Skin Cancer Test Accuracy: {test_accuracy_skin:.4f} ({test_accuracy_skin * 100:.2f}%)</p>', unsafe_allow_html=True)
        history_skin = load_training_history('json_files/skin cancer/training_history.json')
        plot_training_accuracy(history_skin)
        plot_training_loss(history_skin)
        cm_skin = np.array([[447, 53], [135, 365]])
        classes_skin = ['Benign', 'Malignant']
        plot_confusion_matrix(cm_skin, classes_skin)

if __name__ == "__main__":
    app()

