import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SoundSense AI",
    page_icon="🔊",
    layout="centered"
)

# --- LOAD MODEL & ENCODER (Cached for Speed) ---
@st.cache_resource
def load_model_and_encoder():
    try:
        # Load the Keras model
        model = tf.keras.models.load_model('urban_sound_classifier.keras')
        
        # Load the Label Encoder
        with open('label_encoder_v2.pkl', 'rb') as f:
            le = pickle.load(f)
            
        return model, le
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, le = load_model_and_encoder()

# --- PREPROCESSING FUNCTION ---
def preprocess_audio(file_path):
    # Load audio file
    # We use a fixed sample rate and duration to match training
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
    
    # Extract MFCCs (Must match training: 40 features)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Scale features (Average over time)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Reshape for the CNN (1 sample, 40 features, 1 channel)
    mfccs_input = mfccs_scaled.reshape(1, 40, 1)
    
    return mfccs_input, audio, sample_rate

# --- MAIN UI ---
st.title("🔊 SoundSense: Urban Sound Classifier")
st.write("Upload an audio file (.wav or .mp3) to identify the urban sound.")

# File Uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save file temporarily to disk (Librosa needs a path)
    with open("temp_audio_file", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Listening...")
        st.audio(uploaded_file, format='audio/wav')

    # Analyze Button
    if st.button("Analyze Sound"):
        with st.spinner("Analyzing audio patterns..."):
            try:
                # 1. Preprocess
                features, audio_data, sr = preprocess_audio("temp_audio_file")
                
                # 2. Predict
                prediction = model.predict(features)
                predicted_index = np.argmax(prediction, axis=1)
                confidence = np.max(prediction)
                
                # 3. Decode Label
                predicted_class = le.inverse_transform(predicted_index)[0]
                
                # --- DISPLAY RESULTS ---
                st.success("Analysis Complete!")
                
                # Big Result Metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric(label="Predicted Sound", value=predicted_class.replace("_", " ").upper())
                with metric_col2:
                    st.metric(label="Confidence Level", value=f"{confidence * 100:.2f}%")
                
                # --- VISUALIZATION ---
                st.subheader("Confidence Distribution")
                
                # Get top 3 predictions
                top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                top_3_values = prediction[0][top_3_indices]
                top_3_labels = le.inverse_transform(top_3_indices)
                
                # Plot Bar Chart
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.barh(top_3_labels, top_3_values, color='#4CAF50')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                st.pyplot(fig)
                
                # --- SPECTROGRAM VISUALIZATION ---
                with st.expander("See Audio Spectrogram"):
                    fig2, ax2 = plt.subplots(figsize=(10, 3))
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
                    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
                    plt.colorbar(format='%+2.0f dB')
                    st.pyplot(fig2)

            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Cleanup temp file
if os.path.exists("temp_audio_file"):
    os.remove("temp_audio_file")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info(
    """
    **SoundSense** uses a Convolutional Neural Network (CNN) 
    trained on the UrbanSound8K dataset to classify 10 types 
    of city sounds.
    """
)
st.sidebar.write("Developed for SDG 11: Sustainable Cities")
