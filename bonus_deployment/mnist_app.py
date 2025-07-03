"""
Bonus Task: MNIST Digit Classifier Web Application
A Streamlit app that uses the trained CNN model to classify handwritten digits.

Author: [Your Team Name]
Date: [Current Date]
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps, ImageFilter
import cv2
import matplotlib.pyplot as plt
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the model in h5 format
        model = keras.models.load_model("../part2_practical/task2_deep_learning/mnist_cnn_model.h5")
        return model
    except:
        try:
            # Try to load the model in SavedModel format
            model = keras.models.load_model("../part2_practical/task2_deep_learning/mnist_cnn_model")
            return model
        except:
            st.error("Model not found. Please make sure the model is saved in the correct location.")
            return None

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    if image.mode != "L":
        image = image.convert("L")
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Apply some image processing to improve recognition
    image = ImageOps.invert(image)  # Invert colors to match MNIST format
    image = np.array(image)
    
    # Normalize pixel values
    image = image / 255.0
    
    # Reshape for model input
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Function to get model predictions
def predict_digit(image, model):
    # Get preprocessed image
    processed_image = preprocess_image(image)
    
    # Get model predictions
    predictions = model.predict(processed_image)
    predicted_digit = np.argmax(predictions)
    confidence = float(predictions[0][predicted_digit])
    
    return predicted_digit, confidence, predictions[0]

# Function to create plot for prediction probabilities
def create_prediction_plot(probabilities):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    ax.bar(x, probabilities)
    ax.set_xticks(x)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# Main app
def main():
    # Title and description
    st.title("✏️ MNIST Handwritten Digit Classifier")
    st.write("""
    ## Welcome to our Digit Recognition App!
    This application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify handwritten digits.
    
    You can either:
    - Draw a digit directly in the app
    - Upload an image of a handwritten digit
    
    The model will predict which digit (0-9) you've drawn or uploaded.
    """)
    
    # Load the trained model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check that the model file exists.")
        return
    
    # Create two columns for the app layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Methods")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["Draw a digit", "Upload an image"])
        
        # Drawing canvas
        with tab1:
            st.write("Draw a digit below (0-9):")
            
            # Create a canvas for drawing
            canvas_result = st.empty()
            
            # Use HTML/CSS/JS for a drawing canvas
            canvas_html = """
            <canvas id="canvas" width="280" height="280" style="border:1px solid #000000;"></canvas>
            <script>
            var canvas = document.getElementById("canvas");
            var ctx = canvas.getContext("2d");
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            var isDrawing = false;
            var lastX = 0;
            var lastY = 0;
            
            canvas.addEventListener("mousedown", function(e) {
                isDrawing = true;
                [lastX, lastY] = [e.offsetX, e.offsetY];
            });
            
            canvas.addEventListener("mousemove", function(e) {
                if (isDrawing) {
                    ctx.strokeStyle = "white";
                    ctx.lineWidth = 15;
                    ctx.lineCap = "round";
                    ctx.beginPath();
                    ctx.moveTo(lastX, lastY);
                    ctx.lineTo(e.offsetX, e.offsetY);
                    ctx.stroke();
                    [lastX, lastY] = [e.offsetX, e.offsetY];
                }
            });
            
            canvas.addEventListener("mouseup", function() {
                isDrawing = false;
            });
            
            canvas.addEventListener("mouseout", function() {
                isDrawing = false;
            });
            
            function clearCanvas() {
                ctx.fillStyle = "black";
                ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
            
            function getImage() {
                return canvas.toDataURL("image/png");
            }
            </script>
            <button onclick="clearCanvas()" style="margin-top: 10px;">Clear Canvas</button>
            """
            
            st.components.v1.html(canvas_html, height=350)
            
            # Get image from canvas
            if st.button("Predict from Drawing"):
                st.write("Drawing recognition not implemented in this preview")
                st.info("In a real implementation, the canvas drawing would be sent to the server for prediction.")
        
        # Image upload
        with tab2:
            st.write("Upload an image of a handwritten digit:")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=200)
                
                if st.button("Predict Digit"):
                    with st.spinner("Predicting..."):
                        # Make prediction
                        predicted_digit, confidence, probabilities = predict_digit(image, model)
                        
                        # Display results
                        st.success(f"Prediction: **{predicted_digit}**")
                        st.progress(confidence)
                        st.write(f"Confidence: {confidence*100:.2f}%")
    
    with col2:
        st.header("Prediction Results")
        
        # Display prediction and probabilities
        st.write("The model's prediction will appear here:")
        result_placeholder = st.empty()
        
        # Example prediction for demonstration
        demo_probabilities = np.zeros(10)
        demo_probabilities[5] = 0.9  # Example prediction for digit 5
        
        # Create and display the example plot
        plot_buf = create_prediction_plot(demo_probabilities)
        result_placeholder.image(plot_buf, caption="Prediction Probabilities (Example)")
        
        # Technical details section
        st.markdown("---")
        st.subheader("About the Model")
        st.write("""
        The model used in this application is a Convolutional Neural Network (CNN) trained on the MNIST dataset of handwritten digits. 
        
        **Model Architecture:**
        - 3 convolutional layers
        - Max pooling
        - Dropout for regularization
        - Dense output layer with softmax activation
        
        **Performance:**
        - Training accuracy: >99%
        - Test accuracy: >98%
        """)

# Run the app
if __name__ == "__main__":
    main()
