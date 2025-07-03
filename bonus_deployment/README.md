# Streamlit MNIST Digit Classifier App

This is a web application built with Streamlit that serves the trained MNIST digit classifier model.

## Features

- Interactive drawing canvas to draw digits
- Image upload for digit recognition
- Real-time prediction with confidence scores
- Visualization of prediction probabilities

## Installation

1. Install the required dependencies:
```
pip install streamlit tensorflow pillow opencv-python matplotlib
```

2. Make sure the trained model is available at `../part2_practical/task2_deep_learning/mnist_cnn_model.h5`

## Running the App

From this directory, run:
```
streamlit run mnist_app.py
```

This will start the web application on your local machine and open it in your browser.

## Usage

1. **Drawing a digit**: 
   - Use the drawing canvas to draw a digit (0-9)
   - Click "Predict from Drawing" to get the model's prediction

2. **Uploading an image**:
   - Upload an image file of a handwritten digit
   - Click "Predict Digit" to get the model's prediction

The prediction results will show:
- The predicted digit
- Confidence score
- Bar chart of probabilities for all digits

## How It Works

1. The app loads the pre-trained CNN model
2. User input (drawing or image) is preprocessed to match the model's input requirements
3. The model makes a prediction
4. Results are displayed in a user-friendly format

## Deployment

For production deployment, you can:
1. Deploy on Streamlit Sharing
2. Deploy on Heroku
3. Use Docker to containerize the application
