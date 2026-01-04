# Face Mask Detection with CNN

This project implements a Face Mask Detection system using a Convolutional Neural Network (CNN) and Streamlit for the web interface.

## Overview

The application detects whether a person in an uploaded image is wearing a face mask or not. It uses a pre-trained Keras model (`mask_detection_model.keras`) for prediction.

## Features

-   **Image Upload:** Users can upload images (JPG, PNG, JPEG).
-   **Real-time Prediction:** The app processes the image and predicts "Mask Detected" or "No Mask Detected" with a confidence score.
-   **User-friendly Interface:** Built with Streamlit for easy interaction.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/drizzy765/facemask-detection-with-CNN.git
    cd facemask-detection-with-CNN
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2.  Open your browser to the local URL provided (usually `http://localhost:8501`).

3.  Upload an image to text the detection.

## Dependencies

-   tensorflow
-   streamlit
-   opencv-python
-   pillow
-   numpy
-   pandas
-   matplotlib
-   scikit-learn
-   kagglehub


## Model

The model is saved as `mask_detection_model.keras` and is expected to be in the project root directory.
