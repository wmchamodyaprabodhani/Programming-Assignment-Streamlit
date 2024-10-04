# Image Classification App Using EfficientNet

This is an image classification web application that allows users to upload an image and get predictions using a pre-trained EfficientNet model. The app displays the top 5 predicted classes with confidence scores using the ImageNet dataset.

## Features
- Upload an image for classification.
- Classifies images into one of the 1,000 ImageNet classes.
- Displays the top 5 predictions with confidence levels.
- Interactive and clean UI using Streamlit.

## How to Run

1. Install dependencies from the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt

2. Run the Streamlit app.

   ```bash
   streamlit run app.py

Upload an image using the web interface and press "Classify Image" to see predictions.

File Structure
    app.py: Main Streamlit application file.
    requirements.txt: Python packages needed to run the app.
    README.md: Documentation for the app.

Model Used
EfficientNet (pre-trained on ImageNet) is a highly efficient convolutional neural network architecture that balances model accuracy and performance.

Sample Categories for Testing

You can test the app with images of:

    Animals: Dogs, Cats, Birds
    Objects: Furniture, Vehicles
    Scenes: Landscapes, Cities

Sample Screenshot
https://drive.google.com/drive/folders/14QNYsCsFVglm3Tv-MQLzom9vRCh48yQa?usp=sharing

You Can Run a Demo on:
https://blank-app-26q94f34z39.streamlit.app/#classify-your-favorite-images-with-style