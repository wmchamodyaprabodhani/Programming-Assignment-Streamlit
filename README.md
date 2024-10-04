# Image Classification using ResNet-18

CREATED BY: W.M.CHAMODYA PRABODHANI (ITBIN-2110-0087)

This is a **Streamlit web application** that classifies images using the pre-trained **ResNet-18** model. Upload an image, adjust the confidence threshold, and the app will predict the class of the image using the ImageNet dataset.

## Features

- **Image Upload**: Upload an image (JPG format) to be classified by the model.
- **Pre-trained Model**: Uses **ResNet-18**, a smaller convolutional neural network pre-trained on the ImageNet dataset.
- **Top Predictions**: Displays the top 5 predicted classes along with confidence scores.
- **Confidence Threshold**: Use a slider to filter predictions based on confidence levels.
- **Visualizations**: A bar chart visualizing the top predictions using Seaborn.
- **Personalized**: Soft visual theme and a footer with a heart emoji!

## Demo

You can try out the app [here](https://your-streamlit-app-link).

## How to Run Locally

### Requirements

Ensure you have the following installed:
- Python 3.x
- Streamlit
- PyTorch
- Pillow
- Seaborn
- Requests
- Matplotlib

Install the required packages by running:

```bash
pip install -r requirements.txt
