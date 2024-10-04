import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests
import json
import matplotlib.pyplot as plt

# Custom CSS to improve UI design
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container {
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    h1 {
        font-weight: 600;
        color: #333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }
    .sidebar .sidebar-content {
        background-color: #E8F0FE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("Image Classification with EfficientNet")
st.write("Created By: W.M.CHAMODYA PRABODHANI (ITBIN-2110-0087)")
st.write("Upload an image and classify it using the EfficientNet model (pre-trained on ImageNet).")

# Sidebar - App Information
with st.sidebar:
    st.header("App Information")
    st.write("This app classifies images using a pre-trained EfficientNet model.")
    st.write("Simply upload an image, and the model will predict its category, displaying the top 5 results with a confidence bar.")
    st.write("Developed using PyTorch and Streamlit.")
    st.write("Ensure clear images for best results.")
    st.markdown("#### Categories to Test:")
    st.markdown("- **Animals**: Cats, Dogs, Birds\n- **Objects**: Cars, Furniture\n- **Scenes**: Landscapes, Cities")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)

# File uploader widget
uploaded_file = st.file_uploader("Upload your image...", type=["jpg", "png", "jpeg"])

# Button to classify image
if uploaded_file:
    classify_button = st.button("Classify Image")

    if classify_button:
        # Show progress
        st.markdown("#### Processing Image...")

        # Load and preprocess the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(img).unsqueeze(0)

        # Load pre-trained EfficientNet model
        model = models.efficientnet_b0(pretrained=True)
        model.eval()

        # Perform image classification
        with torch.no_grad():
            outputs = model(img_tensor)

        # Get the predicted class
        _, predicted = outputs.max(1)
        class_index = predicted.item()

        # Load ImageNet labels
        LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        response = requests.get(LABELS_URL)
        labels = response.json()

        # Function to get the label from index
        def get_class_label(index):
            return labels[str(index)][1]

        # Show prediction
        class_label = get_class_label(class_index)
        st.markdown(f"### Prediction: **{class_label}**")

        # Top 5 predictions
        st.markdown("#### Top 5 Predictions")
        top_k = torch.topk(outputs, 5).indices.squeeze(0).tolist()
        top_labels = [get_class_label(i) for i in top_k]
        top_values = [outputs[0, i].item() for i in top_k]

        # Display bar chart of confidence levels
        fig, ax = plt.subplots()
        ax.barh(top_labels, top_values, color="teal")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Sidebar Info - Top classes
        st.sidebar.subheader("Top 5 Predictions")
        for i in range(5):
            st.sidebar.write(f"{i + 1}: {get_class_label(top_k[i])} ({top_values[i]:.4f})")

else:
    st.write("Please upload an image to classify.")