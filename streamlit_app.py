import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import requests
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Title with a softer tone
st.markdown("<h1 style='text-align: center; color: #FF69B4;'>Classify Your Favorite Images!</h1>", unsafe_allow_html=True)

# Sidebar for file upload and user interactions
st.sidebar.header("Upload Your Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file (JPG format)", type="jpg")

# Sidebar confidence threshold slider
confidence_threshold = st.sidebar.slider("Confidence Threshold for Predictions", 0.0, 1.0, 0.5, 0.1)

# Progress bar and status updates
progress_bar = st.progress(0)
status_text = st.empty()

if uploaded_file is not None:
    # Show progress
    status_text.text("Loading and Processing Image...")
    progress_bar.progress(20)

    # Display the uploaded image on the main page
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Image transformation for ResNet-18
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Process the image
    img_tensor = preprocess(img).unsqueeze(0)

    # Load ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Show progress
    status_text.text("Classifying Image...")
    progress_bar.progress(60)

    # Perform classification
    with torch.no_grad():
        outputs = model(img_tensor)

    # Fetch ImageNet labels
    LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    response = requests.get(LABELS_URL)
    labels = response.json()

    # Helper function to get the class label
    def get_class_label(index):
        return labels[str(index)][1]

    # Top predictions and applying confidence threshold
    top_k = torch.topk(outputs, 5).indices.squeeze(0).tolist()
    top_labels = [get_class_label(i) for i in top_k]
    top_values = [outputs[0, i].item() for i in top_k]

    # Filter predictions based on confidence threshold
    filtered_labels = []
    filtered_values = []
    for label, value in zip(top_labels, top_values):
        if value >= confidence_threshold:
            filtered_labels.append(label)
            filtered_values.append(value)

    # Show progress
    status_text.text("Finalizing results...")
    progress_bar.progress(100)

    # Prediction summary and results
    st.write(f"### Summary of Predictions with Confidence Above {confidence_threshold:.2f}:")
    for label, value in zip(filtered_labels, filtered_values):
        st.write(f"- **{label}**: {value:.4f} confidence")

    # Visualizing results using Seaborn
    st.write("### Top Predictions Visualization")
    if filtered_labels:
        df = pd.DataFrame({
            'Labels': filtered_labels,
            'Confidence': filtered_values
        })

        plt.figure(figsize=(8, 4))
        sns.barplot(x="Confidence", y="Labels", data=df, palette="coolwarm")
        st.pyplot(plt)
    else:
        st.write("No predictions above the confidence threshold.")

else:
    st.write("Please upload an image to get predictions.")

# Footer with love note (optional touch)
st.sidebar.markdown("Created by: W.M.Chamodya Prabodhani (ITBIN-2110-0087)")
