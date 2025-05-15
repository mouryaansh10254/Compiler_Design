# filename: dsl_streamlit_ui.py

import streamlit as st
import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import traceback


# --- Classifier ---
class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.unclassified = []
        self.image_paths = []

    def load_data(self, data_path):
        dataset = datasets.ImageFolder(data_path, transform=self.transform)
        self.class_names = dataset.classes
        self.data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    def define_model(self, model_type="resnet18"):
        if model_type == "resnet18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_ftrs = base_model.fc.in_features
            # Matching saved model structure
            base_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, len(self.class_names))
            )
            self.model = base_model.to(self.device)

    def train_model(self, epochs, save_path):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

        for epoch in range(epochs):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        torch.save(self.model.state_dict(), save_path)

    def load_model(self, path):
        if self.model is None:
            raise ValueError("Model must be defined before loading weights.")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def load_unclassified_images(self, folder):
        self.unclassified = []
        self.image_paths = []
        for file in os.listdir(folder):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder, file)
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                self.unclassified.append(image)
                self.image_paths.append(img_path)

    def classify_images(self):
        results = []
        self.model.eval()
        with torch.no_grad():
            for image in self.unclassified:
                input_tensor = image.unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                _, pred = torch.max(output, 1)
                results.append(self.class_names[pred.item()])
        return results

    def organize_images(self, output_dir, results):
        os.makedirs(output_dir, exist_ok=True)
        for img_path, label in zip(self.image_paths, results):
            class_dir = os.path.join(output_dir, label)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(img_path, os.path.join(class_dir, os.path.basename(img_path)))

# --- DSL Compiler ---
class DSLCompiler:
    def __init__(self, dsl_path):
        self.dsl_path = dsl_path
        self.classifier = ImageClassifier()
        self.ir_log = []
        self.execution_result = []

    def compile_and_run(self):
        with open(self.dsl_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        for line in lines:
            self.ir_log.append(f"Processing: {line}")
            if line.startswith("load_images from"):
                path = self._extract_path(line)
                self.classifier.load_data(path)
                self.ir_log.append(f"Loaded images from {path}")

            elif line.startswith("define_model"):
                model_type = line.split()[1]
                self.classifier.define_model(model_type)
                self.ir_log.append(f"Defined model: {model_type}")

            elif line.startswith("train_model for"):
                epochs = int(line.split()[2])
                path = "models/model.pth"
                if "save_model_to" in line:
                    path = self._extract_path(line)
                self.classifier.train_model(epochs, path)
                self.ir_log.append(f"Trained model for {epochs} epochs and saved to {path}")

            elif line.startswith("load_model from"):
                path = self._extract_path(line)
                self.classifier.load_model(path)
                self.ir_log.append(f"Loaded model from {path}")

            elif line.startswith("predict_images from"):
                path = self._extract_path(line)
                self.classifier.load_unclassified_images(path)
                self.ir_log.append(f"Loaded unclassified images from {path}")

            elif line.startswith("classify_images"):
                results = self.classifier.classify_images()
                self.execution_result.extend(results)
                self.ir_log.append("Classified unclassified images")

            elif line.startswith("organize_images into"):
                path = self._extract_path(line)
                results = self.classifier.classify_images()
                self.classifier.organize_images(path, results)
                self.ir_log.append(f"Organized classified images into {path}")

            else:
                self.ir_log.append(f"Unknown command: {line}")

        return "\n".join(self.ir_log), "\n".join(self.execution_result)

    def _extract_path(self, line):
        return line.split('"')[1] if '"' in line else line.split()[-1]
    

# --- Streamlit UI ---
st.set_page_config(page_title="DSL Compiler", layout="wide")
st.title("🛠️ DSL Compiler for Image Classification")
theme = st.selectbox("🎨 Select Theme", ["Light", "Dark"])

def apply_theme(selected_theme):
       if selected_theme == "Dark":
        st.markdown("""
            <style>
                html, body, [data-testid="stApp"] {
                    background-color: #0e1117;
                    color: #ffffff;
                }
                textarea, input {
                    background-color: #1c1f26 !important;
                    color: #ffffff !important;
                }
                button {
                    background-color: #262730 !important;
                    color: #ffffff !important;
                }
                .stFileUploader label,
                .stTextInput label,
                .stTextArea label {
                    color: #ffffff !important;
                }
                .streamlit-expanderHeader {
                    color: #ffffff !important;
                }
                pre, code {
                    background-color: #1e1e1e !important;
                    color: #f8f8f2 !important;
                }
                [data-testid="stSidebar"] {
                    background-color: #1c1f26;
                }
            </style>
        """, unsafe_allow_html=True)
       else:
          st.markdown("""
            <style>
                html, body, [data-testid="stApp"] {
                    background-color: #ffffff;
                    color: #000000;
                }
                textarea, input {
                    background-color: #f5f5f5 !important;
                    color: #000000 !important;
                }
                button {
                    background-color: #e0e0e0 !important;
                    color: #000000 !important;
                }
                .stFileUploader label,
                .stTextInput label,
                .stTextArea label {
                    color: #000000 !important;
                }
                .streamlit-expanderHeader {
                    color: #000000 !important;
                }
                pre, code {
                    background-color: #f0f0f0 !important;
                    color: #000000 !important;
                }
                [data-testid="stSidebar"] {
                    background-color: #f0f0f0;
                }
            </style>
        """, unsafe_allow_html=True)

apply_theme(theme)
# Add Help Section
with st.expander("ℹ️ Help: How to Use This DSL Compiler"):
    st.markdown("""
    ### 📘 Instructions

    This tool allows you to write or upload a DSL (Domain Specific Language) script to automate image classification tasks.

#### 🔤 DSL Syntax Overview

Each command must be written on a separate line. Here's a list of supported commands:

- `load_images from "<path>"`  
    Loads images from a directory structured by class folders.

- `define_model resnet18`  
    Defines a ResNet18-based image classifier.

- `train_model for <epochs> save_model_to "<model_path>"`  
    Trains the model and saves it to the specified path.  
    Example: `train_model for 5 save_model_to "models/model.pth"`

- `load_model from "<model_path>"`  
    Loads a pre-trained model.

- `predict_images from "<path>"`  
    Loads unclassified images from the specified folder.

- `classify_images`  
    Runs classification on the loaded unclassified images.

- `organize_images into "<output_folder>"`  
    Organizes the classified images into folders based on predicted labels.

#### 🧾 Example DSL Code

```dsl
load_images from "data/train"
define_model resnet18
train_model for 3 save_model_to "models/resnet.pth"
load_model from "models/resnet.pth"
predict_images from "data/unclassified"
classify_images
organize_images into "output" """)
# Text area to input DSL code
dsl_code = st.text_area("✍️ Enter DSL Code", height=250)

# File uploader for DSL file
uploaded_file = st.file_uploader("📂 Or upload DSL file", type=["dsl"])

# Read uploaded file content
if uploaded_file:
    dsl_code = uploaded_file.read().decode("utf-8")
    st.code(dsl_code)

# Compile button
if st.button("⚙️ Compile Code"):
    if not dsl_code.strip():
        st.warning("DSL code is empty.")
    else:
        # Placeholder for backend function call
        # result = compile_and_run_dsl(dsl_code)

        # UI response placeholder
        st.success("✅ Ready to compile. Backend logic not included in this UI-only version.")

        # Placeholder output sections
        st.subheader("📜 Intermediate Representation (IR)")
        # st.code(result.ir_log)
        st.code("IR output goes here...")

        st.subheader("💡 Execution Result")
        # st.code(result.execution_result)
        st.code("Classification results go here...")

