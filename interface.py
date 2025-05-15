# filename: dsl_streamlit_ui.py

import streamlit as st

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

