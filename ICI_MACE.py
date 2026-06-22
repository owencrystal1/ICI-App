import io
import re
import os
import sys
import time
import torch
import joblib
import pickle
import shutil
import imageio
import pydicom
import tempfile
import inference
#import grad_cam2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import streamlit as st
from pathlib import Path
import albumentations as A
from torchvision import models
from urllib.error import URLError
import matplotlib.font_manager as fm
from streamlit_cropper import st_cropper
from utils import model_base, data_utils
from PIL import Image, ImageDraw, ImageFont
from albumentations.pytorch import ToTensorV2

try:
    model_path = r'./models/'
    output_dir = r'./output/'
    os.makedirs(output_dir, exist_ok=True)

    #st.header(f"Thick-Walled Cardiomyopathy Classification Echo AI Tool")
    # Create two columns
    col1, col2 = st.columns([1, 5])  # Adjust width ratio as needed

    with col1:
        st.image("mayo_logo.png", width=900)  # Replace with your image path or URL

    with col2:
        st.title(f"Immune Checkpoint Inhibitor MACE Risk Calculator")
    
    st.write("The tools provided on this platform are intended for research and educational purposes only. Any use of these tools in a clinical workflow is done at the user's risk.")
    st.write("Reference: Ayoub C, Appari L, Pereyra M, Farina JM, Chao CJ, Scalia IG, Mahmoud AK, Abbas MT, Baba NA, Jeong J, Lester SJ, Patel BN, Arsanjani R, Banerjee I. Multimodal Fusion Artificial Intelligence Model to Predict Risk for MACE and Myocarditis in Cancer Patients Receiving Immune Checkpoint Inhibitor Therapy. JACC Adv. 2024 Dec 13;4(1):101435. doi: 10.1016/j.jacadv.2024.101435. PMID: 39759436; PMCID: PMC11699614")
    st.markdown("---")  # horizontal line
    st.header('Demographics and Clinical History')
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.radio("Gender", options=["Male", "Female", "Other"])
    with col2:
        hypertension = st.radio("Hypertension", options=["Yes", "No", "Unknown"])
        stroke = st.radio("Stroke/TIA", options=["Yes", "No", "Unknown"])
    with col3:
        hyperlipidemia = st.radio("Hyperlipidemia", options=["Yes", "No", "Unknown"])
        creatinine = st.radio("Creatinine", options=["Yes", "No", "Unknown"])
    with col4:
        diabetes = st.radio("Diabetes", options=["Yes", "No", "Unknown"])
        copd = st.radio("COPD", options=["Yes", "No", "Unknown"])
    with col5:
        afib = st.radio("Atrial Fibrillation", options=["Yes", "No", "Unknown"])
        vt = st.radio("Ventricular Tachycardia", options=["Yes", "No", "Unknown"])

    st.header('ECG Upload')
    ecg_file = st.file_uploader("Choose an ECG image", type=["png"])

    if ecg_file is not None:
        # Open image with PIL
        ecg_image = Image.open(ecg_file).convert('RGB')
        temp_dir = "./temp_ecg"
        os.makedirs(temp_dir, exist_ok=True)
        ecg_image.save(f"./temp_ecg/uploaded.png")

        st.write("Adjust the crop area to include the entire 12-lead ECG. Double-click to confirm:")
        
    
        # Cropper widget
        ecg_image = st_cropper(ecg_image, box_color='blue', aspect_ratio=None)
        original_image = ecg_image
        ecg_image.save(f"./temp_ecg/cropped.png")

        # Show cropped image
        st.write('ECG Image Preview')
        st.image(ecg_image, use_container_width=True)

        ecg_path = Path(f"./temp_ecg/{ecg_file.name}")

        buffer = io.BytesIO()

        ecg_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()


        os.makedirs(ecg_path.parent, exist_ok=True)
        #ecg_image.save(ecg_path)


        with open(ecg_path, "wb") as f:
        #f.write(image_bytes.getbuffer())
            f.write(image_bytes)

        save_df = pd.DataFrame({
            'png_path': [ecg_path]
        })

        df_ecg = {
            'test': save_df
        }
        ecg_image = np.array(ecg_image)
        original_ecg_shape = ecg_image.shape

        for c in range(3):
            channel = ecg_image[..., c]
            min_val = channel.min()
            max_val = channel.max()
            ecg_image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) * 255

        ecg_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        ecg_image = ecg_transform(image=ecg_image)['image']
        st.success(f'Successfully uploaded ECG image.')
        

    else:
        st.write('No ECG file detected.')
        
    
    if age < 40:
        age_40 = 1
        age_40_60 = 0
        age_60_80 = 0
        age_80 = 0
    elif age > 80:
        age_40 = 0
        age_40_60 = 0
        age_60_80 = 0
        age_80 = 1
    elif age >= 40 and age <= 60:
        age_40 = 0
        age_40_60 = 1
        age_60_80 = 0
        age_80 = 0
    elif age > 60 and age <= 80:
        age_40 = 0
        age_40_60 = 0
        age_60_80 = 1
        age_80 = 0

    if gender == "Male":
        male = 1
        female = 0
    elif gender == "Female":
        male = 0
        female = 1
    else:
        male = 0
        female = 0
    
    tab_feats = {
        'F': female,
        'M': male,
        '40': age_40,
        '40-60': age_40_60,
        '60-80': age_60_80,
        '80': age_80,
        'Stroke/TIA_Base (0=no; 1=yes)': stroke,
        'VT_Base (0=no; 1=yes)': vt,
        'Creatinine_Baseline': creatinine,
        'Hyperlipidemia_Base (0=no; 1=yes)': hyperlipidemia,
        'Afib_Base (0=no; 1=yes)': afib,
        'HT_Baseline (0=no; 1=yes)': hypertension,
        'COPD Hx (0=no; 1=yes)': copd,
        'DBT_Base (0=no; 1=yes)': diabetes
    }
    df = pd.DataFrame({k: [v] for k, v in tab_feats.items()})


    df.replace('Yes', 1, inplace=True)
    df.replace('No', 0, inplace=True)
    df.replace('Unknown', -1, inplace=True)


    df = df[['F' ,'M', '40', '40-60', '60-80', '80', 'Stroke/TIA_Base (0=no; 1=yes)', 'VT_Base (0=no; 1=yes)', 'Creatinine_Baseline',
    'Hyperlipidemia_Base (0=no; 1=yes)', 'Afib_Base (0=no; 1=yes)', 'HT_Baseline (0=no; 1=yes)', 'COPD Hx (0=no; 1=yes)' ,'DBT_Base (0=no; 1=yes)']]


    if st.button('Get Predictions') and ecg_file is not None:


        # ECG MODEL
        with st.spinner('Running Model...'):
            ecg_model_file = './models/ecg_only.pth'
            
            results, ecg_dataloader = inference.get_preds(ecg_model_file, df_ecg, ecg='ecg')
            prob = results['score'].mean()

                    
            #ecg_gradcam = grad_cam2.gen_ecg_gradcam(ecg_model_file, ecg_dataloader, original_image)
            #w, h = original_ecg_shape[:2]

            #ecg_gradcam = ecg_gradcam.resize((h,w))
            st.write('----------------------------------------------------------')
            st.subheader(f"ECG-Based MACE Risk: {prob:.4f}")

            #st.image(ecg_gradcam, caption="ECG GradCAM", use_container_width=True)
            
            #print('Displaying GradCAM')

            # temp_path = Path(f"./temp")
            # if temp_path.exists() and temp_path.is_dir():
            #     try:    
            #         shutil.rmtree(temp_path)
                    
            #         #print(f"Deleted temp file: {dicom_path}")
            #     except Exception as e:
            #         print(f"Could not delete temp folder: {e}")
        
            #demo_model = joblib.load("./models/tabular_ICI_model.pkl")
            #prediction = demo_model.predict_proba(df)
            #st.subheader(f"Demographics-Based MACE Risk: {prediction[0][1]:.2f}")

            fusion_model = model_base.ECGFusionModel(n_features=14)
            old_model = torch.load("./models/fusion.pth", weights_only=False, map_location="cpu")

            old_state = old_model.state_dict()

            fusion_model.load_state_dict(old_state, strict=False)

            test_dataset = data_utils.test_ici_dataloader(df_ecg, df, 1, 'transform', 1)

            

            # # pre ECG data
            # image_data = Image.open(ecg_path).convert('RGB')
            # image = np.array(image_data)
            

            # for c in range(3):
            #     channel = image[..., c]
            #     min_val = channel.min()
            #     max_val = channel.max()
            #     image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) #* 255

            # transform = A.Compose([
            #     A.Resize(224, 224),
            #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     ToTensorV2(),
            # ])
            # frame = transform(image=image)['image']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # feats = df.iloc[0, :].values.astype(np.float32)
            # feats_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

            fusion_model = fusion_model.to(device)
            fusion_model.eval()

            # frame = frame.unsqueeze(0)
            # frame = frame.to(device)
            # feats_tensor = feats_tensor.to(device)

            for frame, feats_tensor in tqdm(test_dataset['test']):

                frame = frame.to(device)
                feats_tensor = feats_tensor.to(device)

                with torch.no_grad():
                    output = fusion_model(feats_tensor, frame)

                softmax = nn.Softmax()
                scores = softmax(output)

            # with torch.no_grad():
            #     output = fusion_model(feats_tensor, frame)
            
            fusion_prob = scores[0, 1].item()
            st.subheader(f"Fusion MACE Risk: {fusion_prob:.4f}")

            if fusion_prob > 0.4763:
                st.markdown(
                    "<h3 style='color:red;'>High Risk of MACE within 1 year</h3>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<h3 style='color:black;'>Low Risk of MACE within 1 year</h3>",
                    unsafe_allow_html=True
                )


        temp_path = Path(f"./temp_ecg")
        if temp_path.exists() and temp_path.is_dir():
            try:    
                shutil.rmtree(temp_path)
                
                #print(f"Deleted temp file: {dicom_path}")
            except Exception as e:
                print(f"Could not delete temp folder: {e}")


except URLError as e:
    st.error(
        """
        **This demo requires internet access.**

        Connection error: %s
    """
        % e.reason
    )