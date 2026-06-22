# INFERENCE SCRIPT #


import os
import sys
import torch
import pickle
import pandas as pd
import predict_inference
import dataloader_base_inf
import preprocessing_pipeline
import model_base
from tqdm import tqdm
import torch.nn as nn
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--echo_csv', type=str, help='Input path to echocardiography data (csv file)') 
parser.add_argument('--ecg_csv', type=str, help='Input path to ECG data (csv file)')
parser.add_argument('--fusion_csv', type=str, help='Input path to mapped ECG/Echo data (csv file)')
parser.add_argument('--echo_png_path', type=str, help='Input path to save png Echo files')
args = parser.parse_args()
 

params = {
    'workers': 8,
    'num_classes': 3,
    'ngpu': 3,
    'batch_size': 32, 
    'model_type': 'fusion', # fusion, ecg, echo
    'focused_leads': ['II', 'V1'],
    'input_size': 224
}

if __name__ == "__main__":

    test_num = 1 # [1, 2, 3, 4, 5, 6, 7]
    cuda_num = 3
    view = 'AP4' 

    # Gokul providing echo and ecg csvs:
    params['echo_csv'] = pd.read_csv(args.echo_csv)
    params['fusion_csv'] = pd.read_csv(args.fusion_csv)
    params['ecg_dataframe'] = pd.read_csv(args.ecg_csv)


    # input arguments 
    params['png_path'] = args.echo_png_path
    params['results_path'] = './test_results/'


    # put paths to dicom files into list
    dcm_list = (params['echo_csv']['path'])


    echo_mrns = []
    echo_dates = []
    echo_acc_nums = []
    echo_paths = []
    binary_array = np.load('mask_crop.npy')
    # Step 1: Convert DICOM files to PNGs and extract conical ROI from echo images in echo csv

    for i in tqdm(range(len(dcm_list)), desc='Converting DICOM to PNG for Echo data'):

        # convert DICOM file into png files containing conical ROIs (saved in png path)
        pt_ids, dates, acc_num, paths = preprocessing_pipeline.dicom2roi(dcm_list[i], params['png_path'], binary_array)

        # store patient IDs
        echo_mrns  = echo_mrns + pt_ids
        echo_dates  = echo_dates + dates
        echo_acc_nums = echo_acc_nums + acc_num
        echo_paths = echo_paths + paths
    
    print('ROIs extracted from', i, 'DICOM files and saved as PNG files.')
        
    # create df with paths and IDs to save as pickle file (if needed to be saved)
    echo_data = pd.DataFrame({
        'path': echo_paths,
        'PatientID': echo_mrns,
        'AccessionNumber': echo_acc_nums,
        'AcquisitionDate': echo_dates
        })

    params['echo_dataframe'] = echo_data

    print('Paths to Echo ROI .png files saved into echo-only csv file with meta-data.')


    ###############################################

    # Step 2: Convert DICOM files to PNGs and extract conical ROI from echo images in fusion csv

    fusion_dcm_list = (params['fusion_csv']['echo_path'])

    fusion_echo_mrns = []
    fusion_echo_dates = []
    fusion_echo_acc_nums = []
    fusion_echo_paths = []

    for i in tqdm(range(len(fusion_dcm_list)), desc='Converting DICOM to PNG for Fusion data'):

        # convert DICOM file into png files containing conical ROIs (saved in png path)
        pt_ids, dates, acc_num, paths = preprocessing_pipeline.dicom2roi(fusion_dcm_list[i], params['png_path'], binary_array)

        # store patient IDs
        fusion_echo_mrns = fusion_echo_mrns + pt_ids
        fusion_echo_dates = fusion_echo_dates + dates
        fusion_echo_acc_nums = fusion_echo_acc_nums + acc_num
        fusion_echo_paths = fusion_echo_paths + paths
    
    print('ROIs extracted from', i, 'DICOM files and saved as PNG files.')

    # create df with paths and IDs to save as pickle file (if needed to be saved)
    fusion_data = pd.DataFrame({
        'path': fusion_echo_paths,
        'PatientID': fusion_echo_mrns,
        'AccessionNumber': fusion_echo_acc_nums,
        'AcquisitionDate': fusion_echo_dates,
        'ecg_path':,
        'ecg_date':
        })

    # Step 3: Map path to ECG pickle file to each Echo png file

    for i in tqdm(range(len(fusion_data)),desc='Mapping ECG to Echo PNG'):
        mrn = fusion_data['PatientID'][i]

        # find which row in original fusion csv where this mrn occurs
        row_ecg = params['fusion_csv'][(params['fusion_csv']['PatientID'] == mrn)].index

        # save corresponding path to ECG pickle file and ecg date in same row - map ECG to Echo PNG
        fusion_data['ecg_path'][i] = params['fusion_csv']['ecg_path'][row_ecg]
        fusion_data['ecg_date'][i] = params['fusion_csv']['ecg_date'][row_ecg]

    params['fusion_dataframe'] = fusion_data

    print('Paths to Echo ROI .png files and corresponding ECG pickle file paths saved into csv with meta-data.')
    
    print('Loading data for inference...')

    # use same data loader for each model type, adjust testing according to model to see what data to load in
    ecg_dataloaders_dict = dataloader_base_inf.ecg_dataloader(params, mode='ecg')
    echo_dataloaders_dict = dataloader_base_inf.ecg_dataloader(params, mode='echo')
    fusion_dataloaders_dict = dataloader_base_inf.ecg_dataloader(params, mode='fusion')

    print('Dataloaders created.')

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')  # Use GPU
        print("GPU is available. Using CUDA.")
    else:
        device = torch.device('cpu')   # Fallback to CPU
        print("GPU is not available. Using CPU.")

    ecg_model_path = './models/ecg_cnn.pth'
    echo_model_path = './models/ap4_echo_cnn.pth'
    fusion_model_path = './models/fusion_model.pth'

    ecg_model_arch = model_base.ECG_CNN(num_conv_layers=6, first_layer_filters=128, kernel_size=7) 
    ecg_model_arch.load_state_dict(torch.load(ecg_model_path))
    print('ECG model loaded.')

    echo_model = torch.load(echo_model_path)
    print('Echo model loaded.')

    fusion_model = torch.load(fusion_model_path)
    print('Fusion model loaded.')

    print('Making predictions...')
    predict_inference.echo_predict(device, echo_model, echo_dataloaders_dict, params)
    predict_inference.ecg_predict(device, ecg_model_arch, ecg_dataloaders_dict, params)
    predict_inference.fusion_predict(device, fusion_model, fusion_dataloaders_dict, params)
    print('Predictions saved.')

