# !/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import pickle
import pandas as pd
import numpy as np
from src import training_base, dataloader_base
from src_v1 import data_utils
from torchvision import models
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


def model_densenet(num_classes):
    base_model =  models.densenet121(pretrained=True)
    base_model.classifier = nn.Sequential(
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, num_classes)  # <- Change to your desired number of output classes
    )
    return base_model

def get_preds(model_path, df, ecg=None):

    params = {
        'workers': 1,
        'num_classes': 2,
        'ngpu': 1,
        'batch_size': 1,  
        'task': 'clf',
        'multitask': False,
        'model_type': 'fusion', # fusion, ecg, echo
        'focused_leads': ['II', 'V1'],
        'input_size': 224
    }


    params['dataframe_pkl'] = df

    params['view'] = 'AP4'  

    params['model_name'] = 'resnext101'

    # update parameters with the input_size of the model
    params['input_size'] = 224

    # Detect if we have a GPU available
    cuda_num = 0
    device = torch.device("cuda:{}".format(cuda_num) if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")

    if ecg == 'ecg':
        try:
            model = model_densenet(4)
            state_dict = torch.load(model_path, weights_only=True, map_location=device)
            model.load_state_dict(state_dict)
        except:
            model = torch.load(model_path, map_location=device)

        dataloaders_dict = data_utils.test_loader(df, params['batch_size'], params['workers'])
    else:
        model = torch.load(model_path, map_location=device)
        dataloaders_dict = dataloader_base.ecg_dataloader(params, mode='test')

    model = model.to(device)

    print('Generating predictions...')

    score_nc = training_base.model_predict(device, model, dataloaders_dict, params)

    results = pd.DataFrame(columns=['score'])
    results.score = score_nc

    print('Inference complete.')

    return results, dataloaders_dict
    
