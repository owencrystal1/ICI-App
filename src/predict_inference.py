import copy
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def fusion_predict(device, model, dataloader, params):
    """
    Uses the given model to predict on the dataloader data and output the true and predicted labels
    """
    # setting model to evaluate mode
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    score_0 = []
    score_1 = []
    score_2 = []
    pt_ids = []

    for inputs, ecg_features, IDs in tqdm(dataloader):
        inputs = inputs.to(device)
        ecg_features = ecg_features.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs, ecg_features)

            softmax = nn.Softmax()
            score = softmax(outputs)
                
            _, preds = torch.max(score, 1)
            score_0_batch = score[:, 0]
            score_1_batch = score[:, 1]
            score_2_batch = score[:, 2]
            
            pt_ids.extend(IDs)
            score_0.extend(score_0_batch.tolist())
            score_1.extend(score_1_batch.tolist())
            score_2.extend(score_2_batch.tolist())

    results = pd.DataFrame(columns=['pt_id', 'AMY', 'HCM', 'HTN'])
    results.pt_id = pt_ids
    results.AMY = score_0
    results.HCM = score_1
    results.HTN = score_2

    df_avg = results.groupby('pt_id').agg({
        'AMY': 'mean',
        'HCM': 'mean',
        'HTN': 'mean',
    }).reset_index()

    df_avg.to_csv(params['results_path'] + 'fusion_predictions.csv')

def echo_predict(device, model, dataloader, params):
    """
    Uses the given model to predict on the dataloader data and output the true and predicted labels
    """
    # setting model to evaluate mode
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    score_0 = []
    score_1 = []
    score_2 = []
    pt_ids = []

    for inputs, IDs in tqdm(dataloader):
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            softmax = nn.Softmax()
            score = softmax(outputs)
                
            _, preds = torch.max(score, 1)
            score_0_batch = score[:, 0]
            score_1_batch = score[:, 1]
            score_2_batch = score[:, 2]
            
            pt_ids.extend(IDs)
            score_0.extend(score_0_batch.tolist())
            score_1.extend(score_1_batch.tolist())
            score_2.extend(score_2_batch.tolist())

    results = pd.DataFrame(columns=['pt_id', 'AMY', 'HCM', 'HTN'])
    results.pt_id = pt_ids
    results.AMY = score_0
    results.HCM = score_1
    results.HTN = score_2

    df_avg = results.groupby('pt_id').agg({
        'AMY': 'mean',
        'HCM': 'mean',
        'HTN': 'mean',
    }).reset_index()

    df_avg.to_csv(params['results_path'] + '/echo_predictions.csv')

def ecg_predict(device, model, dataloader, params):
    """
    Uses the given model to predict on the dataloader data and output the true and predicted labels
    """
    # setting model to evaluate mode
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    score_0 = []
    score_1 = []
    score_2 = []
    pt_ids = []

    for inputs, IDs in tqdm(dataloader):
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            softmax = nn.Softmax()
            score = softmax(outputs)
                
            _, preds = torch.max(score, 1)
            score_0_batch = score[:, 0]
            score_1_batch = score[:, 1]
            score_2_batch = score[:, 2]
            
            pt_ids.extend(IDs)
            score_0.extend(score_0_batch.tolist())
            score_1.extend(score_1_batch.tolist())
            score_2.extend(score_2_batch.tolist())

    results = pd.DataFrame(columns=['pt_id', 'AMY', 'HCM', 'HTN'])
    results.pt_id = pt_ids
    results.AMY = score_0
    results.HCM = score_1
    results.HTN = score_2

    results.to_csv(params['results_path'] + '/ecg_predictions.csv')