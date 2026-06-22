import copy
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

def model_predict(device, model, dataloader, params):
    """
    Uses the given model to predict on the dataloader data and output the true and predicted labels
    """
    # setting model to evaluate mode
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    score_1 = []


    pt_ids = []
    try: 
        loader = dataloader['test']
    except:
        loader = dataloader

    for inputs in tqdm(loader):
        inputs = inputs.to(device)

        with torch.set_grad_enabled(False):

            outputs = model(inputs)
            softmax = nn.Softmax()
            score = softmax(outputs)

            score_1_batch = score[:, 1]
            score_1.extend(score_1_batch.tolist())

    return score_1 

