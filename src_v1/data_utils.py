import re
import os
import torch 
import random
import pickle
#import pydicom 
import numpy as np 
import pandas as pd 
#import neurokit2 as nk
from PIL import Image
from skimage import exposure
import torch.nn.functional as F
from collections import defaultdict
from itertools import cycle
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MinMaxScaler
#import pydicom.pixel_data_handlers.util as util
from torch.utils.data import Dataset, DataLoader, Sampler


def echo_dataloader(pkl_file, batch_size, transform, num_workers):
        
    train_dataset = VideoDataset(pkl_file=pkl_file, phase='train', transform=transform)

    #batch_sampler = BalancedBatchSampler(train_dataset, batch_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ])

    val_dataset = VideoDataset(pkl_file=pkl_file, phase='val', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader}

def test_loader(pkl_file, batch_size, num_workers):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ])

    test_dataset = VideoDataset(pkl_file=pkl_file, phase='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'test': test_dataloader}

def ecg_dataloader(pkl_file, batch_size, transform, num_workers):
    train_dataset = ECGDataset_1D(pkl_file=pkl_file, phase='train')
    val_dataset = ECGDataset_1D(pkl_file=pkl_file, phase='val')
    test_dataset = ECGDataset_1D(pkl_file=pkl_file, phase='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

class VideoDataset(Dataset):
    def __init__(self, pkl_file, phase, transform):
        #data = pd.read_pickle(pkl_file)
        data = pkl_file[phase]
        #data = data.dropna(subset=['start_frame'])
        data.reset_index(drop=True, inplace=True)
        self.data = data
        self.transform = transform
        #self.histogram_ref = histogram_ref

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row['png_path']
        #label = row['Diagnosis']

        image = Image.open(file_path).convert('RGB')

        image = np.array(image)

        for c in range(3):
            channel = image[..., c]
            min_val = channel.min()
            max_val = channel.max()
            image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) * 255

        frame = self.transform(image=image)['image']

        #pt_id = row['MRN']

        return frame #, label, pt_id

def load_ecg_signals(focused_leads, file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        nums = data.values()
        all_leads = [data[key] for key in focused_leads]
    return torch.tensor(all_leads, dtype=torch.float32)

class ECGDataset_1D(Dataset):


    def __init__(self, pkl_file, phase):

        data = pd.read_csv(pkl_file)
        data = data[data['split'] == phase]
        data.reset_index(drop=True, inplace=True)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = row['Diagnosis']
        pt_id = row['MRN']


        ecg_file = row['File_Name']
        ecg_signals = load_ecg_signals(['V1', 'II'], ecg_file)
        ecg_signals = ecg_signals[:,:2500]


        scaled_tensor = torch.empty_like(ecg_signals)

        # filter x using PT function
        for i in range(ecg_signals.shape[0]):
            
            # filtering each individual channel/lead signal
            ch1 = nk.ecg_clean(ecg_signals[i,:], sampling_rate=500, method="pantompkins1985") # pre processing ECG signal
            #ch1 = x[i,:]

            # save channel signal into new tensor to be normalized
            scaled_tensor[i,:] = torch.tensor(ch1)

        # min/max normalization 
        scaler = MinMaxScaler()

        for i in range(ecg_signals.shape[0]):
            lead = scaled_tensor[i,:].reshape(-1,1)

            scaled_lead = scaler.fit_transform(lead)
            scaled_tensor[i,:] = torch.tensor(scaled_lead.flatten(), dtype=torch.float32)


        sample = (scaled_tensor, label, pt_id)

        return sample

def fusion_dataloader(ecg_pkl_file, echo_pkl_file, batch_size, ecg_transform, hist, num_workers):

    train_transform = transforms.Compose([
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dataset = FusionDataset(pd_file=echo_pkl_file, phase='train', ecg_data=ecg_pkl_file, ecg_transform=ecg_transform, echo_transform=train_transform, hist=hist)
    val_dataset = FusionDataset(pd_file=echo_pkl_file, phase='val', ecg_data=ecg_pkl_file, ecg_transform=ecg_transform, echo_transform=eval_transform, hist=hist)
    test_dataset = FusionDataset(pd_file=echo_pkl_file, phase='test', ecg_data=ecg_pkl_file, ecg_transform=ecg_transform, echo_transform=eval_transform, hist=hist)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

class FusionDataset(Dataset):
    """Custom HeadCT dataset with just class labels
    note:
        since it's a custom dataset we need to use:
        for i, data in enumerate(train_loader, 0):
        to enumerate over the train loaders, and the i is the ith train_loader and the data is the sample we generate
    """

    def __init__(self, pd_file, phase, ecg_data, ecg_transform, echo_transform, hist):

        self.df_ecg = pd.read_csv(ecg_data) # dataframe
        
        df = pd.read_pickle(pd_file)
        df = df[phase] # only want train part of dict()
        df_view = df[df['view'] == 'AP4']
        df_view.reset_index(inplace=True, drop=True)
        self.label = df_view
        self.hist = hist

        self.ecg_transform = ecg_transform
        self.echo_transform = echo_transform

    def __len__(self):
        return len(self.label)
        #return len(self.df_ecg)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image_path = os.path.join(self.label['path'].iloc[idx])
        image = Image.open(image_path).convert('L')  # convert to grayscale

        hist_norm = self.hist
        image = image.resize((442,442), resample=Image.BILINEAR)

        image = np.array(image)
        matched_image = exposure.match_histograms(image, hist_norm)
        matched_image = Image.fromarray(np.uint8(matched_image))

        image = matched_image.convert('RGB')
        
        label = self.label['label'].iloc[idx]
        resid = self.label['pt_id'].iloc[idx]

        if self.echo_transform is not None:
            image = self.echo_transform(image)


        if len(resid) < 10 or resid.startswith('Amyloidosis'):
            pt_id = resid.split('_')[1]
            if resid.startswith('NC'):
                pt_id = 'RESID' + pt_id[1:]

            else:
                if pt_id.startswith('R00'):
                    pt_id = 'R' + pt_id[3:]
                elif pt_id.startswith('R0'):
                    pt_id = 'R' + pt_id[2:]
        else:
            pt_id = resid
        
        row_ecg = self.df_ecg[(self.df_ecg['Echo_File'] == pt_id) & (self.df_ecg['Diagnosis'] == label)].index[0]
        ecg_file_path = self.df_ecg['png_path'][row_ecg]

        ecg_image = Image.open(ecg_file_path).convert('RGB')

        ecg_image = np.array(ecg_image)

        for c in range(3):
            channel = ecg_image[..., c]
            min_val = channel.min()
            max_val = channel.max()
            ecg_image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) * 255

        ecg_image = self.ecg_transform(image=ecg_image)['image']


        sample = (image, ecg_image, label, resid)

        return sample



    
