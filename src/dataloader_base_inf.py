import os
import random
import pickle
import pandas as pd
import neurokit2 as nk
from PIL import Image
from cardiac_echo_processes import *
from sklearn.preprocessing import MinMaxScaler
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler


def ecg_dataloader(params, mode):
    dataset = load_png_wID(params=params, mode=mode,
                        transform=transforms.Compose([
                            transforms.Resize((params['input_size'], params['input_size'])),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]))

    custom_dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                                                num_workers=params['workers'])

    return custom_dataloader

def load_ecg_signals(directory, focused_leads):
        with open(directory, 'rb') as f:
            data = pickle.load(f)
            nums = data.values()
            all_leads = [data[key] for key in focused_leads]
        return torch.tensor(all_leads, dtype=torch.float32)


class load_png_wID(Dataset):


    def __init__(self, params, mode, quadrant=0, view=None, transform=None):

        self.model_type = mode
        self.params = params
        self.transform = transform


        if self.model_type == 'echo':
            self.df = self.params['echo_dataframe']
        elif self.model_type == 'ecg':
            self.df = self.params['ecg_dataframe']
        elif self.model_type == 'fusion':
            self.df = self.params['fusion_dataframe']


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.model_type == 'echo':
            

            # get image
            image_path = os.path.join(self.df['path'].iloc[idx])
            image = Image.open(image_path).convert('L')  # convert to grayscale

            # histogram matching
            hist_norm = np.array(pickle.load(open('normalized_train_array.pkl', 'rb'))).reshape(1, 195239) # reshape to whatever size the vector is
            image = image.resize((442,442), resample=Image.BILINEAR)

            image = np.array(image)
            matched_image = exposure.match_histograms(image, hist_norm)
            matched_image = Image.fromarray(np.uint8(matched_image))

            image = matched_image.convert('RGB')
            
            self.pt_id = self.df['PatientID'].iloc[idx]

            # apply transforms
            if self.transform is not None:
                image = self.transform(image)

            sample = (image, self.pt_id)
        
        elif self.model_type == 'ecg':

            self.pt_id = self.df['MRN'].iloc[idx]

            ecg_file_path = self.df['File_Name'].iloc[idx]
            ecg_signals = load_ecg_signals(ecg_file_path, self.params['focused_leads'])
            ecg_signals = ecg_signals[:,:2500]


            scaled_tensor = torch.empty_like(ecg_signals)

            # filter x using PT function
            for i in range(ecg_signals.shape[0]):
                
                # filtering each individual channel/lead signal
                ch1 = nk.ecg_clean(ecg_signals[i,:], sampling_rate=500, method="pantompkins1985") # pre processing ECG signal

                # save channel signal into new tensor to be normalized
                scaled_tensor[i,:] = torch.tensor(ch1)

            # min/max normalization 
            scaler = MinMaxScaler()

            for i in range(ecg_signals.shape[0]):
                lead = scaled_tensor[i,:].reshape(-1,1)

                scaled_lead = scaler.fit_transform(lead)
                scaled_tensor[i,:] = torch.tensor(scaled_lead.flatten(), dtype=torch.float32)
                noise = np.random.normal(0, 0.1, scaled_tensor[i,:].shape)

                # Add noise to the tensor
                scaled_tensor[i,:] = scaled_tensor[i,:] + noise
            
            sample = (scaled_tensor, self.pt_id)
        
        elif self.model_type == 'fusion':

            self.pt_id = self.df['PatientID'].iloc[idx]

            # get image
            image_path = os.path.join(self.df['echo_path'].iloc[idx])
            image = Image.open(image_path).convert('L')  # convert to grayscale

            # histogram matching
            hist_norm = np.array(pickle.load(open('normalized_train_array.pkl', 'rb'))).reshape(1, 195239) # reshape to whatever size the vector is
            image = image.resize((442,442), resample=Image.BILINEAR)

            image = np.array(image)
            matched_image = exposure.match_histograms(image, hist_norm)
            matched_image = Image.fromarray(np.uint8(matched_image))

            image = matched_image.convert('RGB')
            
            if self.transform is not None:
                image = self.transform(image)

            ecg_file_path = self.df['ecg_path'].iloc[idx]
            ecg_signals = load_ecg_signals(ecg_file_path, self.params['focused_leads'])
            ecg_signals = ecg_signals[:,:2500]

            scaled_tensor = torch.empty_like(ecg_signals)

            # filter ECG signal, each lead separately
            for i in range(ecg_signals.shape[0]):
                
                # filtering each individual channel/lead signal
                ch1 = nk.ecg_clean(ecg_signals[i,:], sampling_rate=500, method="pantompkins1985") # pre processing ECG signal

                # save channel signal into new tensor to be normalized
                scaled_tensor[i,:] = torch.tensor(ch1)

            # min/max normalization 
            scaler = MinMaxScaler()

            for i in range(ecg_signals.shape[0]):
                lead = scaled_tensor[i,:].reshape(-1,1)

                scaled_lead = scaler.fit_transform(lead)
                scaled_tensor[i,:] = torch.tensor(scaled_lead.flatten(), dtype=torch.float32)
                noise = np.random.normal(0, 0.1, scaled_tensor[i,:].shape)

                # Add noise to the tensor
                scaled_tensor[i,:] = scaled_tensor[i,:] + noise

            sample = (image, scaled_tensor, self.pt_id) 

        return sample

