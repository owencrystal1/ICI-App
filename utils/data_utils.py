import re
import os
import cv2
import torch 
import random
import pickle
#import pydicom 
import numpy as np 
import pandas as pd 
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
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler


def crop_red_grid(img_array, pad_ratio=0.05):
    """
    Detect red ECG grid and crop to bounding box.
    Args:
        img_array (np.ndarray): RGB image, dtype=uint8
        pad_ratio (float): extra margin around crop
    Returns:
        np.ndarray: cropped RGB image
    """
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    coords = cv2.findNonZero(mask)
    if coords is not None:
        x,y,w,h = cv2.boundingRect(coords)
        pad = int(pad_ratio * max(w,h))
        x = max(x-pad, 0)
        y = max(y-pad, 0)
        w = min(w+2*pad, img_array.shape[1]-x)
        h = min(h+2*pad, img_array.shape[0]-y)
        cropped = img_array[y:y+h, x:x+w]
    else:
        cropped = img_array  # fallback

    return cropped

class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, drop_last=True):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Automatically find unique class labels
        self.classes = sorted(set(labels))
        self.num_classes = len(self.classes)

        # Group indices by class
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        # Ensure every class has at least one sample
        for cls in self.classes:
            if len(self.class_to_indices[cls]) == 0:
                raise ValueError(f"Class {cls} has no samples.")

        # Shuffle each class list initially
        for cls_indices in self.class_to_indices.values():
            random.shuffle(cls_indices)

    def __iter__(self):
        class_cursors = {cls: 0 for cls in self.classes}
        all_indices = list(range(len(self.labels)))
        random.shuffle(all_indices)

        batch = []
        samples_yielded = 0
        max_samples = len(self.labels)  # total samples per epoch

        while samples_yielded < max_samples:
            # Add one sample per class
            for cls in self.classes:
                indices = self.class_to_indices[cls]
                cursor = class_cursors[cls]

                if cursor >= len(indices):
                    random.shuffle(indices)
                    class_cursors[cls] = 0
                    cursor = 0

                batch.append(indices[cursor])
                class_cursors[cls] += 1

            # Fill rest of batch randomly
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                candidates = [i for i in all_indices if i not in batch]
                if len(candidates) < remaining:
                    random.shuffle(all_indices)
                    candidates = [i for i in all_indices if i not in batch]
                batch.extend(random.sample(candidates, min(remaining, len(candidates))))

            if len(batch) == self.batch_size:
                yield batch
                samples_yielded += self.batch_size
                batch = []
            else:
                if not self.drop_last and batch:
                    yield batch
                    samples_yielded += len(batch)
                break

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        else:
            return (len(self.labels) + self.batch_size - 1) // self.batch_size

from torch.utils.data import WeightedRandomSampler

def makeWeightedSampler(dataset):
    """makes a weighted sampler based on the input dataset"""
    # Compute class weight each class should get the same weight based on its balance
    temp_df = dataset.df
    # get the count of each label
    benCount = temp_df.labelMACE_1yr[temp_df.labelMACE_1yr == 0].count()
    malCount = temp_df.labelMACE_1yr[temp_df.labelMACE_1yr == 1].count()
    # get weight of each label
    benWeight = 1. / benCount
    malWeight = 1. / malCount
    #print('sliceweights: ben:{}, mal:{}'.format(benWeight, malWeight))
    # make a tensor of the weights
    sample_weights = []
    for label in temp_df.labelMACE_1yr:
        if label == 0:
            sample_weights.append(benWeight)
        else:
            sample_weights.append(malWeight)
    sample_weights = torch.tensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights))

class ResizeAndPadTo224(A.ImageOnlyTransform):
    def __init__(self, fill=0, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.target_size = 224
        self.fill = fill

    def apply(self, image, **params):
        h, w = image.shape[:2]

        # Resize while preserving aspect ratio
        scale = min(self.target_size / h, self.target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Calculate padding
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        # Pad image
        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=[self.fill] * image.shape[2] if len(image.shape) == 3 else self.fill
        )

        return padded

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.labels = dataset.df.labelMACE_1yr
        self.batch_size = batch_size
        self.class_indices = defaultdict(list)

        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)

        self.num_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.num_classes
        assert self.samples_per_class > 0, "Batch size must be at least equal to the number of classes"

    def __iter__(self):
        # Make sure we can cycle through all classes multiple times
        class_iterators = {
            cls: cycle(indices) for cls, indices in self.class_indices.items()
        }

        # Estimate how many batches we can generate
        total_samples = len(self.labels)
        num_batches = total_samples // self.batch_size

        for _ in range(num_batches):
            batch = []
            for cls in self.class_indices:
                cls_indices = class_iterators[cls]
                batch.extend([next(cls_indices) for _ in range(self.samples_per_class)])

            # If the batch is too small (due to integer division), pad randomly
            while len(batch) < self.batch_size:
                random_cls = random.choice(list(self.class_indices.keys()))
                batch.append(next(class_iterators[random_cls]))

            random.shuffle(batch)
            yield batch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def echo_dataloader(pkl_file, batch_size, transform, num_workers):
        
    train_dataset = VideoDataset(pkl_file=pkl_file, phase='train', transform=transform)

    df = pd.read_csv(pkl_file)
    df.dropna(inplace=True)
    print('total:', len(df))
    df = df[df['split'] == 'train']
    df.reset_index(inplace=True, drop=True)
    labels = df["label"].values  # shape [N]
    print('train', len(labels))
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=sampler, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    transform = A.Compose([
        A.Resize(224, 224),
        #A.ToFloat(max_value=255.0),
        #ResizeAndPadTo224(fill=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ])

    val_dataset = VideoDataset(pkl_file=pkl_file, phase='val', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    test_dataset = VideoDataset(pkl_file=pkl_file, phase='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

def test_loader(pkl_file, batch_size, num_workers, transform):

    # transform = A.Compose([
    #     A.Resize(224, 224),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2()
    #     ])
    # transform = A.Compose([
    #     A.Resize(224, 224),
    #     #A.RandomResizedCrop(height=224, width=224, scale=(scale, scale), p=1.0),  # replaces Resize
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ToTensorV2()
    # ])
        
    test_dataset = VideoDataset(pkl_file=pkl_file, phase='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'test': test_dataloader}

def ecg_dataloader(pkl_file, batch_size, transform, num_workers):
    train_dataset = ECGDataset_1D(pkl_file=pkl_file, phase='train')
    val_dataset = ECGDataset_1D(pkl_file=pkl_file, phase='val')
    test_dataset = ECGDataset_1D(pkl_file=pkl_file, phase='test')

    df = pd.read_csv(pkl_file)
    df.dropna(inplace=True)
    print('total:', len(df))
    df = df[df['split'] == 'train']
    df.reset_index(inplace=True, drop=True)
    labels = df["label"].values  # shape [N]
    print('train', len(labels))
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

class VideoDataset(Dataset):
    def __init__(self, pkl_file, phase, transform):
        data = pd.read_csv(pkl_file)
        data = data[data['split'] == phase]
        #data = data[data['MRN'] == 12742526]
        #print(type(data['MRN'][20]))
        data = data.dropna(subset=['ECGPngPath'])
        data.reset_index(drop=True, inplace=True)
        self.data = data
        self.transform = transform
        #self.histogram_ref = histogram_ref

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        #file_path = row['png_path']
        file_path = row['ECGPngPath']
        file_path = file_path.replace("ECGEcho/Gradient/PulledDatav2/data", "/media/Datacenter_storage/Owen/Gradient_Project/NewData")
        file_path = file_path.replace("home/owen", "media")
        #label = row['Diagnosis']
        label = row['label']

        image = Image.open(file_path).convert('RGB')

        image = np.array(image)

        for c in range(3):
            channel = image[..., c]
            min_val = channel.min()
            max_val = channel.max()
            image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) * 255

        frame = self.transform(image=image)['image']

        pt_id = row['MRN']

        return frame, label, pt_id

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

def fusion_dataloader(echo_pkl_file, batch_size, ecg_transform, hist, num_workers, processor=None):
    
    if processor is not None:
    
        train_transform = transforms.Compose([
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                #transforms.Resize((224, 224)),
                #transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # train_transform = transforms.Compose([
        #         transforms.RandomRotation(40),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         AddGaussianNoise(0., 0.05),   # <--- added here
        #         transforms.Normalize([0.485, 0.456, 0.406],
        #                             [0.229, 0.224, 0.225])
        #     ])


        eval_transform = None

        # eval_transform = A.Compose([
        #     A.Resize(224, 224),
        #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ToTensorV2()
        #     ])
    
    else:
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

    train_dataset = FusionDataset(pd_file=echo_pkl_file, phase='train', ecg_transform=ecg_transform, echo_transform=train_transform, hist=hist, processor=processor)
    val_dataset = FusionDataset(pd_file=echo_pkl_file, phase='val', ecg_transform=ecg_transform, echo_transform=eval_transform, hist=hist, processor=processor)
    test_dataset = FusionDataset(pd_file=echo_pkl_file, phase='test', ecg_transform=ecg_transform, echo_transform=eval_transform, hist=hist, processor=processor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

class FusionDataset(Dataset):
    """Custom HeadCT dataset with just class labels
    note:
        since it's a custom dataset we need to use:
        for i, data in enumerate(train_loader, 0):
        to enumerate over the train loaders, and the i is the ith train_loader and the data is the sample we generate
    """

    def __init__(self, pd_file, phase, ecg_transform, echo_transform, hist, processor=None):

        #self.df_ecg = pd.read_csv(ecg_data) # dataframe
        try:
            df = pd.read_pickle(pd_file)
        except:
            df = pd.read_csv(pd_file)

        #df = df[phase] # only want train part of dict()
        #df_view = df[df['view'] == 'AP4']
        #df = df[df['pt_id'] != 466637]
        df.dropna(subset='ECGPngPath', inplace=True)
        df_view = df[df['split'] == phase]
        df_view.reset_index(inplace=True, drop=True)
        self.label = df_view
        self.hist = hist

        self.ecg_transform = ecg_transform
        self.echo_transform = echo_transform

        self.processor = processor

    def __len__(self):
        return len(self.label)
        #return len(self.df_ecg)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.processor:
            image_path = os.path.join(self.label['png_path'].iloc[idx])
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if self.echo_transform is not None:
                image = self.echo_transform(image)
                
            matched_image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
            image = matched_image
        else:
            # get image
            image_path = os.path.join(self.label['png_path'].iloc[idx])
            image = Image.open(image_path).convert('L')  # convert to grayscale

            hist_norm = self.hist
            image = image.resize((442,442), resample=Image.BILINEAR)

            image = np.array(image)
            matched_image = exposure.match_histograms(image, hist_norm)
            matched_image = Image.fromarray(np.uint8(matched_image))

            image = matched_image.convert('RGB')
            if self.echo_transform is not None:
                #image = self.echo_transform(image=image)['image']
                image = self.echo_transform(image)
        #image = np.array(image)
        
        label = self.label['label'].iloc[idx]
        resid = self.label['MRN'].iloc[idx]

        # if len(resid) < 10 or resid.startswith('Amyloidosis'):
        #     pt_id = resid.split('_')[1]
        #     if resid.startswith('NC'):
        #         pt_id = 'RESID' + pt_id[1:]

        #     else:
        #         if pt_id.startswith('R00'):
        #             pt_id = 'R' + pt_id[3:]
        #         elif pt_id.startswith('R0'):
        #             pt_id = 'R' + pt_id[2:]
        # else:
        #     pt_id = resid
        
        pt_id = resid
        
        #row_ecg = self.df_ecg[(self.df_ecg['Echo_File'] == pt_id) & (self.df_ecg['Diagnosis'] == label)].index[0]

        #ecg_file_path = self.df_ecg['ecg_path'][row_ecg]
        try:
            ecg_file_path = os.path.join(self.label['ECGPngPath'].iloc[idx])
        except:
            print(self.label['ECGPngPath'].iloc[idx])
        ecg_file_path = ecg_file_path.replace("ECGEcho/Gradient/PulledDatav2/data", "/media/Datacenter_storage/Owen/Gradient_Project/NewData")
        #ecg_file_path = os.path.join(self.label['ecg_path'].iloc[idx])
        #ecg_file_path = ecg_file_path.replace("EKG", "EKG cropped")

        # /media/Datacenter_storage/Owen/Gradient_Project/NewData/ECG


        ecg_image = Image.open(ecg_file_path).convert('RGB')

        ecg_image = np.array(ecg_image)
        ecg_image = crop_red_grid(ecg_image)


        for c in range(3):
            channel = ecg_image[..., c]
            min_val = channel.min()
            max_val = channel.max()
            ecg_image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) * 255

        ecg_image = self.ecg_transform(image=ecg_image)['image']


        sample = (image, ecg_image, label, resid)

        return sample

def ici_dataloader(pkl_file, tab_feats, batch_size, transform, num_workers):

    train_dataset = ICIFusionDataset(df=pkl_file, phase='train', tab_feats=tab_feats, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_dataset = ICIFusionDataset(df=pkl_file, phase='val', tab_feats=tab_feats, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    test_dataset = ICIFusionDataset(df=pkl_file, phase='test', tab_feats=tab_feats, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

def ici_dataloader(pkl_file, tab_feats, batch_size, transform, num_workers):
    # Training dataset
    train_dataset = ICIFusionDataset(df=pkl_file, phase='train', tab_feats=tab_feats, transform=transform)

    # Extract class labels from the dataset
    train_labels = [sample[1] for sample in train_dataset]  # assuming __getitem__ returns (img, feats, label)
    num_classes = len(set(train_labels))


    # Custom sampler
    train_sampler = ClassBalancedBatchSampler(
        labels=train_labels,
        batch_size=batch_size,
    )
    #sampler = BalancedBatchSampler(dataset=train_dataset, batch_size=batch_size)


    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True
    )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     sampler=sampler,
    #     pin_memory=True,
    #     num_workers=num_workers,
    #     prefetch_factor=2,
    #     persistent_workers=True
    # )


    # Common transform for val/test
    val_test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_dataset = ICIFusionDataset(df=pkl_file, phase='val', tab_feats=tab_feats, transform=val_test_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    test_dataset = ICIFusionDataset(df=pkl_file, phase='test', tab_feats=tab_feats, transform=val_test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

def test_ici_dataloader(pkl_file, tab_feats, batch_size, transform, num_workers):
    # Common transform for val/test
    val_test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    test_dataset = ICIFusionDataset(df=pkl_file, phase='test', tab_feats=tab_feats, transform=val_test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, prefetch_factor=2, persistent_workers=True)

    return {'test': test_dataloader}

class ICIFusionDataset(Dataset):
    def __init__(self, df, phase, tab_feats, transform):

        #f = pd.read_csv(df)
        #df = df[df['split'] == phase]
        df = df['test']
        df.reset_index(inplace=True, drop=True)
        self.df = df

        self.tab_feats = tab_feats

        self.transform = transform

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #self.df['path'] = self.df['path'].astype(str)
        #file_path = self.df.iloc[idx]['path']
        file_path = row['png_path']
        #file_path = str(self.df.iloc[idx]['path']).strip()
        



        #file_path = row['path']
        #label = row['labelMACE_1yr']
        try:
            image = Image.open(file_path).convert('RGB')
        except:
            print(f"file_path: {file_path} | type: {type(file_path)}")

        image = np.array(image)
        og_img = image
        orig_image = torch.from_numpy(og_img.transpose(2,0,1)).float()  # (C,H,W)


        for c in range(3):
            channel = image[..., c]
            min_val = channel.min()
            max_val = channel.max()
            image[..., c] = (channel - min_val) / (max_val - min_val + 1e-8) #* 255

        frame = self.transform(image=image)['image']

        #pt_id = int(row['MRN'])
        #self.tab_feats['MRN'] = self.tab_feats['MRN'].astype(int)

        #feat_df = self.tab_feats[self.tab_feats['MRN'] == pt_id]
        #feats = feat_df.iloc[0, 1:].values.astype(np.float32)

        feats = self.tab_feats.iloc[0, :].values.astype(np.float32)
        feats_tensor = torch.tensor(feats, dtype=torch.float32)

        return frame, feats_tensor

        #return frame, label, pt_id, feats_tensor, orig_image, file_path
    
