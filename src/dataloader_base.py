import os
import random
import pickle
import pandas as pd
from PIL import Image
from src.cardiac_echo_processes import *
from skimage import exposure
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import default_collate

def add_banner(banner, target_image, loc='top'):

    original_banner_height = banner.height  # original banner image

    crop_rows = 18  # how many rows you want from the original banner

    # Resize banner to match the target_image width
    target_width, target_height = target_image.size
    banner_resized = banner.resize((target_width, int(banner.height * target_width / banner.width)), Image.ANTIALIAS)

    # Compute how many pixels correspond to the top 18 rows after resizing
    banner_pixel_height = int((crop_rows / original_banner_height) * banner_resized.height)

    # Crop top of the resized banner
    banner_crop = banner_resized.crop((0, 0, target_width, banner_pixel_height))

    if loc == 'top':
        # Create new padded image and paste both
        padded_image = Image.new("RGB", (target_width, banner_pixel_height + target_height))
        padded_image.paste(banner_crop, (0, 0))
        padded_image.paste(target_image, (0, banner_pixel_height))
    
    elif loc == 'bottom':
        padded_image = Image.new("RGB", (target_width, target_height + banner_pixel_height))
        padded_image.paste(target_image, (0, 0))
        padded_image.paste(banner_crop, (0, target_height))

    elif loc == 'left':
        # Rotate the banner 90° clockwise (so it becomes vertical)
        banner_vert = banner_crop.transpose(Image.ROTATE_270)

        # Create new image with extra width
        padded_image = Image.new("RGB", (banner_vert.width + target_width, target_height))
        padded_image.paste(banner_vert, (0, 0))  # banner on the left
        padded_image.paste(target_image, (banner_vert.width, 0))  # original image shifted right

    elif loc == 'right':
        banner_vert = banner_crop.transpose(Image.ROTATE_90)

        padded_image = Image.new("RGB", (target_width + banner_vert.width, target_height))
        padded_image.paste(target_image, (0, 0))  # image on left
        padded_image.paste(banner_vert, (target_width, 0))  # banner on right

    return padded_image


def makeWeightedSampler(dataset):
    """makes a weighted sampler based on the input dataset"""
    # Compute class weight each class should get the same weight based on its balance
    temp_df = dataset.label
    # get the count of each label
    benCount = temp_df.label[temp_df.label == 0].count()
    malCount = temp_df.label[temp_df.label == 1].count()
    # get weight of each label
    benWeight = 1. / benCount
    malWeight = 1. / malCount
    #print('sliceweights: ben:{}, mal:{}'.format(benWeight, malWeight))
    # make a tensor of the weights
    sample_weights = []
    for label in temp_df.label:
        if label == 0:
            sample_weights.append(benWeight)
        else:
            sample_weights.append(malWeight)
    sample_weights = torch.tensor(sample_weights)
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def load_ecg_signals(directory, focused_leads):
        with open(directory, 'rb') as f:
            data = pickle.load(f)
            nums = data.values()
            all_leads = [data[key] for key in focused_leads]
        return torch.tensor(all_leads, dtype=torch.float32)

def custom_collate(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    return default_collate(batch)  

class NeighborhoodRandomPixelization:
    def __init__(self, percent=0.3, kernel_size=3):
        """
        Args:
            percent (float): Fraction of image pixels (0 to 1) to corrupt via neighborhood pixelization.
            kernel_size (int): Size of square kernel to apply random pixelization.
        """
        self.percent = percent
        self.kernel_size = kernel_size

    def __call__(self, pil_img):
        img = np.array(pil_img)
        h, w, _ = img.shape
        output = img.copy()

        pad = self.kernel_size // 2
        valid_y = np.arange(pad, h - pad)
        valid_x = np.arange(pad, w - pad)

        total_pixels = h * w
        num_to_corrupt = int(self.percent * total_pixels / (self.kernel_size ** 2))

        ys = np.random.choice(valid_y, size=num_to_corrupt, replace=True)
        xs = np.random.choice(valid_x, size=num_to_corrupt, replace=True)

        for y, x in zip(ys, xs):
            gray_patch = np.random.randint(
                0, 256, size=(self.kernel_size, self.kernel_size, 1), dtype=np.uint8
            )
            random_patch = np.repeat(gray_patch, 3, axis=2)
            output[y-pad:y+pad+1, x-pad:x+pad+1] = random_patch
        return Image.fromarray(output)

def ecg_dataloader(params, mode):

    dataset = load_png(params=params, pd_file=params['dataframe_pkl'], phase=mode, view=params['view'],
                        transform=transforms.Compose([
                            transforms.Resize((params['input_size'], params['input_size'])),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]))
    custom_dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                                                num_workers=params['workers'], collate_fn=custom_collate)
    return custom_dataloader



class load_png(Dataset):
    """Custom HeadCT dataset with just class labels
    note:
        since it's a custom dataset we need to use:
        for i, data in enumerate(train_loader, 0):
        to enumerate over the train loaders, and the i is the ith train_loader and the data is the sample we generate
    """

    def __init__(self, params, pd_file, phase, quadrant=0, view=None, transform=None):
        """
        Args:
            pd_file (string): Path to the pickled file with image names and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #df = pd.read_csv(pd_file)
        #df.reset_index(inplace=True, drop=True)
        self.label = pd_file
        self.transform = transform
        self.remove_quadrant = quadrant
        self.params = params

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image_path = os.path.join(self.label['file_path'].iloc[idx])
        image = Image.open(image_path)
        if self.remove_quadrant != 0:
            image_np = crop_external_cardio(np.array(image), quadrant=self.remove_quadrant)
            image = Image.fromarray(image_np)

        image = image.resize((442,442), resample=Image.BILINEAR)

        image = np.array(image)

        hist_norm = self.params['hist']

        #if hist_norm.dtype == np.uint8:
            #hist_norm = hist_norm.astype(np.float32) / 255.0
        matched_image = exposure.match_histograms(image, hist_norm)
        #matched_image = np.clip(matched_image * 255, 0, 255).astype(np.uint8)
        matched_image = Image.fromarray(np.uint8(matched_image))

        #banner = Image.open("/home/owen/Datacenter_storage/Owen/ThickWalls/banners/banner.png").convert("RGB")

        #target_image = matched_image.convert('RGB')

        #padded_image = add_banner(banner, target_image, loc='left')

        #image = padded_image
        image = matched_image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        sample = image

        return sample