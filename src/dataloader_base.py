import os
import random
import pickle
import pandas as pd
from PIL import Image
from skimage import exposure
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataloader import default_collate


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