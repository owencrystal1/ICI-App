import os
import cv2
import pickle
import torch # type: ignore
import torch.nn as nn # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import torchvision.transforms as transforms # type: ignore
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import binary_fill_holes
from skimage import exposure
import torch.nn.functional as F
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

def show_cam_on_image_thresholded(img: np.ndarray, heatmap: np.ndarray, threshold: float = 0.5, alpha: float = 0.5) -> np.ndarray:
    """
    Custom GradCAM overlay with thresholding.
    
    Args:
        img: Original image in float range [0, 1], shape (H, W, 3).
        heatmap: GradCAM heatmap in range [0, 1], shape (H, W).
        threshold: Threshold value for heatmap. Only values > threshold will be overlaid.
        alpha: Blending factor for overlay.
        
    Returns:
        Overlay image, float range [0, 1], shape (H, W, 3).
    """
    # Resize heatmap to match image shape
    h, w = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Apply threshold
    heatmap_thresh = np.where(heatmap_resized > threshold, heatmap_resized, 0)

    # Convert to uint8 for colormap
    heatmap_uint8 = np.uint8(255 * heatmap_thresh)

    # Apply JET colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Convert original image to uint8
    img_uint8 = np.uint8(img * 255)

    # Create output overlay (start as copy of original)
    overlay = img_uint8.copy()

    # Mask for non-zero heatmap
    mask = heatmap_thresh > 0

    # Blend only on mask
    # Create mask 3 channels
    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Blend entire heatmap image and original image
    blended = cv2.addWeighted(heatmap_color, alpha, img_uint8, 1 - alpha, 0)

    # Assign blended only where mask
    overlay = np.where(mask_3c, blended, img_uint8)


    # Convert back to float range
    overlay = overlay / 255.0

    return overlay


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


def gen_gradcam(model_path, dataloader, save_path, file_name):
    input_size = 224
    batch_size = 1
    num_workers = 16
    cuda_num = 0

    device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() else 'cpu')

    model = torch.load(model_path, map_location=device)
    model = model.to(device)

    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataloaders_dict = dataloader

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = None

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
    ])
    final_png_paths = []
    final_gcam_paths = []
    final_aggregate_mask = None  # Will hold the OR of all masks


    for i, (input_tensor) in enumerate(dataloaders_dict):
        img = invTrans(input_tensor[0].cpu())
        img = img.detach().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        img_uint8 = (img * 255).astype(np.uint8)
        mask = (img_uint8 > 30)[:, :, 1] 
        filled_mask = binary_fill_holes(mask)  # Returns a boolean array

        filled_mask_uint8 = (filled_mask.astype(np.uint8)) * 255

        # Use a small kernel to avoid over-expansion
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(filled_mask_uint8, kernel, iterations=1)
        final_mask = dilated_mask > 0  # boolean mask again
        final_mask = binary_fill_holes(final_mask)

        if final_aggregate_mask is None:
            final_aggregate_mask = final_mask
        else:
            final_aggregate_mask |= final_mask  # logical OR

    for i, (input_tensor) in enumerate(dataloaders_dict):
        
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        # Get the CAM (shape: [B, H, W])
        grayscale_cams = cam(input_tensor=input_tensor)

        img = invTrans(input_tensor[0].cpu())
        img = img.detach().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        # Overlay CAM
        #binary_mask = cv2.resize(binary_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
        img_uint8 = (img * 255).astype(np.uint8)

        # Convert boolean mask to uint8
        #final_mask_uint8 = (final_aggregate_mask.astype(np.uint8)) * 255

        # Ensure the mask is uint8
        mask_uint8 = (final_aggregate_mask.astype(np.uint8)) * 255

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

        # Skip background (label 0), find the largest component
        if num_labels > 1:
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # +1 to skip background
            largest_mask = (labels == largest_component).astype(np.uint8) * 255
        else:
            largest_mask = mask_uint8  # fallback if only background

        # Apply erosion with a small kernel to smooth edges and remove noise
        kernel = np.ones((5, 5), np.uint8)  # You can try (5, 5) if more aggressive smoothing is needed
        eroded_mask = cv2.erode(largest_mask, kernel, iterations=1)

        # Convert back to boolean if needed
        final_cleaned_mask = eroded_mask > 0


        cam_overlay = show_cam_on_image(img, grayscale_cams[0]*final_cleaned_mask, use_rgb=True)
        #cam_overlay = show_cam_on_image(img, grayscale_cams[0], use_rgb=True)

        # Save as PNG
        gcam_file = os.path.join(save_path, f"{file_name}_gradcam_{i}.png")
        Image.fromarray(cam_overlay).save(gcam_file)

        save_file = os.path.join(save_path, f"{file_name}_{i}.png")
        
        Image.fromarray(img_uint8).save(save_file)

        #Image.fromarray(img).save(save_file)
        #Image.fromarray(cam_overlay).resize((448, 448), Image.BILINEAR).save(save_file)

        final_gcam_paths.append(gcam_file)
        final_png_paths.append(save_file)
        
    
    print('Saved GradCAMs for', len(final_png_paths), 'frames.')

    return final_png_paths, final_gcam_paths

def gen_ecg_gradcam(model_path, dataloader, original_ecg):
    input_size = 224
    batch_size = 1
    num_workers = 16
    cuda_num = 0

    ecg_image = np.array(original_ecg)
    original_ecg_shape = ecg_image.shape
    h, w = original_ecg_shape[:2]

    device = torch.device('cuda:{}'.format(cuda_num) if torch.cuda.is_available() else 'cpu')

    try:
        model = model_densenet(4)
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict)
    except:
        model = torch.load(model_path, map_location=device)

    transform = A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


    #target_layers = [model.features.denseblock4[-1]]
    last_key = list(model.features.denseblock4._modules.keys())[-1]

    # Get the corresponding layer
    target_layer = model.features.denseblock4._modules[last_key]

    # Define target_layers
    target_layers = [target_layer]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = None

    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                        std = [ 1., 1., 1. ]),
    ])
    final_png_paths = []
    final_gcam_paths = []

    try: 
        dataloaders_dict = dataloader['test']
    except:
        dataloaders_dict = dataloader

    for i, (input_tensor) in enumerate(dataloaders_dict):
        
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        grayscale_cams = cam(input_tensor=input_tensor)

        img = invTrans(input_tensor[0].cpu())
        img = img.detach().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        #ecg_image = np.clip(ecg_image, 0, 1)
        #print(ecg_image.shape, ecg_image.dtype, ecg_image.min(), ecg_image.max())

        ecg_image = ecg_image.astype(np.float32)
        #print(ecg_image.shape, ecg_image.dtype, ecg_image.min(), ecg_image.max())
        if ecg_image.max() > 1.0:
            ecg_image = ecg_image / 255.0


        #heatmap = cv2.resize(grayscale_cams[0], (w, h))
        overlay = show_cam_on_image_thresholded(ecg_image, grayscale_cams[0], threshold=0.5, alpha=0.5)
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        gradcam_viz = Image.fromarray(overlay_uint8)

        return gradcam_viz
