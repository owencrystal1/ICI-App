import os
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torchvision
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from torch.optim.lr_scheduler import CyclicLR
from sklearn.utils.class_weight import compute_class_weight

def unnormalize(tensor, mean, std):
    """
    tensor: a [C, H, W] tensor (not batched)
    mean, std: lists of 3 values
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

from sklearn.metrics import roc_auc_score

def train_model(model, dataloader, num_epochs, batch_size, optimizer, patience, save_path, model_name, device):

    train_losses = []
    val_losses = []
    train_aurocs, val_aurocs = [], []
    early_counter = 0
    best_loss = float('inf')
    early_stop = False
    #scheduler = CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-4, step_size_up=1300)
    #weights = torch.tensor([1, 19.11]) #19.11
    weights = torch.tensor([2.52, 10.13, 2.68, 6.43]) #19.11
    weights = weights.to(device)
    device_ids = [0,3]
    root = torch.device(f'cuda:{device_ids[0]}')
    #model = nn.DataParallel(model, device_ids=device_ids).to(root)

    model = model.to(root)
    criterion = nn.CrossEntropyLoss(weight=weights)

    visualized = True
    #optimizer.zero_grad()
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            total_loss = 0.0  
            num_samples = 0
            all_probs = []
            all_labels = []

            if epoch == 0:
                visualized = True
            else:
                visualized = False
            # (image, ecg_image, label, resid)
            for ecg_image, labels, pt_id in tqdm(dataloader[phase], desc=f"Epoch {epoch+1} - {phase}"): #image, ecg_image, label, resid

                #echo  = echo.to(root,  non_blocking=True)
                labels = labels.to(root, non_blocking=True)
                #image = image.to(root, non_blocking=True)
                ecg_image = ecg_image.to(root, non_blocking=True)

                if phase == 'train':
                    optimizer.zero_grad(set_to_none=True)
                
                with torch.set_grad_enabled(phase == 'train'):
                    #out = model(tabs, ecg)
                    out = model(ecg_image)
                    loss = criterion(out, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

                #total_loss += loss.item()*batch_size
                #num_samples += batch_size

                actual_batch_size = labels.size(0)
                total_loss += loss.item() * actual_batch_size
                num_samples += actual_batch_size

                # --- Collect predictions for AUROC ---
                # probs = F.softmax(out, dim=1)[:, 1].detach().cpu().numpy()  # positive class prob
                # all_probs.extend(probs)
                # all_labels.extend(labels.detach().cpu().numpy())

            avg_loss = total_loss / num_samples
            #auroc = roc_auc_score(all_labels, all_probs)

            if phase == 'train':
                train_loss = avg_loss
                train_losses.append(train_loss)
                #train_aurocs.append(auroc)
            else:
                val_loss = avg_loss
                val_losses.append(val_loss)
                #val_aurocs.append(auroc)


                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model.load_state_dict(best_model_wts)
                    torch.save(model, save_path + '{}_best.pth'.format(model_name))
                    print(f"Best model saved at epoch {epoch+1}")
                    early_counter = 0
                else:
                    early_counter +=1
                    if early_counter >= patience:
                        print(f'Early stopping triggered after {patience} epochs without improvement.')
                        early_stop = True
                        break
        if early_stop:
            break  # break from epoch loop
        #torch.cuda.empty_cache()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        #print(f"Epoch [{epoch+1}/{num_epochs}], "
        #      f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
        #      f"Train AUROC: {train_aurocs[-1]:.4f}, Val AUROC: {val_aurocs[-1]:.4f}")

    # Plot loss curves (train and val)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + '{}_loss_curves.png'.format(model_name))

    # plt.figure(figsize=(10, 6))
    # plt.plot(train_aurocs, label='Training Loss')
    # plt.plot(val_aurocs, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('AUROC')
    # plt.title('Training and Validation Losses')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(save_path + '{}_AUROC.png'.format(model_name))
    return model


def occlusion_sensitivity(model, tab_x, img_x_resized, patch_size=8, stride=8, 
                          target_class=1, baseline=0.0, orig_img=None):
    """
    Occlusion sensitivity with optional mapping back to original image size.

    model: ECGFusionModel
    tab_x: (1, n_features)
    img_x_resized: (1, C, H, W) -> resized/padded tensor for model
    orig_img: (1, C, H_orig, W_orig) -> optional original image tensor for visualization
    """
    model.eval()
    _, _, H, W = img_x_resized.shape
    sensitivity_map = torch.zeros((H, W))

    # --- baseline prediction ---
    with torch.no_grad():
        logits = model(tab_x, img_x_resized)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        base_score = logits[0, target_class].item()

    # --- sliding window occlusion ---
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            img_occ = img_x_resized.clone()

            # Mask patch
            y1, y2 = y, min(y+patch_size, H)
            x1, x2 = x, min(x+patch_size, W)
            img_occ[:, :, y1:y2, x1:x2] = baseline

            with torch.no_grad():
                #score = model(tab_x, img_occ)[0, target_class].item()
                probs = F.softmax(model(tab_x, img_occ), dim=1)
                score = probs[0, target_class].item()


            diff = base_score - score


            sensitivity_map[y1:y2, x1:x2] += diff

    # Normalize
    #print(sensitivity_map.min(), sensitivity_map.max())
    sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-6)

    # --- resize back to original image size if provided ---
    if orig_img is not None:
        _, _, H_orig, W_orig = orig_img.shape
        sensitivity_map = F.interpolate(sensitivity_map.unsqueeze(0).unsqueeze(0),
                                        size=(H_orig, W_orig),
                                        mode="bilinear",
                                        align_corners=False).squeeze()

    return sensitivity_map  # (H, W) or (H_orig, W_orig) if orig_img given


def show_overlay(img, saliency, threshold=0.0, save_path=None, title=None):

    # Create mask: only keep values above threshold
    mask = saliency.cpu().numpy()
    alpha = (mask > threshold).astype(float)  # transparency mask

    plt.imshow(img, cmap="gray")
    plt.imshow(mask, cmap="jet", alpha=alpha * 0.4)  # overlay only where mask > threshold
    if title:
        plt.title(title)
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def model_predict(model, dataloader, patch_size, params=None):
    device = 'cuda:1'

    #if torch.cuda.is_available():
    model = model.to(device)
    #model = nn.DataParallel(model, device_ids=[0])  # wraps the model to use all GPUs on the machine

    model.eval()

    true_labels = []
    score_0, score_1, score_2, score_3 = [], [], [], []
    pt_ids = []
    visualized = False
    save_saliency=False

    #for ecg, label, pt_id, tabs, orig_img, ecg_file_path in tqdm(dataloader['test']):
    for ecg_image, label, pt_id in tqdm(dataloader['test']):

        #image = image.to(device)
        ecg_image = ecg_image.to(device)

        with torch.no_grad():
            output = model(ecg_image)

        softmax = nn.Softmax()
        scores = softmax(output)

        score_0_batch = scores[:, 0]
        score_1_batch = scores[:, 1]
        true_labels.extend(label.cpu().tolist())
        score_0.extend(score_0_batch.tolist())
        score_1.extend(score_1_batch.tolist())
        pt_ids.extend(pt_id)

        #if params['num_class'] == 4:
        score_2_batch = scores[:, 2]
        score_3_batch = scores[:, 3]
        score_2.extend(score_2_batch.tolist())
        score_3.extend(score_3_batch.tolist())
        
        # if save_saliency:
        #     # Run without no_grad to allow gradients
        #     ecg.requires_grad_(True)
        #     output = model(tabs, ecg)
        #     preds = output.argmax(dim=1)
        #     #saliency = generate_saliency_map(model, tabs, ecg, orig_img=orig_img, target_class=preds)
        #     saliency = occlusion_sensitivity(model, tabs, ecg, target_class=label, orig_img=orig_img)

        #     #for i in range(ecg.size(0)):  # per image in batch
        #     img = ecg[0].detach().cpu()

        #     original_ecg = Image.open(ecg_file_path[0]).convert('RGB')
        #     original_ecg = np.array(original_ecg).astype(np.float32)/255.0

        #     img = unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     img = img.clamp(0, 1)
        #     sal = saliency
        #     save_path = f"saliency_maps5/pt_{pt_id.item()}.png"
        #     show_overlay(original_ecg, sal, save_path=save_path, title=f"ID {pt_id.item()}")
            
    df = pd.DataFrame()
    df['pt_id'] = pt_ids
    df['label'] = true_labels
    df['score_0'] = score_0
    df['score_1'] = score_1

    # if params['num_class'] == 4:
    df['score_2'] = score_2
    df['score_3'] = score_3

    return df