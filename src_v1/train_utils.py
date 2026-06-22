import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision
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

def train_model(model, dataloader, num_epochs, batch_size, optimizer, patience, save_path, model_name, device):

    train_losses = []
    val_losses = []
    early_counter = 0
    best_loss = float('inf')
    early_stop = False
    scheduler = CyclicLR(optimizer, base_lr=1e-8, max_lr=1e-4, step_size_up=1300)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    visualized = True
    optimizer.zero_grad()

    for epoch in range(num_epochs):

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            total_loss = 0.0  
            num_samples = 0
            visualized = True

            for batch, ecg, labels, _ in tqdm(dataloader[phase], desc=f"Epoch {epoch+1} - {phase}"):

                ##dynamic weights
                # Compute class weights for current batch
                # classes = torch.unique(labels)
                # weights = compute_class_weight(
                # class_weight='balanced',
                # classes=classes.cpu().numpy(),
                # y=labels.cpu().numpy()
                # )
                # weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)

                # # Match weights to class indices
                # class_to_weight = dict(zip(classes.tolist(), weights))
                # full_weight_tensor = torch.ones(4, device=device)
                # for cls, w in class_to_weight.items():
                #     full_weight_tensor[cls] = w

                # Create new loss function for this batch
                #criterion = nn.CrossEntropyLoss(weight=full_weight_tensor)
                

                #optimizer.zero_grad()

                labels = labels.to(device)
                batch = batch.to(device)
                ecg = ecg.to(device)

                if phase == 'val':
                    with torch.no_grad():
                        out = model(batch, ecg)
                else:
                    out = model(batch, ecg)

                loss = criterion(out, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()*batch_size
                num_samples += batch_size

            avg_loss = total_loss / num_samples

            if phase == 'train':
                train_loss = avg_loss
                train_losses.append(train_loss)
            else:
                val_loss = avg_loss
                val_losses.append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model.load_state_dict(best_model_wts)
                    #torch.save(model.state_dict(), save_path + '{}_fusion_best.pth'.format(model_name))
                    torch.save(model, save_path + '{}_fusion_best.pth'.format(model_name))
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
        torch.cuda.empty_cache()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    #writer.close()
    # Plot loss curves (train and val)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + '{}_fusion_loss_curves.png'.format(model_name))
    return model

def model_predict(model, dataloader, patch_size):
    device = 'cuda:3'

    #if torch.cuda.is_available():
    model = model.to(device)
    #model = nn.DataParallel(model, device_ids=[0])  # This wraps the model to use all GPUs on the machine

    model.eval()

    true_labels = []
    score_0 = []
    score_1 = []
    score_2 = []
    score_3 = []
    pt_ids = []

    for batch, ecg, label, pt_id in tqdm(dataloader['test']):

        batch = batch.to(device)
        ecg = ecg.to(device)

        output = model(batch, ecg)

        softmax = nn.Softmax()
        scores = softmax(output)

        score_0_batch = scores[:, 0]
        score_1_batch = scores[:, 1]
        score_2_batch = scores[:, 2]
        score_3_batch = scores[:, 3]
        true_labels.extend(label)
        score_0.extend(score_0_batch.tolist())
        score_1.extend(score_1_batch.tolist())
        score_2.extend(score_2_batch.tolist())
        score_3.extend(score_3_batch.tolist())
        pt_ids.extend(pt_id)

    return pt_ids, true_labels, score_0, score_1, score_2, score_3 