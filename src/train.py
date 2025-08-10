import sys
import os

# Add the parent directory to the system path to resolve the module import issue
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from models.dgcnn import DGCNN  # Assuming the model class is saved as model.py
from data_preprocess.point_cloud_dataset import PointCloudDataset
import wandb
import os
from torch.utils.data import DataLoader, random_split

import glob
import numpy as np
from tqdm import tqdm

def get_metrics_score(labels, predicted): 
    assert isinstance(labels, list)
    assert isinstance(predicted, list)
    
    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, zero_division=0.0, average='micro')
    recall = recall_score(labels, predicted, zero_division=0.0, average='micro')
    f1 = f1_score(labels, predicted, zero_division=0.0, average='micro')
    
    return accuracy, precision, recall, f1



def train(model, train_loader, val_loader, device, labels_to_idx, idx_to_labels, epochs, patience=50, min_delta=1e-4,):
    checkpoint_dir = "model_ckpt/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        loss_val_epoch, acc_train_epoch, precision_train_epoch, recall_train_epoch, f1_train_epoch  = [], [], [], [], []
        model.train()
        # Training loop
        for batch in tqdm(train_loader):
            inputs, labels = batch[0].to(device), batch[1]
            # print(labels)
            # print([labels_to_idx[label.item()] for label in labels])
            labels = torch.tensor([labels_to_idx[label.item()] for label in labels], device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(labels.shape)
            # print(outputs.max())
            # print(outputs.min())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(outputs.shape)
            predicted_train = nn.Softmax(dim=1)(outputs)
            _, predicted_train = torch.max(predicted_train, 1)

            acc_batch_train, precision_batch_train, recall_batch_train, f1_batch_train = get_metrics_score(labels.cpu().detach().tolist(), predicted_train.cpu().detach().tolist())

            loss_val_epoch.append(loss.cpu().detach().tolist())
            acc_train_epoch.append(acc_batch_train)
            precision_train_epoch.append(precision_batch_train)
            recall_train_epoch.append(recall_batch_train)
            f1_train_epoch.append(f1_batch_train)
        

        wandb.log({"train/train_loss": np.mean(loss_val_epoch), "train/train_epoch":epoch+1, 'train/train_acc':np.mean(acc_train_epoch), 'train/train_precision':np.mean(precision_train_epoch), 'train/train_recall':np.mean(recall_train_epoch), 'train/train_f1':np.mean(f1_train_epoch)})

        # Validation
        model.eval()
        with torch.no_grad():
            loss_val_epoch, acc_val_epoch, precision_val_epoch, recall_val_epoch, f1_val_epoch  = [], [], [], [], []
            for inputs, labels in tqdm(val_loader):
                inputs, labels = batch[0].to(device), batch[1]
                # print(labels)
                # print([labels_to_idx[label.item()] for label in labels])
                labels = torch.tensor([labels_to_idx[label.item()] for label in labels], device=device)

                outputs = model(inputs)
                # print(labels.shape)
                # print(outputs.max())
                # print(outputs.min())
                loss = criterion(outputs, labels)

                # print(outputs.shape)
                predicted_val = nn.Softmax(dim=1)(outputs)
                _, predicted_val = torch.max(predicted_val, 1)

                acc_batch_val, precision_batch_val, recall_batch_val, f1_batch_val = get_metrics_score(labels.cpu().detach().tolist(), predicted_val.cpu().detach().tolist())

                loss_val_epoch.append(loss.cpu().detach().tolist())
                acc_val_epoch.append(acc_batch_val)
                precision_val_epoch.append(precision_batch_val)
                recall_val_epoch.append(recall_batch_val)
                f1_val_epoch.append(f1_batch_val)

            wandb.log({"val/val_loss": np.mean(loss_val_epoch), "val/val_epoch":epoch+1, 'val/val_acc':np.mean(acc_val_epoch), 'val/val_precision':np.mean(precision_val_epoch), 'val/val_recall':np.mean(recall_val_epoch), 'val/val_f1':np.mean(f1_val_epoch)})
    
            # TODO Just to the model Turn  BELOW code this ON
    #         if np.mean(loss_val_epoch) < best_val_loss - min_delta:
    #             best_val_loss = np.mean(loss_val_epoch)
    #             epochs_no_improve = 0
    #             best_checkpoint = {
    #                 "epoch": epoch,
    #                 "model_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #                 "scheduler_state_dict": scheduler.state_dict(),
    #             }
    #             torch.save(best_checkpoint, os.path.join(checkpoint_dir, "best_model.pth"))
    #             print(f"Validation loss improved to {best_val_loss}, saving best model checkpoint...")
    #         else:
    #             epochs_no_improve += 1

    #         # Save checkpoint every 100 epochs
    #         if (epoch + 1) % 100 == 0:
    #             checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
    #             checkpoint = {
    #                 "state_dict": model.state_dict(),
    #                 "optimizer": optimizer.state_dict(),
    #                 "scheduler": scheduler.state_dict(),
    #                 "epoch": epoch
    #             }
    #             torch.save(checkpoint, checkpoint_path)
    #             print(f"Checkpoint saved: {checkpoint_path}")

    #         # Early stopping
    #         if epochs_no_improve >= patience:
    #             print(f"Early stopping triggered after {epoch + 1} epochs with patience of {patience}.")
    #             break

    #         # Step the scheduler after each epoch
    #         scheduler.step()


    # checkpoint_path = os.path.join(checkpoint_dir, f"final_epoch_model.pth")
    # checkpoint = {
    #     "state_dict": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": scheduler.state_dict(),
    #     "epoch": epoch
    # }
    # torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved: {checkpoint_path}")
    
    wandb.finish()




if __name__ == "__main__":

    wandb.init(
        project="teeth_classification", 
        config={
            "epochs":2000,
            "batch_size" : 32,
            "lr" : 0.001, 
            # "dropout":random.uniform(0.01, 0.8)
        }
    )
    trained_model_dir = "model"
    base_dir = "/home/shirshak/Teeth3DS_individual_teeth/pcd_tensors/"
    dataset = PointCloudDataset(base_dir)
    # print(len(dataset)) 23482


    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    


    train_dataloader = DataLoader(train_dataset, batch_size=wandb.config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=wandb.config['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DGCNN(output_channels=32, input_dims=3, k=20, emb_dims=1024, dropout=0.1)
    model.to(device)


    labels_to_idx, idx_to_labels = {}, {}
    labels = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

    for idx, label in enumerate(labels):
        labels_to_idx[label] = idx
        idx_to_labels[idx] = label
    # print(labels_to_idx)
    # print(idx_to_labels)

    train(model, train_dataloader, val_dataloader, device, epochs=wandb.config['epochs'], labels_to_idx=labels_to_idx, idx_to_labels=idx_to_labels)




# For testing purposes moving upper and lower scans of 19 patients from :
# '''
# cd /home/shirshak/Teeth3DS_individual_teeth/pcd_tensors
# mv patient1* patient2* ameziani* baliwish* karklina* mccarthy* 01ENPFHF* 9ZI8RSEP* 9OOBRVB8* 85ARPPFY* YCM36SR6* 7QW884Y4* XNEIPJH8* 6I8A5049* V9CAFAV4* SMF65G9B* PS0WAD4X* 01KK775H* NV3U6JM9* /home/shirshak/Teeth3DS_individual_teeth/pcd_tensors_test/
# 19 patients, 38 upper and lower scans, 518 teeth
# Total : 900 patients, 1800 Intra Oral Scans, 24000 teeth
# '''