import torch 
from torch.utils.data import Dataset
import os 
import numpy as np 
from pathlib import Path 



class PointCloudDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor_path = self.files[idx]
        point_cloud = torch.load(tensor_path)

        # Transpose to match the DGCNN input format: (num_dims, num_points)
        point_cloud = point_cloud.transpose(0, 1)  # Change shape from (2048, 3) to (3, 2048)

        # print(self.files[idx])
        # print(tensor_path.split(".pt")[0].split("fid")[-1])

        label = tensor_path.split(".pt")[0].split("fid")[-1]
        
        return point_cloud, torch.tensor(int(label))





if __name__ == "__main__":
    train_dataset = PointCloudDataset(data_dir='./data/pcd_tensors')
    print(train_dataset[0])