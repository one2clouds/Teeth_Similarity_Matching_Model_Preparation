import os
import torch
import open3d as o3d
import numpy as np
import glob
import argparse
from pathlib import Path
from tqdm import tqdm


def preprocess_and_save_obj(file_path, save_path, num_points=2048):
    """ Preprocess the OBJ file into a point cloud and save as a tensor. """
    # Generate the tensor file path
    # print(file_path)
    # print(file_path.split("/")[-2])


    filename = f"{file_path.split('/')[-2]}_________" + os.path.basename(file_path).replace('.stl', '.pt')
    # print(filename)
    # print(filename)
    tensor_path = os.path.join(save_path, filename)


    # print(tensor_path)
    
    if os.path.exists(tensor_path):
        # If the tensor file already exists, load it
        points = torch.load(tensor_path)
        # print(f"Loaded existing tensor for {file_path}")
    else:
        # Preprocess the point cloud
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

        #normalize PCD
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        points -= centroid  # Centering
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points /= max_distance  # Scaling
        pcd.points = o3d.utility.Vector3dVector(points)
        # visualize_pcd_in_notebook(pcd) #visualize

        #saving point_cloud
        points = np.asarray(pcd.points)
        points = torch.tensor(points, dtype=torch.float32)
        
        # Save the tensor
        torch.save(points, tensor_path)
        # print(f"Saved tensor file : {tensor_path}")
    
    return points


def main():
    file_names = sorted(glob.glob('/home/shirshak/STL_client_data/*/*.stl'))
    save_path = Path('/home/shirshak/00_teeth_similarity_matching/client_pcd_tensors/')
    # file_names = glob.glob('/home/shirshak/Teeth3DS_individual_teeth/individual_teeth/*.obj')
    # save_path = Path('/home/shirshak/Teeth3DS_individual_teeth/pcd_tensors/')
    # os.makedirs(save_path, exist_ok=True)

    # print(len(file_names)) # 24000
    for file_name in tqdm(file_names):
        preprocess_and_save_obj(file_name, save_path, num_points=2048)

if __name__ == "__main__":
    main()

