import torch 
import torch.nn as nn 
from glob import glob 

import sys 
import os 
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dgcnn import DGCNN 

from tqdm import tqdm 
import json




if __name__ == "__main__":

    num_classes = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DGCNN(output_channels=num_classes)
    # model


    checkpoint = torch.load("/home/shirshak/00_teeth_similarity_matching/model_ckpt/best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Removing last layers with Identity which will not perform any operations 
    model.dp2 = nn.Identity()
    model.linear3 = nn.Identity()
    model = model.to(device)



    # LOAD THE POINT CLOUDS     
    point_cloud_file_names = sorted(glob("/home/shirshak/00_teeth_similarity_matching/client_pcd_tensors/*"))

    mesh_location = "/home/shirshak/STL_client_data/"

    count = 0
    data_json = []

    for file_name in tqdm(point_cloud_file_names): 
        point_cloud = torch.load(file_name)

        point_cloud = point_cloud.transpose(0, 1).unsqueeze(0)  # Change shape from (2048, 3) to (3, 2048), and unsqueeze to (1,3,2048)

        # print(point_cloud.shape) # (1,3,2048)

        model.eval()
        with torch.no_grad():
            point_cloud = point_cloud.to(device)
            feature_256 = model(point_cloud)
        
        feature_256 = feature_256.cpu().numpy().flatten()
        feature_256 = feature_256.tolist()

        file_name = os.path.basename(file_name).replace('.pt', '.stl')
        final_mesh_location = mesh_location + os.path.basename(file_name).split('_________')[0] + f"/{os.path.basename(file_name).split('_________')[1]}"

        # print(final_mesh_location)


        data_json.append({
            "mesh_location":final_mesh_location,
            # 'thumbnail_location': thumbnail_location,
            "label": os.path.basename(file_name).split('_________')[0],
            "feature_vector": feature_256
        })


    json_filename = "feature_info_client_data.json"

    with open(json_filename, "w") as f:
        json.dump(data_json, f)

