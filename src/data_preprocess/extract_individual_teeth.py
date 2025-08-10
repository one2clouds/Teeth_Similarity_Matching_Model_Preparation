import glob
import os
import json
import time
import numpy as np


#extact individual teeth from all the mesh in the mesh_paths and save then in data/lower/individual_teeth
fids_lower = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
fids_upper = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]

def extract_teeth(indir, outdir):

    json_file_path = glob.glob(indir + "/*.json")[0]
    mesh_path = glob.glob(indir + "/*.obj")[0]
    # print(mesh_path)
    base_mesh_name =  mesh_path.split("/")[-1]
    base_mesh_name = base_mesh_name.replace(".obj", "")
    json_file = open(json_file_path)
    json_data = json.load(json_file)

    labels = np.array(json_data["labels"])

    with open(mesh_path) as obj_file:
        obj_lines = obj_file.readlines()

    c = 0
    while obj_lines[c][0] != "v":
        c += 1
    v=c
    while obj_lines[c][0] != "f":
        c += 1
    # f = c

    if not os.path.exists(outdir):
            os.makedirs(outdir)

    
    if "lower" in base_mesh_name:
        fids_selection = fids_lower
    elif "upper" in base_mesh_name:
        fids_selection = fids_upper


    for teeth_no in fids_selection:
        start_time = time.time()
        # print("processing teeeth:",teeth_no)
        tooth_to_extract = [teeth_no]

        # vertices_to_extract_index = list()
        # for index, value in enumerate(json_data["labels"]):
        #     if value in tooth_to_extract:
        #         vertices_to_extract_index.append(index)
        
        vertices_to_extract_index = np.where(labels == teeth_no)[0]
        
        if len(vertices_to_extract_index) == 0:
            continue
                   
        # print("^^^^^^^^^^HERE^^^^^^^^^^")
        # print(base_mesh_name)
        f = c
        with open(outdir + "/" + base_mesh_name + f"_fid{teeth_no}.obj", "w") as new_obj_file:
            for i in vertices_to_extract_index:
                new_obj_file.writelines(obj_lines[i + v])
            
            while f < len(obj_lines):
                _, v1, v2, v3 = obj_lines[f].replace("\n", "").split(" ")
                try:
                    v1, v2, v3 = int(v1) - 1, int(v2) - 1, int(v3) - 1
                except:
                    v1, v2, v3 = int(v1.split("//")[0]) - 1, int(v2.split("//")[0]) - 1, int(v3.split("//")[0]) - 1
                l1, l2, l3 = json_data["labels"][v1], json_data["labels"][v2], json_data["labels"][v3]
                if (l1 == l2 and  l2 == l3 and l1 in tooth_to_extract):
                    # """ or (l1 == l2 and l1 in tooth_to_extract and l3 in  tooth_to_extract) or (l1 == l3 and l1 in tooth_to_extract and l2 in  tooth_to_extract):"""
                    # new_v1, new_v2, new_v3 = vertices_to_extract_index.index(v1) + 1, vertices_to_extract_index.index(v2) + 1, vertices_to_extract_index.index(v3) + 1
                    
                    new_v1 = np.where(vertices_to_extract_index == v1)[0][0] + 1
                    new_v2 = np.where(vertices_to_extract_index == v2)[0][0] + 1
                    new_v3 = np.where(vertices_to_extract_index == v3)[0][0] + 1

                    new_obj_file.write(f"f {new_v1} {new_v2} {new_v3}\n")
                f += 1
        end_time = time.time()
        print(f"Time taken for extracting tooth {teeth_no} from {indir}: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    #inputs
    
    mesh_folder_paths = glob.glob("/home/shirshak/Teeth3DS/*/*/")      #input path of the original raw data, no colored, having labels in json files
    outdir = "/home/shirshak/Teeth3DS_individual_teeth/individual_teeth/"       #output folder of the indivial teeth

    for mesh_folder_path in mesh_folder_paths:
        # print(mesh_folder_path)
        patient_id = mesh_folder_path.replace(".obj", "").split("/")[-1]
        # print(patient_id)
        # print(os.path.join(outdir + patient_id))
        start_time = time.time()
        
        extract_teeth(mesh_folder_path, os.path.join(outdir + patient_id))
        end_time = time.time()
        print(f"\n Time taken for extracting all tooth from IOS: {patient_id}: {end_time - start_time:.2f} seconds")