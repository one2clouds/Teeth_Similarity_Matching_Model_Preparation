import argparse
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData
import os

# Mapping of face colors to FDI labels
_teeth_color = {
        (228.0, 228.0, 228.0): 0,
        (255.0, 153.0, 153.0) : 11,
        (255.0, 230.0, 153.0) : 12,
        (204.0, 255.0, 153.0) : 13,
        (153.0, 255.0, 179.0) : 14,
        (153.0, 255.0, 255.0) : 15,
        (153.0, 179.0, 255.0) : 16,
        (204.0, 153.0, 255.0) : 17,
        (255.0, 153.0, 230.0) : 18,
        (255.0, 102.0, 102.0) : 21,
        (255.0, 217.0, 102.0) : 22,
        (179.0, 255.0, 102.0) : 23,
        (102.0, 255.0, 140.0) : 24,
        (102.0, 255.0, 255.0) : 25,
        (102.0, 140.0, 255.0) : 26,
        (179.0, 102.0, 255.0) : 27,
        (255.0, 102.0, 217.0) : 28,
        (255.0, 153.0, 154.0) : 31,
        (255.0, 230.0, 154.0) : 32,
        (204.0, 255.0, 154.0) : 33,
        (153.0, 255.0, 180.0) : 34,
        (153.0, 255.0, 254.0) : 35,
        (153.0, 179.0, 254.0) : 36,
        (204.0, 153.0, 254.0) : 37,
        (255.0, 153.0, 231.0) : 38,
        (255.0, 102.0, 103.0) : 41,
        (255.0, 217.0, 103.0) : 42,
        (179.0, 255.0, 103.0) : 43,
        (102.0, 255.0, 141.0) : 44,
        (102.0, 255.0, 254.0) : 45,
        (102.0, 140.0, 254.0) : 46,
        (179.0, 102.0, 254.0) : 47,
        (255.0, 102.0, 218.0) : 48,
    }

def read_vertices(ply_path: str):
    with open(ply_path, 'rb') as f:
        ply_data = PlyData.read(f)

    vertices = []
    faces = []
    vertex_labels = []

    if 'vertex' in ply_data:
        for vertex in ply_data['vertex']:
            vertices.append([vertex['x'], vertex['y'], vertex['z']])
        vertex_labels = [0] * len(vertices)  # Initialize vertex labels with default (gum)

    if 'face' in ply_data:
        for i, face in enumerate(ply_data['face']):
            faces.append(face['vertex_indices'])
            color = (int(face['red']), int(face['green']), int(face['blue']))
            fdi_number = _teeth_color.get(color, 0)
            
            # Assign label to vertices belonging to this face
            for vertex_idx in face['vertex_indices']:
                vertex_labels[vertex_idx] = fdi_number

    return np.array(vertices), np.array(faces), vertex_labels

def extract_patient_info(file_name):
    parts = file_name.split('_')
    patient_name = parts[0] if len(parts) > 1 else "Unknown"
    jaw_type = "lower" if "lower" in file_name.lower() else "upper" if "upper" in file_name.lower() else "Unknown"
    return patient_name, jaw_type

def save_as_obj(vertices, faces, obj_path):
    with open(obj_path, 'w') as obj_file:
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

def save_as_json(patient_name, jaw_type, vertex_labels, json_path):
    data = {
        "id_patient": patient_name,
        "jaw": jaw_type,
        "labels": vertex_labels
    }
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def process_ply(ply_path: str, out_dir: str) -> bool:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = Path(ply_path)

    if ply_path.is_dir():
        ply_files = [file for file in ply_path.iterdir() if file.is_file() and file.suffix == ".ply"]
    else:
        ply_files = [ply_path]

    if len(ply_files) == 0:
        print("[Error:] No PLY files found in directory.")
        return False

    for file_path in ply_files:
        file_name = file_path.stem
        obj_path = out_dir / f"{file_name}.obj"
        json_path = out_dir / f"{file_name}.json"

        vertices, faces, vertex_labels = read_vertices(str(file_path))
        patient_name, jaw_type = extract_patient_info(file_name)

        save_as_obj(vertices, faces, obj_path)
        save_as_json(patient_name, jaw_type, vertex_labels, json_path)

        print(f"Processed: {file_name} -> OBJ and JSON saved")

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PLY to OBJ and JSON')
    parser.add_argument('--ply_path', type=str, default="sample_obj_conversion",
                        help='Path to PLY file or directory containing PLY files')
    parser.add_argument('--out_dir', type=str, default="sample_obj_conversion",
                        help='Directory where OBJ and JSON will be saved')
    args = parser.parse_args()

    process_ply(args.ply_path, args.out_dir)
    print("Conversion completed.")
