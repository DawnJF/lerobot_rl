import numpy
import os
import json
import numpy as np
import shutil

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_json(path):
    with open(path, "r", encoding="utf-8") as fr:
        desc = json.load(fr)
    
    return desc

def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(obj, fw, ensure_ascii=False)

def copy_a_file(src_path, target_path):
    shutil.copy2(src_path, target_path)

root = "/media/robot/30F73268F87D0FEF/Hil_Serl_Dataset/0820"
cube_paths = os.listdir(root)
for cube_path in cube_paths:
    # process data.json
    data_json_file_path = os.path.join(root, cube_path, "data.json")
    copy_a_file(data_json_file_path, os.path.join(root, cube_path, "data_bak.json"))
    data_list = read_json(data_json_file_path) # list[dict]
    for index in range(1, len(data_list)):
        position = np.array(data_list[index]["pose"]["position"])   # numpy
        position_prev = np.array(data_list[index-1]["pose"]["position"])   # numpy
        position_true = (position - position_prev).tolist()   # list
        data_list[index-1]["pose"]["position"] = position_true
    write_json(data_list, data_json_file_path)

    # process images folders
    images_folder_path = os.path.join(root, cube_path, "image")
    cameras = os.listdir(images_folder_path)
    '''
    cameras: depth  rgb  scene  wrist
    '''
    for camera in cameras:
        images_folder_path_camera = os.path.join(images_folder_path, camera)
        images = os.listdir(images_folder_path_camera)
        num_images = len(images) - 1
        if camera in ['depth', 'rgb']:
            resolution = '960x540'
        else:
            resolution = '640x480'
        folder_name = f"{camera}_{resolution}_{num_images}"
        delete_image_path = os.path.join(images_folder_path_camera, folder_name)
        if os.path.exists(delete_image_path):
            os.remove(delete_image_path)
        else:
            print(f"{delete_image_path} does not exist")

