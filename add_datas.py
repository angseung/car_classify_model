import os
import shutil
import json
from PIL import Image

base_dir = "./data/train/Addon"
additional_class_labels = os.listdir(base_dir)

new_labels = ["Hyundai Santafe", "Fiesta hatchback"]

with open("./data/train/labels/meta.json") as meta_file:
    meta_dict = json.load(meta_file)

with open("./data/train/labels/annotations/0a501a4dbe2f4e.json") as f:
    sample_dict = json.load(f)

meta_dict["class_names"] = meta_dict["class_names"] + new_labels

# label 6 Hyundai Santafe
img_list_label_6 = os.listdir(f"{base_dir}/{additional_class_labels[0]}")

for fname in img_list_label_6:
    json_file_name = f"{fname[:-3]}json"
    # with open(f"./data/train/labels/annotations/{json_file_name}", "w") as f:
    with open(f"./data/train/Addon/{json_file_name}", "w") as f:
        img = Image.open(f"{base_dir}/{additional_class_labels[0]}/{fname}")
        sample_dict["image"]["width"] = img.width
        sample_dict["image"]["height"] = img.height
        sample_dict["image"]["height"] = img.height
        sample_dict["image"]["file_name"] = fname
        sample_dict["instances"][0]["category_id"] = 6
        sample_dict["image"]["id"] = 6
        json.dump(sample_dict, f)

# label 7 Fiesta hatchback
img_list_label_7 = os.listdir(f"{base_dir}/{additional_class_labels[1]}")

for fname in img_list_label_7:
    json_file_name = f"{fname[:-3]}json"
    # with open(f"./data/train/labels/annotations/{json_file_name}", "w") as f:
    with open(f"./data/train/Addon/{json_file_name}", "w") as f:
        img = Image.open(f"{base_dir}/{additional_class_labels[1]}/{fname}")
        sample_dict["image"]["width"] = img.width
        sample_dict["image"]["height"] = img.height
        sample_dict["image"]["height"] = img.height
        sample_dict["image"]["file_name"] = fname
        sample_dict["instances"][0]["category_id"] = 7
        sample_dict["image"]["id"] = 7
        json.dump(sample_dict, f)
