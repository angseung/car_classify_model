import os
import json
import shutil

base_dir = "./data/train/"
target_dir = "./data/train_imbalanced"
num_samples_per_label = [100, 50, 30, 90, 60, 70, 40, 110]
num_samples_per_label_done = [0] * len(num_samples_per_label)
total_samples = sum(num_samples_per_label)

if not os.path.isdir(target_dir):
    os.makedirs(target_dir)

for json_file in os.listdir(f"{base_dir}/labels/annotations"):
    with open(f"{base_dir}/labels/annotations/{json_file}") as f:
        json_info = json.load(f)
        f_name = json_info["image"]["file_name"]
        img_file = f"{base_dir}/images/{f_name}"
        curr_label = json_info["instances"][0]["category_id"]

        if num_samples_per_label[curr_label] > num_samples_per_label_done[curr_label]:
            if not os.path.isdir(f"{target_dir}/images"):
                os.makedirs(f"{target_dir}/images")

            if not os.path.isdir(f"{target_dir}/labels/annotations"):
                os.makedirs(f"{target_dir}/labels/annotations")

            shutil.copy(
                f"{base_dir}/labels/annotations/{json_file}",
                f"{target_dir}/labels/annotations/{json_file}",
            )
            shutil.copy(f"{base_dir}/images/{f_name}", f"{target_dir}/images/{f_name}")

            num_samples_per_label_done[curr_label] += 1

with open(f"{base_dir}/labels/meta.json") as f:
    json_meta = json.load(f)
    json_meta["num_images"] = total_samples

with open(f"{target_dir}/labels/meta.json", "w") as f:
    json.dump(json_meta, f)
